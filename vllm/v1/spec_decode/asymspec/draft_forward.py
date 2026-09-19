# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial, role-local Qwen3.5 draft forwards for AsymSpec.

This is deliberately narrower than a proposer.  It executes exactly one
permanently bound draft view using the metadata assembled for that view; it
does not interact with the V1 scheduler, target runner, or speculative state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import _encode_layer_name

from .cache_binding import AsymSpecDraftCacheBindingRuntime
from .execution_metadata import AsymSpecViewExecutionMetadata
from .views import AsymSpecDraftViews, AsymSpecViewRole


@dataclass(frozen=True)
class AsymSpecDraftForwardResult:
    """Hidden states and logits from one role-local draft forward."""

    role: AsymSpecViewRole
    hidden_states: torch.Tensor
    logits: torch.Tensor


def _dual_view_gdn_forward(full_gdn, base_gdn, hidden_states: torch.Tensor,
                           full_num_tokens: int) -> torch.Tensor:
    """Run the shared GDN projections over FULL+BASE in one traversal.

    This is the native counterpart of the frozen SpecSteer inner batch.  The
    recurrent custom-op calls remain separate because their cache bindings
    are intentionally independent; projections and output projection are
    shared over the concatenated token batch.
    """
    mixed_qkvz, _ = full_gdn.in_proj_qkvz(hidden_states)
    mixed_ba, _ = full_gdn.in_proj_ba(hidden_states)
    full_qkvz, base_qkvz = mixed_qkvz.split(
        [full_num_tokens, hidden_states.shape[0] - full_num_tokens])
    full_ba, base_ba = mixed_ba.split(
        [full_num_tokens, hidden_states.shape[0] - full_num_tokens])
    use_fused = (full_gdn.enable_fused_gdn_decode
                 and hidden_states.dtype == torch.bfloat16
                 and full_gdn.norm.weight.dtype in (torch.bfloat16, torch.float32))
    if use_fused:
        def core(module, qkvz, ba):
            out = torch.zeros(
                (qkvz.shape[0], module.num_v_heads // module.tp_size,
                 module.head_v_dim), dtype=hidden_states.dtype,
                device=hidden_states.device)
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                qkvz, ba, out, layer_name=_encode_layer_name(module.prefix))
            return out
        full_out = core(full_gdn, full_qkvz, full_ba)
        base_out = core(base_gdn, base_qkvz, base_ba)
        return full_gdn.out_proj(torch.cat((full_out, base_out), dim=0).flatten(-2))[0]

    full_qkv, full_z, full_b, full_a = full_gdn.prepare_gdn_attention_core_inputs(
        full_qkvz, full_ba, full_qkvz.shape[0])
    base_qkv, base_z, base_b, base_a = base_gdn.prepare_gdn_attention_core_inputs(
        base_qkvz, base_ba, base_qkvz.shape[0])
    def core(module, qkv, b, a):
        out = torch.zeros(
            (qkv.shape[0], module.num_v_heads // module.tp_size,
             module.head_v_dim), dtype=hidden_states.dtype,
             device=hidden_states.device)
        torch.ops.vllm.qwen_gdn_attention_core(
            qkv, b.contiguous(), a.contiguous(), out,
            layer_name=_encode_layer_name(module.prefix))
        return out
    full_out = core(full_gdn, full_qkv, full_b, full_a)
    base_out = core(base_gdn, base_qkv, base_b, base_a)
    return full_gdn._output_projection(
        torch.cat((full_out, base_out), dim=0),
        torch.cat((full_z, base_z), dim=0))


def _dual_view_attention_forward(full_attn, base_attn,
                                 hidden_states: torch.Tensor,
                                 positions: torch.Tensor,
                                 full_num_tokens: int) -> torch.Tensor:
    """Batch token-local attention projections around independent KV calls."""
    qkv, _ = full_attn.qkv_proj(hidden_states)
    q, k, v, gate = full_attn._project_qkv_gate(qkv, positions)
    full_q, base_q = q.split([full_num_tokens, q.shape[0] - full_num_tokens])
    full_k, base_k = k.split([full_num_tokens, k.shape[0] - full_num_tokens])
    full_v, base_v = v.split([full_num_tokens, v.shape[0] - full_num_tokens])
    full_out = full_attn.attn(full_q, full_k, full_v)
    base_out = base_attn.attn(base_q, base_k, base_v)
    out = torch.cat((full_out, base_out), dim=0)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return full_attn.o_proj(out)[0]


@torch.no_grad()
def execute_asymspec_dual_view_forward(
    *, full_input_ids: torch.Tensor, base_input_ids: torch.Tensor,
    full_metadata: AsymSpecViewExecutionMetadata,
    base_metadata: AsymSpecViewExecutionMetadata,
    views: AsymSpecDraftViews,
    cache_bindings: AsymSpecDraftCacheBindingRuntime,
) -> tuple[AsymSpecDraftForwardResult, AsymSpecDraftForwardResult]:
    """Execute one frozen-style shared FULL/BASE model traversal."""
    if (full_metadata.role is not AsymSpecViewRole.FULL
            or base_metadata.role is not AsymSpecViewRole.BASE):
        raise ValueError("AsymSpec dual forward requires FULL then BASE metadata")
    if full_input_ids.ndim != 1 or base_input_ids.ndim != 1:
        raise ValueError("AsymSpec dual input IDs must be one-dimensional")
    if (full_input_ids.numel() != full_metadata.query_len
            or base_input_ids.numel() != base_metadata.query_len):
        raise ValueError("AsymSpec dual input IDs do not match metadata")
    if cache_bindings is not getattr(
            getattr(views, "draft_cache_bindings", None), "runtime", None):
        raise ValueError("AsymSpec dual forward requires permanent bindings")
    full_model = views.view(AsymSpecViewRole.FULL).model
    base_model = views.view(AsymSpecViewRole.BASE).model
    full_text = getattr(full_model, "language_model", full_model)
    base_text = getattr(base_model, "language_model", base_model)
    full_core, base_core = full_text.model, base_text.model
    if len(full_core.layers) != len(base_core.layers):
        raise RuntimeError("AsymSpec FULL/BASE layer trees differ")
    ids = torch.cat((full_input_ids, base_input_ids), dim=0)
    positions = torch.cat((qwen3_5_text_positions(full_metadata.positions),
                           qwen3_5_text_positions(base_metadata.positions)), dim=1)
    layer_metadata = {
        **full_metadata.layer_metadata, **base_metadata.layer_metadata
    }
    slot_mapping = {
        **full_metadata.layer_slot_mapping, **base_metadata.layer_slot_mapping
    }
    with set_forward_context(
            layer_metadata, views.draft_vllm_config or views.vllm_config,
            num_tokens=ids.numel(), slot_mapping=slot_mapping):
        hidden = full_core.embed_input_ids(ids)
        residual = None
        full_n = full_input_ids.numel()
        for full_layer, base_layer in zip(full_core.layers, base_core.layers):
            if full_layer.layer_type != base_layer.layer_type:
                raise RuntimeError("AsymSpec FULL/BASE layer types differ")
            if residual is None:
                residual = hidden
                hidden = full_layer.input_layernorm(hidden)
            else:
                hidden, residual = full_layer.input_layernorm(hidden, residual)
            full_hidden, base_hidden = hidden[:full_n], hidden[full_n:]
            if full_layer.layer_type == "linear_attention":
                hidden = _dual_view_gdn_forward(full_layer.linear_attn,
                                                base_layer.linear_attn,
                                                hidden, full_n)
            elif full_layer.layer_type == "full_attention":
                hidden = _dual_view_attention_forward(full_layer.self_attn,
                                                      base_layer.self_attn,
                                                      hidden, positions, full_n)
            else:
                raise RuntimeError(
                    f"unsupported AsymSpec layer type {full_layer.layer_type}")
            if full_layer.layer_scale:
                hidden = hidden * (full_layer.attn_layer_scale.to(hidden.dtype)[0] + 1)
            hidden, residual = full_layer.post_attention_layernorm(hidden, residual)
            hidden = full_layer.mlp(hidden)
            if full_layer.layer_scale:
                hidden = hidden * (full_layer.ffn_layer_scale.to(hidden.dtype)[0] + 1)
        hidden, _ = full_core.norm(hidden, residual)
        full_hidden, base_hidden = hidden[:full_n], hidden[full_n:]
        full_logits = full_text.compute_logits(full_hidden[-1:])
        base_logits = base_text.compute_logits(base_hidden[-1:])
    return (AsymSpecDraftForwardResult(AsymSpecViewRole.FULL, full_hidden, full_logits),
            AsymSpecDraftForwardResult(AsymSpecViewRole.BASE, base_hidden, base_logits))


def qwen3_5_text_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return Qwen3.5's native three equal text M-RoPE axes."""
    if positions.ndim != 1:
        raise ValueError("AsymSpec Qwen3.5 positions must be one-dimensional.")
    return positions.unsqueeze(0).expand(3, -1)


def initialize_fresh_asymspec_view_state(
    *,
    role: AsymSpecViewRole,
    cache_bindings: AsymSpecDraftCacheBindingRuntime,
) -> None:
    """Reset the active view's compact recurrent pages for a fresh request.

    The frozen compact ``MambaSpec`` path begins from zeroed committed and
    speculative state pages.  Resetting the selected view's attention backing
    storage as well makes a fresh canonical request independent of any prior
    disposable use of that view; it never touches the inactive view.
    """
    for binding in cache_bindings.bindings.values():
        if binding.role is role:
            binding.raw_tensor.zero_()


@torch.no_grad()
def execute_asymspec_draft_forward(
    *,
    role: AsymSpecViewRole,
    input_ids: torch.Tensor,
    metadata: AsymSpecViewExecutionMetadata,
    views: AsymSpecDraftViews,
    cache_bindings: AsymSpecDraftCacheBindingRuntime,
) -> AsymSpecDraftForwardResult:
    """Execute one FULL or BASE Qwen3.5 forward with its own context.

    Args:
        role: The sole active logical view.
        input_ids: Packed, one-dimensional token IDs for the selected view.
        metadata: Role-local native attention and recurrent metadata.
        views: Permanently distinct shared-weight draft module trees.
        cache_bindings: Permanent role-local draft cache bindings.

    Returns:
        The selected view's hidden states and logits for every input token.

    Raises:
        ValueError: If the role, token count, device, or permanent bindings do
            not agree with the supplied view metadata.
    """
    if metadata.role is not role:
        raise ValueError("AsymSpec forward role does not match its metadata.")
    if input_ids.ndim != 1 or input_ids.numel() != metadata.query_len:
        raise ValueError("AsymSpec input IDs must match metadata query length.")
    marker = getattr(views, "draft_cache_bindings", None)
    if cache_bindings is not getattr(marker, "runtime", None):
        raise ValueError("AsymSpec forward requires the views' permanent bindings.")
    if input_ids.device != metadata.positions.device:
        raise ValueError("AsymSpec input IDs and metadata must share a device.")

    model = views.view(role).model
    with set_forward_context(
        metadata.layer_metadata,
        views.vllm_config,
        num_tokens=input_ids.numel(),
        slot_mapping=metadata.layer_slot_mapping,
    ):
        result = model(
            input_ids=input_ids,
            positions=qwen3_5_text_positions(metadata.positions),
            inputs_embeds=None,
        )
    hidden_states = result[0] if isinstance(result, tuple) else result
    return AsymSpecDraftForwardResult(
        role=role,
        hidden_states=hidden_states,
        logits=model.compute_logits(hidden_states[-1:]),
    )
