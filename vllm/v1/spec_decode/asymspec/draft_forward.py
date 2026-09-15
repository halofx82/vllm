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

from .cache_binding import AsymSpecDraftCacheBindingRuntime
from .execution_metadata import AsymSpecViewExecutionMetadata
from .views import AsymSpecDraftViews, AsymSpecViewRole


@dataclass(frozen=True)
class AsymSpecDraftForwardResult:
    """Hidden states and logits from one role-local draft forward."""

    role: AsymSpecViewRole
    hidden_states: torch.Tensor
    logits: torch.Tensor


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
