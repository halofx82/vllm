# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free, role-aware KV cache plans for AsymSpec draft views."""

from dataclasses import dataclass, replace
from typing import Any

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
)

from .hybrid import AsymSpecHybridStateSpec
from .views import AsymSpecViewRole


@dataclass(frozen=True)
class AsymSpecCachePlan:
    """Native cache specifications owned by one logical AsymSpec view.

    This is deliberately only a plan: it does not carry group IDs, blocks,
    block tables, tensors, or a cache-manager reference. Stage 2d will map
    these role-owned plans into the engine-global cache configuration.
    """

    role: AsymSpecViewRole
    max_model_len: int
    layer_specs: dict[str, KVCacheSpec]
    cache_groups: list[KVCacheGroupSpec]


def _layer_specs_from_model(
    model: Any,
    hybrid_spec: AsymSpecHybridStateSpec,
    draft_vllm_config: VllmConfig,
) -> dict[str, KVCacheSpec]:
    """Obtain the same native cache specs each loaded layer publishes."""
    modules = dict(model.named_modules())
    layer_specs: dict[str, KVCacheSpec] = {}

    for state_spec in hybrid_spec.attention:
        layer = modules.get(state_spec.layer_name)
        if not isinstance(layer, AttentionLayerBase):
            raise ValueError(
                "AsymSpec attention metadata does not resolve to an attention "
                f"layer: {state_spec.layer_name!r}."
            )
        spec = layer.get_kv_cache_spec(draft_vllm_config)
        if not isinstance(spec, AttentionSpec):
            raise ValueError(
                "AsymSpec attention layer did not provide a native attention "
                f"KV cache spec: {state_spec.layer_name!r}."
            )
        # Match the native GPU runner's backend-specific spec finalization.
        # Synthetic unit-test layers and non-standard attention subclasses can
        # still publish an already-native AttentionSpec directly.
        if isinstance(layer, Attention):
            backend = layer.get_attn_backend()
            with set_current_vllm_config(draft_vllm_config):
                indexes_by_block_stride = backend.indexes_kv_by_block_stride()
            spec = replace(spec, indexes_kv_by_block_stride=indexes_by_block_stride)
            spec = backend.customize_spec(spec)
        layer_specs[state_spec.layer_name] = spec

    for state_spec in hybrid_spec.recurrent:
        layer = modules.get(state_spec.layer_name)
        if not isinstance(layer, MambaBase):
            raise ValueError(
                "AsymSpec recurrent metadata does not resolve to a Mamba/GDN "
                f"layer: {state_spec.layer_name!r}."
            )
        spec = layer.get_kv_cache_spec(draft_vllm_config)
        if not isinstance(spec, MambaSpec):
            raise ValueError(
                "AsymSpec recurrent layer did not provide a native MambaSpec: "
                f"{state_spec.layer_name!r}."
            )
        if spec.mamba_cache_mode != "none":
            raise ValueError(
                "AsymSpec requires the compact/native Mamba cache mode 'none', "
                f"got {spec.mamba_cache_mode!r}."
            )
        layer_specs[state_spec.layer_name] = spec

    return layer_specs


def build_asymspec_cache_plan(
    *,
    role: AsymSpecViewRole,
    max_model_len: int,
    model: Any,
    hybrid_spec: AsymSpecHybridStateSpec,
    draft_vllm_config: VllmConfig,
) -> AsymSpecCachePlan:
    """Build one role-owned plan using vLLM's native spec/grouping logic."""
    if max_model_len <= 0:
        raise ValueError(f"AsymSpec {role.value} max_model_len must be positive.")

    layer_specs = _layer_specs_from_model(model, hybrid_spec, draft_vllm_config)
    # Native grouping is allocation-free. It may adjust a *per-plan* spec map
    # for physical-page compatibility, so each role starts with a fresh map.
    cache_groups = get_kv_cache_groups(draft_vllm_config, layer_specs)
    return AsymSpecCachePlan(
        role=role,
        max_model_len=max_model_len,
        layer_specs=layer_specs,
        cache_groups=cache_groups,
    )
