# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Permanent, role-local bindings for AsymSpec draft cache storage.

The generic :func:`bind_kv_cache` helper is deliberately not used here.  It
assumes a single runner-owned cache list and a single forward context.  An
AsymSpec worker instead owns two distinct draft module trees whose weights
alias but whose cache state must remain permanently independent.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from vllm.v1.worker.utils import select_common_block_size

from .cache_plan import AsymSpecCacheDomain, AsymSpecGlobalCachePlan
from .logical_cache import AsymSpecLogicalBlockPoolRuntime
from .physical_cache import (
    AsymSpecPhysicalCacheRuntime,
    AsymSpecPhysicalCacheTensorPlan,
)
from .views import AsymSpecDraftViews, AsymSpecViewRole


@dataclass(frozen=True)
class AsymSpecDraftCacheBinding:
    """One permanently bound cache view for one draft layer."""

    role: AsymSpecViewRole
    global_layer_name: str
    physical_layer_name: str
    module: AttentionLayerBase
    raw_tensor: torch.Tensor
    cache: torch.Tensor | tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AsymSpecDraftCacheBindingRuntime:
    """AsymSpec-owned permanent draft bindings and logical pools.

    This runtime object intentionally does not enter ``GPUModelRunner.kv_caches``.
    TARGET cache binding remains wholly on the ordinary vLLM path.
    """

    bindings: dict[str, AsymSpecDraftCacheBinding]
    logical_pools: AsymSpecLogicalBlockPoolRuntime

    @property
    def full_bindings(self) -> tuple[AsymSpecDraftCacheBinding, ...]:
        return tuple(
            binding
            for binding in self.bindings.values()
            if binding.role is AsymSpecViewRole.FULL
        )

    @property
    def base_bindings(self) -> tuple[AsymSpecDraftCacheBinding, ...]:
        return tuple(
            binding
            for binding in self.bindings.values()
            if binding.role is AsymSpecViewRole.BASE
        )


def _domain_for_role(role: AsymSpecViewRole) -> AsymSpecCacheDomain:
    return (
        AsymSpecCacheDomain.FULL
        if role is AsymSpecViewRole.FULL
        else AsymSpecCacheDomain.BASE
    )


def _attention_cache_view(
    *,
    layer: AttentionLayerBase,
    spec: AttentionSpec,
    raw_tensor: torch.Tensor,
    num_pages: int,
    vllm_config: VllmConfig,
) -> torch.Tensor:
    """Build the native attention backend's view over one raw byte tensor."""
    backend = layer.get_attn_backend()
    kernel_block_size = select_common_block_size(spec.block_size, [backend])
    if spec.block_size % kernel_block_size:
        raise ValueError("AsymSpec attention block size cannot be kernel-split.")
    kernel_num_blocks = num_pages * (spec.block_size // kernel_block_size)
    shape_block_size = (
        spec.storage_block_size
        if spec.storage_block_size != spec.block_size
        else kernel_block_size
    )
    cache_dtype = (
        "auto"
        if spec.kv_quant_mode == KVQuantMode.NONE
        else getattr(spec, "cache_dtype_str", None) or "auto"
    )
    with set_current_vllm_config(vllm_config):
        kv_cache_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            shape_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        try:
            stride_order = backend.get_kv_cache_stride_order()
            if len(stride_order) != len(kv_cache_shape):
                raise ValueError("invalid attention cache stride order")
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(kv_cache_shape)))
    return _reshape_attention_kv_cache(
        raw_tensor,
        spec,
        kv_cache_shape,
        stride_order,
        kernel_num_blocks,
        packing=None,
    )


def _mamba_cache_view(
    *, raw_tensor: torch.Tensor, spec: MambaSpec, num_pages: int
) -> torch.Tensor:
    return raw_tensor[: num_pages * spec.page_size_bytes].view(
        num_pages, 1, 1, spec.page_size_bytes
    )


def _validate_allocation(
    allocation: AsymSpecPhysicalCacheTensorPlan,
    *,
    global_plan: AsymSpecGlobalCachePlan,
) -> tuple[AsymSpecViewRole, str]:
    binding = global_plan.registry.binding_for(allocation.global_layer_name)
    expected_domain = _domain_for_role(binding.role)
    if allocation.domain is not expected_domain:
        raise ValueError(
            "AsymSpec draft cache allocation domain does not match its view "
            f"role for {allocation.global_layer_name!r}."
        )
    if allocation.physical_layer_name != binding.physical_layer_name:
        raise ValueError(
            "AsymSpec physical cache layer name does not match registry for "
            f"{allocation.global_layer_name!r}."
        )
    return binding.role, binding.physical_layer_name


def bind_asymspec_draft_caches(
    *,
    views: AsymSpecDraftViews,
    physical_runtime: AsymSpecPhysicalCacheRuntime,
    global_plan: AsymSpecGlobalCachePlan,
    logical_pools: AsymSpecLogicalBlockPoolRuntime,
    vllm_config: VllmConfig,
) -> AsymSpecDraftCacheBindingRuntime:
    """Bind the independent FULL/BASE tensors to their own module trees.

    The operation is intentionally permanent. Repeating it with the same
    runtime is idempotent; any other attempt is rejected rather than silently
    overwriting cache state on either tree.
    """
    existing: Any = getattr(views, "draft_cache_bindings", None)
    if existing is not None:
        if (
            existing.physical_runtime is physical_runtime
            and existing.global_plan is global_plan
        ):
            return existing.runtime
        raise RuntimeError("AsymSpec draft caches are already permanently bound.")

    if logical_pools.plan.physical_tensors_by_layer != {
        plan.global_layer_name: plan for plan in physical_runtime.plan.tensors
    }:
        raise ValueError("AsymSpec logical pools and physical cache plan differ.")

    expected_names = {
        allocation.global_layer_name
        for allocation in physical_runtime.plan.tensors
        if allocation.domain in (AsymSpecCacheDomain.FULL, AsymSpecCacheDomain.BASE)
    }
    if set(physical_runtime.raw_tensors) != {
        allocation.global_layer_name for allocation in physical_runtime.plan.tensors
    }:
        raise ValueError("AsymSpec raw cache runtime does not match its plan.")

    modules = {
        role: dict(views.view(role).model.named_modules()) for role in AsymSpecViewRole
    }
    bindings: dict[str, AsymSpecDraftCacheBinding] = {}
    seen_modules: set[tuple[AsymSpecViewRole, int]] = set()
    for allocation in physical_runtime.plan.tensors:
        if allocation.domain is AsymSpecCacheDomain.TARGET:
            continue
        role, physical_name = _validate_allocation(allocation, global_plan=global_plan)
        layer = modules[role].get(physical_name)
        if not isinstance(layer, AttentionLayerBase):
            raise ValueError(
                "AsymSpec cache alias does not resolve to an attention/recurrent "
                f"layer in the {role.value} tree: {physical_name!r}."
            )
        key = (role, id(layer))
        if key in seen_modules:
            raise ValueError(
                "AsymSpec cache aliases resolve to the same draft module more "
                f"than once: {role.value}:{physical_name!r}."
            )
        seen_modules.add(key)
        raw_tensor = physical_runtime.raw_tensors[allocation.global_layer_name]
        spec = allocation.kv_cache_spec
        if isinstance(spec, AttentionSpec) and not isinstance(layer, MambaBase):
            cache: torch.Tensor | tuple[torch.Tensor, ...] = _attention_cache_view(
                layer=layer,
                spec=spec,
                raw_tensor=raw_tensor,
                num_pages=allocation.num_pages,
                vllm_config=vllm_config,
            )
        elif isinstance(spec, MambaSpec) and isinstance(layer, MambaBase):
            cache_view = _mamba_cache_view(
                raw_tensor=raw_tensor, spec=spec, num_pages=allocation.num_pages
            )
            layer.bind_kv_cache(cache_view)
            cache = layer.kv_cache
        else:
            raise ValueError(
                "AsymSpec cache spec/module type mismatch for "
                f"{role.value}:{physical_name!r}."
            )
        if isinstance(spec, AttentionSpec) and not isinstance(layer, MambaBase):
            layer.bind_kv_cache(cache)
        bindings[allocation.global_layer_name] = AsymSpecDraftCacheBinding(
            role=role,
            global_layer_name=allocation.global_layer_name,
            physical_layer_name=physical_name,
            module=layer,
            raw_tensor=raw_tensor,
            cache=cache,
        )

    if set(bindings) != expected_names or len(seen_modules) != len(expected_names):
        raise AssertionError("AsymSpec draft cache binding cardinality mismatch.")
    runtime = AsymSpecDraftCacheBindingRuntime(bindings, logical_pools)
    # Keep the ownership marker on AsymSpec views, never on GPUModelRunner.
    views.draft_cache_bindings = _BoundDraftCacheMarker(
        runtime, physical_runtime, global_plan
    )
    views.full.state.kv_state = runtime
    views.base.state.kv_state = runtime
    return runtime


@dataclass(frozen=True)
class _BoundDraftCacheMarker:
    runtime: AsymSpecDraftCacheBindingRuntime
    physical_runtime: AsymSpecPhysicalCacheRuntime
    global_plan: AsymSpecGlobalCachePlan
