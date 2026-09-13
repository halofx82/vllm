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


@dataclass(frozen=True)
class AsymSpecCacheLayerBinding:
    """The logical owner and physical source of one draft cache alias."""

    role: AsymSpecViewRole
    physical_layer_name: str


@dataclass(frozen=True)
class AsymSpecCacheNameRegistry:
    """Reversible cache-name aliases for the two logical draft views.

    The maps are authoritative: callers recover a role through ``binding_for``
    rather than parsing a cache-layer name or depending on a group ID.
    """

    global_to_binding: dict[str, AsymSpecCacheLayerBinding]
    role_physical_to_global: dict[tuple[AsymSpecViewRole, str], str]

    @staticmethod
    def _global_name(role: AsymSpecViewRole, physical_layer_name: str) -> str:
        return f"__asymspec_cache__.{role.value}.{physical_layer_name}"

    @classmethod
    def create(
        cls,
        target_layer_names: set[str],
        plans: tuple[AsymSpecCachePlan, AsymSpecCachePlan],
    ) -> "AsymSpecCacheNameRegistry":
        global_to_binding: dict[str, AsymSpecCacheLayerBinding] = {}
        role_physical_to_global: dict[tuple[AsymSpecViewRole, str], str] = {}
        for plan in plans:
            for physical_layer_name in plan.layer_specs:
                key = (plan.role, physical_layer_name)
                global_name = cls._global_name(*key)
                if global_name in target_layer_names:
                    raise ValueError(
                        "Target KV-cache layer name collides with reserved "
                        f"AsymSpec namespace: {global_name!r}."
                    )
                if global_name in global_to_binding:
                    raise ValueError(
                        "AsymSpec cache-name collision for "
                        f"{plan.role.value}:{physical_layer_name!r}."
                    )
                binding = AsymSpecCacheLayerBinding(*key)
                global_to_binding[global_name] = binding
                role_physical_to_global[key] = global_name
        return cls(global_to_binding, role_physical_to_global)

    def binding_for(self, global_name: str) -> AsymSpecCacheLayerBinding:
        return self.global_to_binding[global_name]

    def global_name_for(self, role: AsymSpecViewRole, physical_layer_name: str) -> str:
        return self.role_physical_to_global[(role, physical_layer_name)]


@dataclass(frozen=True)
class AsymSpecGlobalCachePlan:
    """Allocation-free engine-global cache-spec namespace for AsymSpec."""

    merged_specs: dict[str, KVCacheSpec]
    registry: AsymSpecCacheNameRegistry


@dataclass(frozen=True)
class AsymSpecNativeGroupingDiagnostic:
    """Role composition observed from native, allocation-free grouping."""

    group_roles: tuple[frozenset[str], ...]

    @property
    def role_separated(self) -> bool:
        return all(len(roles) == 1 for roles in self.group_roles)


def compose_asymspec_global_cache_plan(
    *,
    target_specs: dict[str, KVCacheSpec],
    full_plan: AsymSpecCachePlan,
    base_plan: AsymSpecCachePlan,
) -> AsymSpecGlobalCachePlan:
    """Compose target/FULL/BASE specs without mutating either role plan."""
    if full_plan.role is not AsymSpecViewRole.FULL:
        raise ValueError("AsymSpec FULL cache plan must have role FULL.")
    if base_plan.role is not AsymSpecViewRole.BASE:
        raise ValueError("AsymSpec BASE cache plan must have role BASE.")

    registry = AsymSpecCacheNameRegistry.create(
        set(target_specs), (full_plan, base_plan)
    )
    merged_specs = dict(target_specs)
    for plan in (full_plan, base_plan):
        for physical_layer_name, spec in plan.layer_specs.items():
            global_name = registry.global_name_for(plan.role, physical_layer_name)
            merged_specs[global_name] = spec
    return AsymSpecGlobalCachePlan(merged_specs, registry)


def diagnose_asymspec_native_grouping(
    *,
    global_plan: AsymSpecGlobalCachePlan,
    vllm_config: VllmConfig,
) -> AsymSpecNativeGroupingDiagnostic:
    """Characterize whether native grouping preserves role boundaries.

    The native helper can normalize its input mapping, so this intentionally
    passes a fresh mapping. The global plan and role-owned plans stay intact.
    """
    groups = get_kv_cache_groups(vllm_config, dict(global_plan.merged_specs))
    group_roles: list[frozenset[str]] = []
    for group in groups:
        roles = {
            (
                global_plan.registry.binding_for(layer_name).role.value
                if layer_name in global_plan.registry.global_to_binding
                else "target"
            )
            for layer_name in group.layer_names
        }
        group_roles.append(frozenset(roles))
    return AsymSpecNativeGroupingDiagnostic(tuple(group_roles))


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
