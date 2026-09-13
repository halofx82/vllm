# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free, role-aware KV cache plans for AsymSpec draft views."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_uniform_page_size,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
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


class AsymSpecCacheDomain(str, Enum):
    """Semantic owner of an engine-global cache group.

    This deliberately differs from :class:`AsymSpecViewRole`: TARGET is not a
    draft view, while FULL and BASE are. Cache allocation and scheduling can
    therefore use this type without assigning meaning to a global group index.
    """

    TARGET = "target"
    FULL = "full"
    BASE = "base"


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
    cache_groups: tuple[KVCacheGroupSpec, ...]
    group_domains: tuple[AsymSpecCacheDomain, ...]

    def domain_for_group(self, global_group_index: int) -> AsymSpecCacheDomain:
        return self.group_domains[global_group_index]

    def group_indices_for_domain(self, domain: AsymSpecCacheDomain) -> tuple[int, ...]:
        return tuple(
            index
            for index, group_domain in enumerate(self.group_domains)
            if group_domain is domain
        )


@dataclass(frozen=True)
class AsymSpecAllocatorCompatibility:
    """Allocation-free characterization of generic allocator behavior."""

    uniform_page_size_bytes: int | None
    uniform_page_size_error: str | None
    num_blocks: int | None
    tensor_shared_domains: tuple[frozenset[AsymSpecCacheDomain], ...]
    tensor_shares_domains: bool
    config_error: str | None


@dataclass(frozen=True)
class AsymSpecDomainGroupAllocationPlan:
    """Minimum allocation geometry for one native cache group.

    ``min_blocks_per_request`` includes the physical null block reserved by
    vLLM's block pool.  It is therefore intentionally one larger than the
    block-table width for token-addressed groups.  Compact Mamba groups are
    bounded resident state: their physical count is independent of context
    length.
    """

    domain: AsymSpecCacheDomain
    layer_names: tuple[str, ...]
    kv_cache_spec: KVCacheSpec
    block_size: int
    page_size_bytes: int
    request_block_table_entries: int
    min_blocks_per_request: int
    minimum_bytes: int


@dataclass(frozen=True)
class AsymSpecDomainAllocationPlan:
    """Allocation-free lower bound and geometry for one cache domain."""

    domain: AsymSpecCacheDomain
    max_tokens: int
    page_size_bytes: int
    groups: tuple[AsymSpecDomainGroupAllocationPlan, ...]
    min_blocks_per_request: int
    minimum_bytes: int

    @property
    def backing_pool_key(self) -> AsymSpecCacheDomain:
        """Future allocators use this domain key for an independent pool."""
        return self.domain


@dataclass(frozen=True)
class AsymSpecDomainAllocationPlans:
    """Lower-bound allocation metadata for TARGET, FULL, and BASE.

    This is deliberately not ``KVCacheConfig``.  It describes three future
    backing pools, retains no runtime object, and leaves all memory beyond one
    maximum-length request unassigned.
    """

    target: AsymSpecDomainAllocationPlan
    full: AsymSpecDomainAllocationPlan
    base: AsymSpecDomainAllocationPlan
    available_memory_bytes: int | None

    @property
    def domains(self) -> tuple[AsymSpecDomainAllocationPlan, ...]:
        return (self.target, self.full, self.base)

    @property
    def minimum_bytes(self) -> int:
        return sum(domain.minimum_bytes for domain in self.domains)

    @property
    def remaining_unassigned_bytes(self) -> int | None:
        if self.available_memory_bytes is None:
            return None
        return self.available_memory_bytes - self.minimum_bytes

    @property
    def has_minimum_capacity(self) -> bool | None:
        if self.available_memory_bytes is None:
            return None
        return self.remaining_unassigned_bytes >= 0


def compose_asymspec_global_cache_plan(
    *,
    vllm_config: VllmConfig,
    target_specs: dict[str, KVCacheSpec],
    full_plan: AsymSpecCachePlan,
    base_plan: AsymSpecCachePlan,
) -> AsymSpecGlobalCachePlan:
    """Compose role-separated groups without mutating either role plan.

    Native grouping is run independently inside each cache domain. Draft-group
    layer names are then translated to engine-global aliases, preventing a
    shared physical Qwen layer from making FULL/BASE cache ownership ambiguous.
    The resulting group order is deterministic (TARGET, FULL, BASE), but all
    semantic ownership is carried by ``group_domains`` instead of that order.
    """
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

    target_groups = get_kv_cache_groups(vllm_config, dict(target_specs))

    def alias_groups(
        plan: AsymSpecCachePlan,
    ) -> list[KVCacheGroupSpec]:
        return [
            KVCacheGroupSpec(
                layer_names=[
                    registry.global_name_for(plan.role, layer_name)
                    for layer_name in group.layer_names
                ],
                kv_cache_spec=group.kv_cache_spec,
                is_eagle_group=group.is_eagle_group,
            )
            for group in plan.cache_groups
        ]

    full_groups = alias_groups(full_plan)
    base_groups = alias_groups(base_plan)
    cache_groups = tuple(target_groups + full_groups + base_groups)
    group_domains = (
        (AsymSpecCacheDomain.TARGET,) * len(target_groups)
        + (AsymSpecCacheDomain.FULL,) * len(full_groups)
        + (AsymSpecCacheDomain.BASE,) * len(base_groups)
    )
    return AsymSpecGlobalCachePlan(
        merged_specs=merged_specs,
        registry=registry,
        cache_groups=cache_groups,
        group_domains=group_domains,
    )


def _group_layer_specs(group: KVCacheGroupSpec) -> tuple[KVCacheSpec, ...]:
    """Expand a native group to its physical per-layer specs."""
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return tuple(spec.kv_cache_specs[name] for name in group.layer_names)
    return (spec,) * len(group.layer_names)


def _minimum_blocks_for_group(
    *,
    group: KVCacheGroupSpec,
    max_tokens: int,
    domain: AsymSpecCacheDomain,
) -> AsymSpecDomainGroupAllocationPlan:
    """Mirror the frozen validated one-request asymmetric allocation rule."""
    layer_specs = _group_layer_specs(group)
    if len({spec.page_size_bytes for spec in layer_specs}) != 1:
        raise ValueError(
            "AsymSpec requires one physical page size per cache group; "
            f"{domain.value} group {group.layer_names!r} is heterogeneous."
        )
    if len({spec.block_size for spec in layer_specs}) != 1:
        raise ValueError(
            "AsymSpec requires one native block size per cache group; "
            f"{domain.value} group {group.layer_names!r} is heterogeneous."
        )

    spec = group.kv_cache_spec
    if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "none":
        # Validated compact semantics: one resident committed state, K
        # speculative states, and one physical null block held by BlockPool.
        request_entries = 1 + spec.num_speculative_blocks
        min_blocks = 2 + spec.num_speculative_blocks
    else:
        request_entries = cdiv(max_tokens, spec.block_size)
        mamba_checkpoint_blocks = (
            spec.num_speculative_blocks if isinstance(spec, MambaSpec) else 0
        )
        # The frozen allocator reserves the physical null block in every
        # token-addressed group and retains Mamba checkpoint blocks where
        # native Mamba semantics require them.
        min_blocks = request_entries + 1 + mamba_checkpoint_blocks

    page_size = layer_specs[0].page_size_bytes
    return AsymSpecDomainGroupAllocationPlan(
        domain=domain,
        layer_names=tuple(group.layer_names),
        kv_cache_spec=spec,
        block_size=spec.block_size,
        page_size_bytes=page_size,
        request_block_table_entries=request_entries,
        min_blocks_per_request=min_blocks,
        minimum_bytes=len(layer_specs) * page_size * min_blocks,
    )


def _build_domain_allocation_plan(
    *,
    domain: AsymSpecCacheDomain,
    max_tokens: int,
    groups: tuple[KVCacheGroupSpec, ...],
) -> AsymSpecDomainAllocationPlan:
    if max_tokens <= 0:
        raise ValueError(f"AsymSpec {domain.value} max_tokens must be positive.")
    if not groups:
        raise ValueError(f"AsymSpec {domain.value} requires at least one cache group.")

    group_plans = tuple(
        _minimum_blocks_for_group(group=group, max_tokens=max_tokens, domain=domain)
        for group in groups
    )
    page_sizes = {group.page_size_bytes for group in group_plans}
    if len(page_sizes) != 1:
        raise ValueError(
            "AsymSpec requires compatible native page sizes within each "
            f"domain; {domain.value} has {sorted(page_sizes)}."
        )
    return AsymSpecDomainAllocationPlan(
        domain=domain,
        max_tokens=max_tokens,
        page_size_bytes=page_sizes.pop(),
        groups=group_plans,
        min_blocks_per_request=max(
            group.min_blocks_per_request for group in group_plans
        ),
        minimum_bytes=sum(group.minimum_bytes for group in group_plans),
    )


def build_asymspec_domain_allocation_plans(
    *,
    global_plan: AsymSpecGlobalCachePlan,
    full_max_model_len: int,
    compressed_max_model_len: int,
    available_memory_bytes: int | None = None,
) -> AsymSpecDomainAllocationPlans:
    """Plan independent TARGET/FULL/BASE lower bounds without allocation.

    TARGET intentionally uses the compressed request budget, not the target
    model's global ``max_model_len``.  The frozen SCALE1 allocator allocated
    exactly these one-request minima and rejected insufficient memory; it did
    not establish a validated policy for distributing surplus capacity.  This
    planner therefore leaves surplus bytes explicitly unassigned.
    """
    if available_memory_bytes is not None and available_memory_bytes < 0:
        raise ValueError("AsymSpec available KV memory cannot be negative.")
    domain_groups = {
        domain: tuple(
            global_plan.cache_groups[index]
            for index in global_plan.group_indices_for_domain(domain)
        )
        for domain in AsymSpecCacheDomain
    }
    return AsymSpecDomainAllocationPlans(
        target=_build_domain_allocation_plan(
            domain=AsymSpecCacheDomain.TARGET,
            max_tokens=compressed_max_model_len,
            groups=domain_groups[AsymSpecCacheDomain.TARGET],
        ),
        full=_build_domain_allocation_plan(
            domain=AsymSpecCacheDomain.FULL,
            max_tokens=full_max_model_len,
            groups=domain_groups[AsymSpecCacheDomain.FULL],
        ),
        base=_build_domain_allocation_plan(
            domain=AsymSpecCacheDomain.BASE,
            max_tokens=compressed_max_model_len,
            groups=domain_groups[AsymSpecCacheDomain.BASE],
        ),
        available_memory_bytes=available_memory_bytes,
    )


def characterize_asymspec_allocator_compatibility(
    *,
    global_plan: AsymSpecGlobalCachePlan,
    vllm_config: VllmConfig,
    available_memory: int,
) -> AsymSpecAllocatorCompatibility:
    """Exercise generic allocation *metadata* without allocating cache memory."""
    try:
        uniform_page_size = get_uniform_page_size(
            group.kv_cache_spec for group in global_plan.cache_groups
        )
    except (AssertionError, NotImplementedError) as error:
        uniform_page_size = None
        page_sizes = sorted(
            {group.kv_cache_spec.page_size_bytes for group in global_plan.cache_groups}
        )
        uniform_page_size_error = (
            f"{type(error).__name__}: incompatible page sizes {page_sizes}"
        )
    else:
        uniform_page_size_error = None

    try:
        config = get_kv_cache_config_from_groups(
            vllm_config, list(global_plan.cache_groups), available_memory
        )
    except (AssertionError, NotImplementedError, ValueError) as error:
        return AsymSpecAllocatorCompatibility(
            uniform_page_size_bytes=uniform_page_size,
            uniform_page_size_error=uniform_page_size_error,
            num_blocks=None,
            tensor_shared_domains=(),
            tensor_shares_domains=False,
            config_error=f"{type(error).__name__}: {error}",
        )

    def domain_for_layer(layer_name: str) -> AsymSpecCacheDomain:
        binding = global_plan.registry.global_to_binding.get(layer_name)
        if binding is None:
            return AsymSpecCacheDomain.TARGET
        return (
            AsymSpecCacheDomain.FULL
            if binding.role is AsymSpecViewRole.FULL
            else AsymSpecCacheDomain.BASE
        )

    tensor_shared_domains = tuple(
        frozenset(domain_for_layer(layer_name) for layer_name in tensor.shared_by)
        for tensor in config.kv_cache_tensors
    )
    return AsymSpecAllocatorCompatibility(
        uniform_page_size_bytes=uniform_page_size,
        uniform_page_size_error=uniform_page_size_error,
        num_blocks=config.num_blocks,
        tensor_shared_domains=tensor_shared_domains,
        tensor_shares_domains=any(
            len(domains) > 1 for domains in tensor_shared_domains
        ),
        config_error=None,
    )


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
