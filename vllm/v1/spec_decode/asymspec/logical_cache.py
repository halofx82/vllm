# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsymSpec's logical cache-coordinate topology.

Physical storage is deliberately independent per logical layer (see
``physical_cache.py``).  This module adds only the separate, frozen-validated
block-ID spaces that will later be attached to request block tables and model
cache bindings.  It does not bind a layer, modify a generic coordinator, or
schedule a request.
"""

from dataclasses import dataclass
from enum import Enum

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from .cache_plan import AsymSpecCacheDomain
from .physical_cache import AsymSpecPhysicalCachePlan, AsymSpecPhysicalCacheTensorPlan


class AsymSpecLogicalCacheGroup(str, Enum):
    """Semantic identities of the validated AsymSpec logical pools.

    These names intentionally replace frozen-run numeric group IDs.  The
    ordering retained by :func:`build_asymspec_logical_cache_plan` is only a
    diagnostic convenience; no caller should assign meaning to an index.
    """

    TARGET_MAMBA_A = "target_mamba_a"
    TARGET_MAMBA_B = "target_mamba_b"
    COMPRESSED_ATTENTION = "compressed_attention"
    FULL_ATTENTION = "full_attention"
    BASE_MAMBA_A = "base_mamba_a"
    FULL_MAMBA_A = "full_mamba_a"
    BASE_MAMBA_B = "base_mamba_b"
    FULL_MAMBA_B = "full_mamba_b"


@dataclass(frozen=True)
class AsymSpecLogicalCacheGroupPlan:
    """One logical block-coordinate space over independently owned tensors."""

    semantic_group: AsymSpecLogicalCacheGroup
    member_layer_names: tuple[str, ...]
    domains: frozenset[AsymSpecCacheDomain]
    kv_cache_spec: KVCacheSpec
    block_size: int
    num_blocks: int
    intentionally_shared_coordinates: bool


@dataclass(frozen=True)
class AsymSpecLogicalCachePlan:
    """Allocation-free mapping of physical layers to logical coordinates."""

    groups: tuple[AsymSpecLogicalCacheGroupPlan, ...]
    layer_to_group: dict[str, AsymSpecLogicalCacheGroup]
    physical_tensors_by_layer: dict[str, AsymSpecPhysicalCacheTensorPlan]

    def group_for(self, layer_name: str) -> AsymSpecLogicalCacheGroupPlan:
        semantic_group = self.layer_to_group[layer_name]
        return next(
            group for group in self.groups if group.semantic_group is semantic_group
        )


@dataclass
class AsymSpecLogicalBlockPoolRuntime:
    """AsymSpec-owned, unbound BlockPools keyed by semantic group."""

    plan: AsymSpecLogicalCachePlan
    pools: dict[AsymSpecLogicalCacheGroup, BlockPool]

    def pool_for_layer(self, layer_name: str) -> BlockPool:
        return self.pools[self.plan.layer_to_group[layer_name]]


def _spec_for_members(
    members: tuple[AsymSpecPhysicalCacheTensorPlan, ...],
) -> KVCacheSpec:
    specs = {member.global_layer_name: member.kv_cache_spec for member in members}
    first_spec = next(iter(specs.values()))
    if all(spec == first_spec for spec in specs.values()):
        return first_spec
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(specs)
    if uniform_spec is None:
        raise ValueError(
            "AsymSpec logical cache group has incompatible native member specs: "
            f"{tuple(specs)!r}."
        )
    return uniform_spec


def _make_group(
    semantic_group: AsymSpecLogicalCacheGroup,
    members: tuple[AsymSpecPhysicalCacheTensorPlan, ...],
    *,
    intentionally_shared_coordinates: bool = False,
) -> AsymSpecLogicalCacheGroupPlan:
    if not members:
        raise ValueError(f"AsymSpec logical group {semantic_group.value} is empty.")
    page_counts = {member.num_pages for member in members}
    block_sizes = {member.kv_cache_spec.block_size for member in members}
    if len(page_counts) != 1 or len(block_sizes) != 1:
        raise ValueError(
            "AsymSpec logical group members must have identical logical page "
            f"geometry: {semantic_group.value}."
        )
    return AsymSpecLogicalCacheGroupPlan(
        semantic_group=semantic_group,
        member_layer_names=tuple(member.global_layer_name for member in members),
        domains=frozenset(member.domain for member in members),
        kv_cache_spec=_spec_for_members(members),
        block_size=block_sizes.pop(),
        num_blocks=page_counts.pop(),
        intentionally_shared_coordinates=intentionally_shared_coordinates,
    )


def _members_for(
    physical_plan: AsymSpecPhysicalCachePlan,
    domain: AsymSpecCacheDomain,
    spec_type: type[KVCacheSpec],
) -> tuple[AsymSpecPhysicalCacheTensorPlan, ...]:
    return tuple(
        allocation
        for allocation in physical_plan.tensors
        if allocation.domain is domain
        and isinstance(allocation.kv_cache_spec, spec_type)
    )


def _split_recurrent_halves(
    members: tuple[AsymSpecPhysicalCacheTensorPlan, ...],
    domain: AsymSpecCacheDomain,
) -> tuple[
    tuple[AsymSpecPhysicalCacheTensorPlan, ...],
    tuple[AsymSpecPhysicalCacheTensorPlan, ...],
]:
    """Recover the frozen two-stratum compact recurrent topology.

    Native grouping identifies compatible Mamba layers; after backend
    normalization all such Qwen3.5/Qwen3.8 layers have the same compact page
    geometry.  The validated runtime coalesced those strata into two equal
    logical state families (48 target layers -> 24+24; 24 draft layers ->
    12+12), rather than assigning semantic meaning to native group indices.
    """
    if len(members) < 2 or len(members) % 2:
        raise ValueError(
            "AsymSpec requires an even, non-empty recurrent layer set to "
            f"derive two {domain.value} state families; got {len(members)}."
        )
    midpoint = len(members) // 2
    return members[:midpoint], members[midpoint:]


def build_asymspec_logical_cache_plan(
    physical_plan: AsymSpecPhysicalCachePlan,
) -> AsymSpecLogicalCachePlan:
    """Build the eight validated logical coordinate spaces.

    The compressed attention pool deliberately contains TARGET and BASE
    members.  This shares *block IDs only*: each member still resolves those
    IDs against its own raw physical tensor.  All other pools are single-domain.
    """
    target_attention = _members_for(
        physical_plan, AsymSpecCacheDomain.TARGET, AttentionSpec
    )
    full_attention = _members_for(
        physical_plan, AsymSpecCacheDomain.FULL, AttentionSpec
    )
    base_attention = _members_for(
        physical_plan, AsymSpecCacheDomain.BASE, AttentionSpec
    )
    target_mamba = _members_for(
        physical_plan, AsymSpecCacheDomain.TARGET, MambaSpec
    )
    full_mamba = _members_for(
        physical_plan, AsymSpecCacheDomain.FULL, MambaSpec
    )
    base_mamba = _members_for(
        physical_plan, AsymSpecCacheDomain.BASE, MambaSpec
    )

    target_mamba_a, target_mamba_b = _split_recurrent_halves(
        target_mamba, AsymSpecCacheDomain.TARGET
    )
    full_mamba_a, full_mamba_b = _split_recurrent_halves(
        full_mamba, AsymSpecCacheDomain.FULL
    )
    base_mamba_a, base_mamba_b = _split_recurrent_halves(
        base_mamba, AsymSpecCacheDomain.BASE
    )

    groups = (
        _make_group(AsymSpecLogicalCacheGroup.TARGET_MAMBA_A, target_mamba_a),
        _make_group(AsymSpecLogicalCacheGroup.TARGET_MAMBA_B, target_mamba_b),
        _make_group(
            AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
            target_attention + base_attention,
            intentionally_shared_coordinates=True,
        ),
        _make_group(AsymSpecLogicalCacheGroup.FULL_ATTENTION, full_attention),
        _make_group(AsymSpecLogicalCacheGroup.BASE_MAMBA_A, base_mamba_a),
        _make_group(AsymSpecLogicalCacheGroup.FULL_MAMBA_A, full_mamba_a),
        _make_group(AsymSpecLogicalCacheGroup.BASE_MAMBA_B, base_mamba_b),
        _make_group(AsymSpecLogicalCacheGroup.FULL_MAMBA_B, full_mamba_b),
    )
    layer_to_group = {
        layer_name: group.semantic_group
        for group in groups
        for layer_name in group.member_layer_names
    }
    if len(layer_to_group) != len(physical_plan.tensors):
        raise ValueError(
            "AsymSpec logical cache topology omitted or duplicated a physical "
            "cache layer."
        )
    return AsymSpecLogicalCachePlan(
        groups=groups,
        layer_to_group=layer_to_group,
        physical_tensors_by_layer={
            tensor.global_layer_name: tensor for tensor in physical_plan.tensors
        },
    )


def instantiate_asymspec_logical_block_pools(
    plan: AsymSpecLogicalCachePlan,
) -> AsymSpecLogicalBlockPoolRuntime:
    """Instantiate isolated, prefix-cache-disabled pools for logical groups."""
    pools = {
        group.semantic_group: BlockPool(
            num_gpu_blocks=group.num_blocks,
            enable_caching=False,
            hash_block_size=group.block_size,
        )
        for group in plan.groups
    }
    if len(pools) != len(plan.groups) or len(
        {id(pool) for pool in pools.values()}
    ) != len(pools):
        raise AssertionError("AsymSpec logical groups must own distinct BlockPools.")
    return AsymSpecLogicalBlockPoolRuntime(plan=plan, pools=pools)


def allocate_synthetic_attention_blocks(
    runtime: AsymSpecLogicalBlockPoolRuntime,
    semantic_group: AsymSpecLogicalCacheGroup,
    num_tokens: int,
) -> tuple[int, ...]:
    """Allocate a disposable attention-only synthetic block table.

    This test helper intentionally has no request/scheduler dependency.  A
    later stage will replace it with real request block-table integration.
    """
    group = next(
        group for group in runtime.plan.groups if group.semantic_group is semantic_group
    )
    if not isinstance(group.kv_cache_spec, (AttentionSpec, UniformTypeKVCacheSpecs)):
        raise ValueError(f"{semantic_group.value} is not an attention group.")
    if num_tokens < 0:
        raise ValueError("AsymSpec synthetic token count cannot be negative.")
    blocks = runtime.pools[semantic_group].get_new_blocks(
        cdiv(num_tokens, group.block_size)
    )
    return tuple(block.block_id for block in blocks)
