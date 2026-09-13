# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent raw GPU backing storage for allocation-planned AsymSpec KV.

This module deliberately stops before cache binding.  In particular, it does
not create a ``BlockPool``, a block table, or a layer-to-cache binding.  Its
only job is to materialize the physical ownership proven by the frozen
AsymSpec implementation: every TARGET, FULL, and BASE logical cache layer
gets one independent byte backing tensor.
"""

from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheTensor

from .cache_plan import (
    AsymSpecCacheDomain,
    AsymSpecDomainAllocationPlans,
    AsymSpecGlobalCachePlan,
    _group_layer_specs,
)


@dataclass(frozen=True)
class AsymSpecPhysicalCacheTensorPlan:
    """One independently owned physical allocation for one logical layer."""

    domain: AsymSpecCacheDomain
    global_layer_name: str
    physical_layer_name: str
    global_group_index: int
    kv_cache_spec: KVCacheSpec
    num_pages: int
    page_size_bytes: int
    total_bytes: int
    kv_cache_tensor: KVCacheTensor


@dataclass(frozen=True)
class AsymSpecPhysicalCachePlan:
    """Raw tensor plan retaining no device allocation or runtime cache state."""

    tensors: tuple[AsymSpecPhysicalCacheTensorPlan, ...]

    @property
    def total_bytes(self) -> int:
        return sum(tensor.total_bytes for tensor in self.tensors)

    def tensors_for_domain(
        self, domain: AsymSpecCacheDomain
    ) -> tuple[AsymSpecPhysicalCacheTensorPlan, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.domain is domain)

    def bytes_for_domain(self, domain: AsymSpecCacheDomain) -> int:
        return sum(tensor.total_bytes for tensor in self.tensors_for_domain(domain))


@dataclass
class AsymSpecPhysicalCacheRuntime:
    """AsymSpec-owned, unbound raw tensors.

    ``raw_tensors`` intentionally does not feed ``GPUModelRunner.kv_caches``.
    A later stage is responsible for logical pools, block tables, and binding.
    """

    plan: AsymSpecPhysicalCachePlan
    raw_tensors: dict[str, torch.Tensor]

    @property
    def total_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in self.raw_tensors.values()
        )

    def bytes_for_domain(self, domain: AsymSpecCacheDomain) -> int:
        names = {
            allocation.global_layer_name
            for allocation in self.plan.tensors_for_domain(domain)
        }
        return sum(
            self.raw_tensors[name].untyped_storage().nbytes() for name in names
        )


def _physical_layer_name(
    global_plan: AsymSpecGlobalCachePlan, global_layer_name: str
) -> str:
    binding = global_plan.registry.global_to_binding.get(global_layer_name)
    return global_layer_name if binding is None else binding.physical_layer_name


def build_asymspec_physical_cache_plan(
    *,
    global_plan: AsymSpecGlobalCachePlan,
    domain_plans: AsymSpecDomainAllocationPlans,
) -> AsymSpecPhysicalCachePlan:
    """Build exact independent per-layer physical allocations.

    The domain allocation plan has already decided each group's minimum page
    count.  Reusing that page count is intentional: this helper must not
    rederive capacity from an engine-global maximum length.
    """
    domain_group_plans = {
        domain: iter(getattr(domain_plans, domain.value).groups)
        for domain in AsymSpecCacheDomain
    }
    tensors: list[AsymSpecPhysicalCacheTensorPlan] = []

    for global_group_index, group in enumerate(global_plan.cache_groups):
        domain = global_plan.domain_for_group(global_group_index)
        group_plan = next(domain_group_plans[domain], None)
        if group_plan is None:
            raise ValueError(
                f"AsymSpec {domain.value} allocation plan has fewer groups "
                "than its global cache plan."
            )
        if tuple(group.layer_names) != group_plan.layer_names:
            raise ValueError(
                "AsymSpec allocation-plan group layer names do not match the "
                f"global cache plan for {domain.value}."
            )

        layer_specs = _group_layer_specs(group)
        if len(layer_specs) != len(group.layer_names):
            raise AssertionError("AsymSpec group layer/spec cardinality mismatch.")
        for global_layer_name, spec in zip(group.layer_names, layer_specs):
            if spec.page_size_bytes != group_plan.page_size_bytes:
                raise ValueError(
                    "AsymSpec physical page size differs from the validated "
                    f"domain plan for {global_layer_name!r}."
                )
            total_bytes = group_plan.min_blocks_per_request * spec.page_size_bytes
            tensor_metadata = KVCacheTensor(
                size=total_bytes,
                shared_by=[global_layer_name],
            )
            tensors.append(
                AsymSpecPhysicalCacheTensorPlan(
                    domain=domain,
                    global_layer_name=global_layer_name,
                    physical_layer_name=_physical_layer_name(
                        global_plan, global_layer_name
                    ),
                    global_group_index=global_group_index,
                    kv_cache_spec=spec,
                    num_pages=group_plan.min_blocks_per_request,
                    page_size_bytes=spec.page_size_bytes,
                    total_bytes=total_bytes,
                    kv_cache_tensor=tensor_metadata,
                )
            )

    for domain, iterator in domain_group_plans.items():
        if next(iterator, None) is not None:
            raise ValueError(
                f"AsymSpec {domain.value} allocation plan has extra groups."
            )
    if len({tensor.global_layer_name for tensor in tensors}) != len(tensors):
        raise ValueError("AsymSpec physical cache plan contains duplicate layers.")
    return AsymSpecPhysicalCachePlan(tuple(tensors))


def allocate_asymspec_physical_cache_tensors(
    *,
    plan: AsymSpecPhysicalCachePlan,
    device: torch.device,
) -> AsymSpecPhysicalCacheRuntime:
    """Materialize unbound independent byte storage on ``device``.

    This mirrors the generic raw-cache allocator's zero-initialized byte
    storage, while intentionally bypassing its cross-group sharing semantics.
    No cache object is bound to an attention or recurrent layer here.
    """
    raw_tensors: dict[str, torch.Tensor] = {}
    for allocation in plan.tensors:
        raw_tensors[allocation.global_layer_name] = torch.zeros(
            allocation.total_bytes, dtype=torch.int8, device=device
        )

    if set(raw_tensors) != {tensor.global_layer_name for tensor in plan.tensors}:
        raise AssertionError("AsymSpec raw allocation omitted or duplicated a layer.")
    pointers = [tensor.untyped_storage().data_ptr() for tensor in raw_tensors.values()]
    if len(set(pointers)) != len(pointers):
        raise AssertionError("AsymSpec cache backing storage must be independent.")
    return AsymSpecPhysicalCacheRuntime(plan=plan, raw_tensors=raw_tensors)
