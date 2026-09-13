# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native AsymSpec components."""

from .cache_plan import (
    AsymSpecAllocatorCompatibility,
    AsymSpecCacheDomain,
    AsymSpecCacheLayerBinding,
    AsymSpecCacheNameRegistry,
    AsymSpecCachePlan,
    AsymSpecDomainAllocationPlan,
    AsymSpecDomainAllocationPlans,
    AsymSpecDomainGroupAllocationPlan,
    AsymSpecGlobalCachePlan,
)
from .hybrid import AsymSpecHybridStateSpec
from .logical_cache import (
    AsymSpecLogicalBlockPoolRuntime,
    AsymSpecLogicalCacheGroup,
    AsymSpecLogicalCacheGroupPlan,
    AsymSpecLogicalCachePlan,
    allocate_synthetic_attention_blocks,
    build_asymspec_logical_cache_plan,
    instantiate_asymspec_logical_block_pools,
)
from .physical_cache import (
    AsymSpecPhysicalCachePlan,
    AsymSpecPhysicalCacheRuntime,
    AsymSpecPhysicalCacheTensorPlan,
    allocate_asymspec_physical_cache_tensors,
    build_asymspec_physical_cache_plan,
)
from .views import AsymSpecDraftViews, AsymSpecView, AsymSpecViewRole

__all__ = [
    "AsymSpecDraftViews",
    "AsymSpecAllocatorCompatibility",
    "AsymSpecCacheDomain",
    "AsymSpecDomainAllocationPlan",
    "AsymSpecDomainAllocationPlans",
    "AsymSpecDomainGroupAllocationPlan",
    "AsymSpecCacheLayerBinding",
    "AsymSpecCacheNameRegistry",
    "AsymSpecCachePlan",
    "AsymSpecGlobalCachePlan",
    "AsymSpecHybridStateSpec",
    "AsymSpecLogicalBlockPoolRuntime",
    "AsymSpecLogicalCacheGroup",
    "AsymSpecLogicalCacheGroupPlan",
    "AsymSpecLogicalCachePlan",
    "AsymSpecPhysicalCachePlan",
    "AsymSpecPhysicalCacheRuntime",
    "AsymSpecPhysicalCacheTensorPlan",
    "AsymSpecView",
    "AsymSpecViewRole",
    "allocate_asymspec_physical_cache_tensors",
    "allocate_synthetic_attention_blocks",
    "build_asymspec_logical_cache_plan",
    "build_asymspec_physical_cache_plan",
    "instantiate_asymspec_logical_block_pools",
]
