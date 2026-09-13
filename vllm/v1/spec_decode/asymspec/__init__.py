# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native AsymSpec components."""

from .cache_binding import (
    AsymSpecDraftCacheBinding,
    AsymSpecDraftCacheBindingRuntime,
    bind_asymspec_draft_caches,
)
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
from .request_state import (
    AsymSpecBaseCoordinates,
    AsymSpecCompressedCoordinates,
    AsymSpecFullCoordinates,
    AsymSpecRecurrentBlockSlots,
    AsymSpecRequestBlockTables,
    AsymSpecRequestState,
    create_asymspec_request_state,
)
from .views import (
    AsymSpecDraftLoadMemory,
    AsymSpecDraftViews,
    AsymSpecView,
    AsymSpecViewRole,
)

__all__ = [
    "AsymSpecDraftViews",
    "AsymSpecDraftCacheBinding",
    "AsymSpecDraftCacheBindingRuntime",
    "AsymSpecDraftLoadMemory",
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
    "AsymSpecBaseCoordinates",
    "AsymSpecCompressedCoordinates",
    "AsymSpecFullCoordinates",
    "AsymSpecRecurrentBlockSlots",
    "AsymSpecRequestBlockTables",
    "AsymSpecRequestState",
    "allocate_asymspec_physical_cache_tensors",
    "bind_asymspec_draft_caches",
    "allocate_synthetic_attention_blocks",
    "build_asymspec_logical_cache_plan",
    "build_asymspec_physical_cache_plan",
    "instantiate_asymspec_logical_block_pools",
    "create_asymspec_request_state",
]
