# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native AsymSpec components."""

from .base_scorer import (
    AsymSpecBasePairScore,
    AsymSpecBasePairScorer,
    AsymSpecBasePairScoringCounters,
)
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
    build_asymspec_domain_allocation_plans,
    compose_asymspec_global_cache_plan,
)
from .candidate_transaction import (
    AsymSpecCandidateTransactionCounters,
    AsymSpecFullCandidateTransaction,
)
from .canonical_driver import (
    AsymSpecCanonicalDraftDriver,
    AsymSpecCanonicalExecutionCounters,
)
from .draft_forward import (
    AsymSpecDraftForwardResult,
    execute_asymspec_draft_forward,
    initialize_fresh_asymspec_view_state,
    qwen3_5_text_positions,
)
from .draft_signal import AsymSpecDraftSignal, build_asymspec_draft_signal
from .execution_metadata import (
    AsymSpecMetadataGroup,
    AsymSpecViewExecutionMetadata,
    build_asymspec_view_execution_metadata,
    recurrent_page_ids,
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
from .proposer import AsymSpecFullK2Proposal, AsymSpecFullK2Proposer
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
    "AsymSpecBasePairScore",
    "AsymSpecBasePairScorer",
    "AsymSpecBasePairScoringCounters",
    "AsymSpecCanonicalDraftDriver",
    "AsymSpecCanonicalExecutionCounters",
    "AsymSpecCandidateTransactionCounters",
    "AsymSpecFullCandidateTransaction",
    "AsymSpecFullK2Proposal",
    "AsymSpecFullK2Proposer",
    "AsymSpecDraftForwardResult",
    "AsymSpecDraftSignal",
    "AsymSpecDraftCacheBinding",
    "AsymSpecDraftCacheBindingRuntime",
    "AsymSpecDraftLoadMemory",
    "AsymSpecMetadataGroup",
    "AsymSpecAllocatorCompatibility",
    "AsymSpecCacheDomain",
    "AsymSpecDomainAllocationPlan",
    "AsymSpecDomainAllocationPlans",
    "AsymSpecDomainGroupAllocationPlan",
    "AsymSpecCacheLayerBinding",
    "AsymSpecCacheNameRegistry",
    "AsymSpecCachePlan",
    "AsymSpecGlobalCachePlan",
    "build_asymspec_domain_allocation_plans",
    "compose_asymspec_global_cache_plan",
    "AsymSpecHybridStateSpec",
    "AsymSpecLogicalBlockPoolRuntime",
    "AsymSpecLogicalCacheGroup",
    "AsymSpecLogicalCacheGroupPlan",
    "AsymSpecLogicalCachePlan",
    "AsymSpecPhysicalCachePlan",
    "AsymSpecPhysicalCacheRuntime",
    "AsymSpecPhysicalCacheTensorPlan",
    "AsymSpecView",
    "AsymSpecViewExecutionMetadata",
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
    "build_asymspec_view_execution_metadata",
    "build_asymspec_draft_signal",
    "instantiate_asymspec_logical_block_pools",
    "create_asymspec_request_state",
    "execute_asymspec_draft_forward",
    "initialize_fresh_asymspec_view_state",
    "qwen3_5_text_positions",
    "recurrent_page_ids",
]
