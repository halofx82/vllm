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
from .context_causal_policy import (
    AsymSpecContextCausalDecision,
    AsymSpecPolicyRow,
    context_causal_bootstrap_token,
    decide_context_causal_k2,
)
from .draft_forward import (
    AsymSpecDraftForwardResult,
    execute_asymspec_draft_forward,
    initialize_fresh_asymspec_view_state,
    qwen3_5_text_positions,
)
from .draft_signal import AsymSpecDraftSignal, build_asymspec_draft_signal
from .evidence_rollout import AsymSpecEvidenceCarrierRecord, AsymSpecEvidenceRollout
from .evidence_transfer import (
    AsymSpecEvidenceRecord,
    evidence_wrapper,
    select_evidence,
)
from .execution_metadata import (
    AsymSpecMetadataGroup,
    AsymSpecViewExecutionMetadata,
    build_asymspec_view_execution_metadata,
    recurrent_page_ids,
)
from .hybrid import AsymSpecHybridStateSpec
from .live_iteration import AsymSpecLiveIterationRuntime, begin_asymspec_live_iteration
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
from .target_diagnostic import (
    AsymSpecTargetRowCapture,
    capture_asymspec_verifier_rows,
)
from .verifier_bridge import (
    DIAGNOSTIC_CANDIDATE_TOKEN_IDS,
    DIAGNOSTIC_LIVE_BASE_LAG_TOKENS,
    DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_OUTPUT_PATH,
    DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS,
    DIAGNOSTIC_VERIFIER_OUTPUT_PATH,
    activate_asymspec_diagnostic_spec_tokens,
    arm_asymspec_live_spec_tokens,
    register_asymspec_diagnostic_spec_tokens,
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
    "AsymSpecEvidenceCarrierRecord",
    "AsymSpecEvidenceRecord",
    "AsymSpecEvidenceRollout",
    "AsymSpecContextCausalDecision",
    "AsymSpecPolicyRow",
    "AsymSpecTargetRowCapture",
    "capture_asymspec_verifier_rows",
    "context_causal_bootstrap_token",
    "decide_context_causal_k2",
    "DIAGNOSTIC_CANDIDATE_TOKEN_IDS",
    "DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS",
    "DIAGNOSTIC_LIVE_BASE_LAG_TOKENS",
    "DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS",
    "DIAGNOSTIC_LIVE_OUTPUT_PATH",
    "DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS",
    "DIAGNOSTIC_VERIFIER_OUTPUT_PATH",
    "AsymSpecLiveIterationRuntime",
    "arm_asymspec_live_spec_tokens",
    "begin_asymspec_live_iteration",
    "activate_asymspec_diagnostic_spec_tokens",
    "register_asymspec_diagnostic_spec_tokens",
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
    "evidence_wrapper",
    "select_evidence",
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
