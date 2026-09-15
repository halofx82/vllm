# SPDX-License-Identifier: Apache-2.0
"""One diagnostic live AsymSpec seed-to-verifier iteration.

This deliberately stops before acceptance.  It turns the target sampler's
ordinary, still-uncomputed output token into canonical FULL/BASE work, keeps
the FULL K=2 transaction live, and returns only its pair to the V1 scheduler.
The target remains entirely on the normal V1 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .base_scorer import AsymSpecBasePairScore, AsymSpecBasePairScorer
from .cache_binding import (
    AsymSpecDraftCacheBindingRuntime,
    bind_asymspec_draft_caches,
)
from .cache_plan import (
    AsymSpecGlobalCachePlan,
    build_asymspec_domain_allocation_plans,
    compose_asymspec_global_cache_plan,
)
from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_signal import AsymSpecDraftSignal, build_asymspec_draft_signal
from .logical_cache import (
    AsymSpecLogicalBlockPoolRuntime,
    build_asymspec_logical_cache_plan,
    instantiate_asymspec_logical_block_pools,
)
from .physical_cache import (
    AsymSpecPhysicalCacheRuntime,
    allocate_asymspec_physical_cache_tensors,
    build_asymspec_physical_cache_plan,
)
from .proposer import AsymSpecFullK2Proposal, AsymSpecFullK2Proposer
from .request_state import AsymSpecRequestState, create_asymspec_request_state
from .verifier_bridge import (
    DIAGNOSTIC_LIVE_BASE_LAG_TOKENS,
    DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_OUTPUT_PATH,
    DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS,
)
from .views import AsymSpecViewRole

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class AsymSpecLiveIterationRuntime:
    """Draft state held only until the following disposable verifier forward."""

    request_id: str
    seed_token_id: int
    request_state: AsymSpecRequestState
    physical: AsymSpecPhysicalCacheRuntime
    logical: AsymSpecLogicalBlockPoolRuntime
    bindings: AsymSpecDraftCacheBindingRuntime
    full: AsymSpecCanonicalDraftDriver
    base: AsymSpecCanonicalDraftDriver
    proposal: AsymSpecFullK2Proposal
    base_score: AsymSpecBasePairScore
    signal: AsymSpecDraftSignal
    output_path: str
    base_lag_tokens: int = 0

    @property
    def candidate_token_ids(self) -> tuple[int, int]:
        return self.proposal.candidate_token_ids

    def write_draft_capture(self) -> None:
        """Write rank-zero diagnostic vectors before native verifier execution."""
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "request_id": self.request_id,
                "seed_token_id": self.seed_token_id,
                "candidate_token_ids": self.candidate_token_ids,
                "a0": self.signal.a0.detach().cpu().to(torch.bfloat16),
                "b0": self.signal.b0.detach().cpu().to(torch.bfloat16),
                "d0": self.signal.d0.detach().cpu(),
                "a1": self.signal.a1.detach().cpu().to(torch.bfloat16),
                "b1": self.signal.b1.detach().cpu().to(torch.bfloat16),
                "d1": self.signal.d1.detach().cpu(),
                "full_canonical_len": self.full.canonical_len,
                "base_canonical_len": self.base.canonical_len,
                "base_observed_len": self.request_state.base.observed_len,
                "base_pending_tokens": list(self.request_state.base.pending_token_ids),
                "base_lag_tokens": self.base_lag_tokens,
                "base_catch_up_tokens": self.base_score.catch_up_tokens,
                "full_transaction_active": self.proposal.transaction.active,
                "historical_replay_tokens": 0,
            },
            path,
        )

    def rollback_and_release(self) -> None:
        """Discard only candidate state; BASE remains canonical through the seed."""
        if self.proposal.transaction.active:
            self.proposal.transaction.rollback()
        self.request_state.release()


def _live_prompt_ids(
    request: CachedRequestState,
) -> tuple[list[int], list[int], list[int], str, int] | None:
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    if not extra_args or DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS not in extra_args:
        return None
    required = (
        DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
        DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
        DIAGNOSTIC_LIVE_OUTPUT_PATH,
    )
    if any(key not in extra_args for key in required):
        raise ValueError("AsymSpec live diagnostic is missing prompt/output metadata.")
    full = extra_args[DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS]
    base = extra_args[DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS]
    if not isinstance(full, (list, tuple)) or not isinstance(base, (list, tuple)):
        raise ValueError("AsymSpec live diagnostic prompts must be token-ID lists.")
    full_ids = [int(token) for token in full]
    base_ids = [int(token) for token in base]
    if not full_ids or not base_ids:
        raise ValueError("AsymSpec live diagnostic prompts must be non-empty.")
    lag = int(extra_args.get(DIAGNOSTIC_LIVE_BASE_LAG_TOKENS, 0))
    preseed = extra_args.get(DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS, ())
    if not isinstance(preseed, (list, tuple)):
        raise ValueError("AsymSpec live pre-seed commits must be token IDs.")
    preseed_ids = [int(token) for token in preseed]
    if lag < 0 or lag > len(preseed_ids):
        raise ValueError("AsymSpec live BASE lag must be within pre-seed commits.")
    return (
        full_ids,
        base_ids,
        preseed_ids,
        str(extra_args[DIAGNOSTIC_LIVE_OUTPUT_PATH]),
        lag,
    )


@torch.no_grad()
def begin_asymspec_live_iteration(
    *,
    runner: GPUModelRunner,
    request: CachedRequestState,
    seed_token_id: int,
) -> AsymSpecLiveIterationRuntime | None:
    """Consume V1's sampled seed in both draft views and make a live K=2 pair."""
    prompt_data = _live_prompt_ids(request)
    if prompt_data is None:
        return None
    full_ids, base_ids, preseed_ids, output_path, base_lag = prompt_data
    if seed_token_id < 0:
        raise ValueError("AsymSpec live diagnostic seed must be non-negative.")
    views = getattr(runner, "asymspec_draft_views", None)
    if views is None:
        raise RuntimeError("AsymSpec live diagnostic requires initialized draft views.")
    required_capacity = max(len(full_ids), len(base_ids)) + len(preseed_ids) + 3
    foundation = getattr(runner, "_asymspec_live_diagnostic_foundation", None)
    if foundation is None:
        # Draft module bindings are deliberately permanent.  Unlike the old
        # per-RPC diagnostics, a live V1 iteration must retain that one
        # binding set across requests instead of rebinding the module trees.
        full_capacity = views.full.state.cache_plan.max_model_len
        base_capacity = views.base.state.cache_plan.max_model_len
        if required_capacity > full_capacity or required_capacity > base_capacity:
            raise ValueError(
                "AsymSpec live diagnostic prompt exceeds configured cache."
            )
        target_specs = runner.get_kv_cache_spec()
        global_plan: AsymSpecGlobalCachePlan = compose_asymspec_global_cache_plan(
            vllm_config=runner.vllm_config,
            target_specs=target_specs,
            full_plan=views.full.state.cache_plan,
            base_plan=views.base.state.cache_plan,
        )
        domain_plans = build_asymspec_domain_allocation_plans(
            global_plan=global_plan,
            full_max_model_len=full_capacity,
            compressed_max_model_len=base_capacity,
        )
        physical = allocate_asymspec_physical_cache_tensors(
            plan=build_asymspec_physical_cache_plan(
                global_plan=global_plan, domain_plans=domain_plans
            ),
            device=runner.device,
        )
        logical = instantiate_asymspec_logical_block_pools(
            build_asymspec_logical_cache_plan(physical.plan)
        )
        bindings = bind_asymspec_draft_caches(
            views=views,
            physical_runtime=physical,
            global_plan=global_plan,
            logical_pools=logical,
            vllm_config=runner.vllm_config,
        )
        foundation = (physical, logical, bindings)
        runner._asymspec_live_diagnostic_foundation = foundation
    physical, logical, bindings = foundation
    state = create_asymspec_request_state(
        request_id=f"live-{request.req_id}",
        compressed_prompt_len=len(base_ids),
        full_prompt_len=len(full_ids),
        augmentation_offset=len(full_ids) - len(base_ids),
        logical_pools=logical,
        canonical_prompt_processed=False,
        device=runner.device,
    )
    common = dict(
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=runner.vllm_config,
        device=runner.device,
    )
    full = AsymSpecCanonicalDraftDriver(role=AsymSpecViewRole.FULL, **common)
    base = AsymSpecCanonicalDraftDriver(role=AsymSpecViewRole.BASE, **common)
    full.prefill(torch.tensor(full_ids, dtype=torch.int32, device=runner.device))
    base.prefill(torch.tensor(base_ids, dtype=torch.int32, device=runner.device))
    # The target seed is canonical-but-uncomputed only in V1 TARGET.  It is
    # immediately canonical and computed in both independent draft views.
    for token in preseed_ids:
        full.commit_token(token)
    full.commit_token(seed_token_id)
    if base_lag:
        for token in preseed_ids[:-base_lag]:
            base.commit_token(token)
        base.begin_deferred_base()
        for token in preseed_ids[-base_lag:]:
            base.observe_committed_token(token)
        base.observe_committed_token(seed_token_id)
    else:
        for token in preseed_ids:
            base.commit_token(token)
        base.commit_token(seed_token_id)
    proposal = AsymSpecFullK2Proposer(full).propose_k2()
    base_score = AsymSpecBasePairScorer(base).score_pair(proposal.candidate_token_ids)
    signal = build_asymspec_draft_signal(
        candidate_token_ids=proposal.candidate_token_ids,
        a0=proposal.canonical_result.logits[0],
        b0=base_score.logits_before_a.logits[0],
        a1=proposal.candidate_results[0].logits[0],
        b1=base_score.logits_before_b.logits[0],
    )
    return AsymSpecLiveIterationRuntime(
        request_id=request.req_id,
        seed_token_id=int(seed_token_id),
        request_state=state,
        physical=physical,
        logical=logical,
        bindings=bindings,
        full=full,
        base=base,
        proposal=proposal,
        base_score=base_score,
        signal=signal,
        output_path=output_path,
        base_lag_tokens=base_lag,
    )
