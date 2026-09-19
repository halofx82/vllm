# SPDX-License-Identifier: Apache-2.0
"""Request-local live AsymSpec seed-to-verifier iterations.

It turns the target sampler's ordinary, still-uncomputed output token into
canonical FULL/BASE work, keeps the FULL K=2 transaction live, and returns
its pair to the V1 scheduler. TARGET itself remains entirely on the normal
V1 path; diagnostic capture and fixed-outcome control are optional overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from .context_causal_policy import context_causal_bootstrap_token
from .draft_signal import AsymSpecDraftSignal, build_asymspec_draft_signal
from .evidence_rollout import AsymSpecEvidenceCarrierRecord, AsymSpecEvidenceRollout
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
    EVIDENCE_CARRIER_IN_MEMORY,
    EVIDENCE_CARRIER_OUTPUT_PATH,
)
from .views import AsymSpecViewRole

ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS = "specsteer_aug_prompt_ids"

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class AsymSpecLiveIterationRuntime:
    """Persistent request-local draft state for one live request."""

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
    # Empty outside explicit diagnostics. It is never required by production.
    output_path: str = ""
    base_lag_tokens: int = 0
    authoritative_suffix_token_ids: list[int] = field(default_factory=list)
    last_accepted_token_ids: tuple[int, ...] = ()
    full_promoted_tokens: int = 0
    full_rollbacks: int = 0
    base_authoritative_tokens_observed: int = 0
    base_catch_up_tokens: int = 0
    bootstrap_target_token_id: int | None = None
    bootstrap_fused_token_id: int | None = None
    bootstrap_context_lift: float | None = None
    identical_context_views: bool = False
    last_policy_decision: object | None = None
    evidence_carrier_record: AsymSpecEvidenceCarrierRecord | None = None
    _full_accept_bonus_prepared: bool = False
    _full_accept_bonus_target_only: bool = False

    @property
    def candidate_token_ids(self) -> tuple[int, int]:
        return self.proposal.candidate_token_ids

    def write_draft_capture(self) -> None:
        """Write rank-zero diagnostic vectors before native verifier execution."""
        if not self.output_path:
            return
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
                "bootstrap_target_token_id": self.bootstrap_target_token_id,
                "bootstrap_fused_token_id": self.bootstrap_fused_token_id,
                "bootstrap_context_lift": self.bootstrap_context_lift,
            },
            path,
        )

    def write_evidence_carrier_record(self, path_value: str) -> None:
        """Serialize the frozen carrier payload for the external runner."""
        if not path_value or self.evidence_carrier_record is None:
            return
        import json

        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.evidence_carrier_record.json()
        payload.update(
            {
                "request_id": self.request_id,
                "rollout_tokens": 8,
                "full_prompt_end": self.full.prompt_len,
                "base_prompt_end": self.base.prompt_len,
            }
        )
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def rollback_and_release(self) -> None:
        """Discard only candidate state; BASE remains canonical through the seed."""
        if self.proposal.transaction.active:
            self.proposal.transaction.rollback()
        self.request_state.release()

    @torch.no_grad()
    def prepare_full_accept_bonus(
        self, target_bonus_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Port frozen full-accept bonus preparation without choosing a token.

        Frozen SCALE1 first proves both K=2 draft positions acceptable, then
        promotes their already-executed FULL state and catches BASE up through
        the authoritative pair.  The resulting post-B rows are the inputs to
        the pure C1 bonus decision against TARGET's ``t2``.  This method owns
        only that established state preparation; it does not select or append
        the bonus token.
        """
        if self._full_accept_bonus_prepared:
            raise RuntimeError("AsymSpec FULL-accept bonus is already prepared.")
        if not self.proposal.transaction.active:
            raise RuntimeError("AsymSpec FULL candidate transaction is not active.")
        if target_bonus_logits.ndim != 1:
            raise ValueError("AsymSpec TARGET bonus logits must be one-dimensional.")
        accepted = self.candidate_token_ids
        result = self.proposal.transaction.promote(2)
        if result is None:
            raise AssertionError("AsymSpec full K=2 promotion returned no result.")
        self.full_promoted_tokens += 2
        full_bonus_logits = result.logits[0]
        # Frozen deferred ``advance_context_bonus`` has an explicit
        # evidence-free fast path: when FULL already agrees with TARGET's
        # bonus row, retain TARGET's bonus and leave BASE deferred. BASE still
        # observes A/B, then consumes them together with the bonus below.
        target_only = self.identical_context_views or int(
            full_bonus_logits.argmax().item()
        ) == int(target_bonus_logits.argmax().item())
        for token_id in accepted:
            self.base.observe_committed_token(token_id)
        self.base_authoritative_tokens_observed += 2
        if target_only:
            self._full_accept_bonus_target_only = True
        else:
            self.base.catch_up_base()
            self.base_catch_up_tokens += 2
            if self.request_state.base.pending_token_ids:
                raise AssertionError(
                    "AsymSpec BASE pending queue survived full accept."
                )
        self._full_accept_bonus_prepared = True
        return (
            full_bonus_logits,
            None if target_only else self.base.last_result.logits[0],
            target_only,
        )

    @torch.no_grad()
    def apply_target_outcome(
        self, *, accepted_count: int, next_seed_token_id: int
    ) -> tuple[int, int]:
        """Synchronize draft canonical state with one completed verifier.

        The caller has already selected TARGET's recurrent state and native
        bookkeeping will append the same accepted suffix plus ``next_seed``.
        FULL promotes its own disposable K=2 state, while BASE deliberately
        observes the authoritative suffix and catches up canonically; BASE's
        candidate scoring state is never promotable.
        """
        if accepted_count not in (0, 1, 2):
            raise ValueError("AsymSpec accepted count must be 0, 1, or 2.")
        if (
            not self.proposal.transaction.active
            and not self._full_accept_bonus_prepared
        ):
            raise RuntimeError("AsymSpec live FULL transaction is not active.")
        if (
            not self._full_accept_bonus_prepared
            and self.proposal.transaction.candidate_token_ids
            != self.candidate_token_ids
        ):
            raise RuntimeError("AsymSpec live FULL transaction pair mismatch.")

        accepted = self.candidate_token_ids[:accepted_count]
        if self._full_accept_bonus_prepared:
            if accepted_count != 2:
                raise RuntimeError(
                    "AsymSpec prepared full-accept bonus requires accepted_count=2."
                )
        else:
            self.proposal.transaction.promote(accepted_count)
            self.full_promoted_tokens += accepted_count
            self.full_rollbacks += int(accepted_count == 0)
        self.full.commit_token(next_seed_token_id)

        # BASE scores A only disposably.  Every accepted token and R therefore
        # enters via the proven authoritative deferred queue before one packed
        # canonical catch-up forward.
        pending_before_seed = len(self.request_state.base.pending_token_ids)
        base_suffix = (
            (int(next_seed_token_id),)
            if self._full_accept_bonus_prepared
            else (*accepted, int(next_seed_token_id))
        )
        for token_id in base_suffix:
            self.base.observe_committed_token(token_id)
        self.base_authoritative_tokens_observed += len(base_suffix)
        self.base.catch_up_base()
        self.base_catch_up_tokens += pending_before_seed + len(base_suffix)
        if self.request_state.base.pending_token_ids:
            raise AssertionError("AsymSpec BASE pending queue survived outcome.")
        expected_full_len = (
            self.base.canonical_len + self.request_state.full.augmentation_offset
        )
        if self.full.canonical_len != expected_full_len:
            raise AssertionError("AsymSpec FULL/BASE canonical coordinates diverged.")

        self.proposal = AsymSpecFullK2Proposer(self.full).propose_k2()
        self.base_score = AsymSpecBasePairScorer(self.base).score_pair(
            self.proposal.candidate_token_ids
        )
        self.signal = build_asymspec_draft_signal(
            candidate_token_ids=self.proposal.candidate_token_ids,
            a0=self.proposal.canonical_result.logits[0],
            b0=self.base_score.logits_before_a.logits[0],
            a1=self.proposal.candidate_results[0].logits[0],
            b1=self.base_score.logits_before_b.logits[0],
        )
        self.authoritative_suffix_token_ids.extend((*accepted, int(next_seed_token_id)))
        self.last_accepted_token_ids = accepted
        self._full_accept_bonus_prepared = False
        self._full_accept_bonus_target_only = False
        return self.candidate_token_ids

    def append_outcome_capture(
        self, *, accepted_count: int, next_seed_token_id: int
    ) -> None:
        """Persist compact post-transition facts for the live-chain harness."""
        if not self.output_path:
            return
        path = Path(self.output_path)
        if not path.exists():
            return
        capture = torch.load(path, weights_only=False)
        decision = self.last_policy_decision
        policy = None
        if decision is not None:
            policy = {
                "accepted_count": int(decision.accepted_count),
                "output_token_ids": list(decision.output_token_ids),
                "next_seed_token_id": int(decision.next_seed_token_id),
                "used_bonus": bool(decision.used_bonus),
                "bonus_token_id": decision.bonus_token_id,
                "bonus_context_lift": decision.bonus_context_lift,
                "rows": [
                    {
                        "position": row.position,
                        "candidate_token_id": row.candidate_token_id,
                        "target_token_id": row.target_token_id,
                        "emitted_token_id": row.emitted_token_id,
                        "accepted": row.accepted,
                        "exact_target_match": row.exact_target_match,
                        "jsd": row.jsd,
                        "gamma_eff": row.gamma_eff,
                        "cda_passed": row.cda_passed,
                        "context_lift": row.context_lift,
                    }
                    for row in decision.rows
                ],
            }
        capture.setdefault("live_sync_rows", []).append(
            {
                "accepted_count": int(accepted_count),
                "next_seed_token_id": int(next_seed_token_id),
                "accepted_token_ids": list(self.last_accepted_token_ids),
                "authoritative_suffix_token_ids": list(
                    self.authoritative_suffix_token_ids
                ),
                "full_canonical_len": self.full.canonical_len,
                "base_canonical_len": self.base.canonical_len,
                "base_observed_len": self.request_state.base.observed_len,
                "base_pending_tokens": list(self.request_state.base.pending_token_ids),
                "next_candidate_token_ids": list(self.candidate_token_ids),
                "full_promoted_tokens": self.full_promoted_tokens,
                "full_rollbacks": self.full_rollbacks,
                "base_authoritative_tokens_observed": (
                    self.base_authoritative_tokens_observed
                ),
                "base_catch_up_tokens": self.base_catch_up_tokens,
                "next_a0": self.signal.a0.detach().cpu().to(torch.bfloat16),
                "next_b0": self.signal.b0.detach().cpu().to(torch.bfloat16),
                "next_a1": self.signal.a1.detach().cpu().to(torch.bfloat16),
                "next_b1": self.signal.b1.detach().cpu().to(torch.bfloat16),
                "policy": policy,
            }
        )
        torch.save(capture, path)


def _live_prompt_ids(
    request: CachedRequestState,
) -> tuple[list[int], list[int], list[int], str, int] | None:
    """Resolve production FULL/BASE coordinates for one target request.

    Frozen SCALE1 uses the native compressed prompt for both views in direct
    engine mode. Its OpenAI server owns ``specsteer_aug_prompt_ids`` and
    attaches it to the sampling request after rendering the complete chat
    prompt.  Diagnostic overrides retain their separate schema.
    """
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    if extra_args and DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS in extra_args:
        required = (
            DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
            DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
            DIAGNOSTIC_LIVE_OUTPUT_PATH,
        )
        if any(key not in extra_args for key in required):
            raise ValueError(
                "AsymSpec live diagnostic is missing prompt/output metadata."
            )
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
    prompt_ids = request.prompt_token_ids
    if not prompt_ids:
        raise ValueError(
            "AsymSpec production requires token-ID prompts; prompt embeds "
            "do not define a draft token coordinate system."
        )
    compressed_ids = [int(token) for token in prompt_ids]
    if extra_args and ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS in extra_args:
        full = extra_args[ASYMSPEC_AUGMENTED_FULL_PROMPT_TOKEN_IDS]
        if not isinstance(full, (list, tuple)) or not full:
            raise ValueError(
                "AsymSpec server full prompt must be a non-empty token-ID list."
            )
        full_ids = [int(token) for token in full]
        if len(full_ids) < len(compressed_ids):
            raise ValueError(
                "AsymSpec full prompt cannot be shorter than compressed prompt."
            )
        # Capturing remains an optional diagnostic overlay; it does not alter
        # the server-owned production coordinate contract.
        output_path = str(extra_args.get(DIAGNOSTIC_LIVE_OUTPUT_PATH, ""))
        return full_ids, compressed_ids, [], output_path, 0
    return compressed_ids, compressed_ids.copy(), [], "", 0


@torch.no_grad()
def begin_asymspec_live_iteration(
    *,
    runner: GPUModelRunner,
    request: CachedRequestState,
    seed_token_id: int,
    bootstrap_target_logits: torch.Tensor | None = None,
) -> AsymSpecLiveIterationRuntime | None:
    """Consume V1's sampled seed in both draft views and make a live K=2 pair."""
    prompt_data = _live_prompt_ids(request)
    if prompt_data is None:
        return None
    full_ids, base_ids, preseed_ids, output_path, base_lag = prompt_data
    if seed_token_id < 0:
        raise ValueError("AsymSpec live seed must be non-negative.")
    views = getattr(runner, "asymspec_draft_views", None)
    if views is None:
        raise RuntimeError("AsymSpec live request requires initialized draft views.")
    full_capacity = views.full.state.cache_plan.max_model_len
    base_capacity = views.base.state.cache_plan.max_model_len
    required_full_capacity = len(full_ids) + len(preseed_ids) + 3
    required_base_capacity = len(base_ids) + len(preseed_ids) + 3
    if required_full_capacity > full_capacity or required_base_capacity > base_capacity:
        raise ValueError("AsymSpec live request prompt exceeds configured draft cache.")
    foundation = getattr(runner, "_asymspec_live_foundation", None)
    if foundation is None:
        # Draft module bindings are deliberately permanent.  Unlike the old
        # per-RPC diagnostics, a live V1 iteration must retain that one
        # binding set across requests instead of rebinding the module trees.
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
        runner._asymspec_live_foundation = foundation
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
    full_prefill = full.prefill(
        torch.tensor(full_ids, dtype=torch.int32, device=runner.device)
    )
    base_prefill = base.prefill(
        torch.tensor(base_ids, dtype=torch.int32, device=runner.device)
    )
    extra_args = getattr(request.sampling_params, "extra_args", None) or {}
    carrier_path = extra_args.get(EVIDENCE_CARRIER_OUTPUT_PATH)
    if carrier_path and not isinstance(carrier_path, str):
        raise ValueError("AsymSpec evidence carrier output path must be a string.")
    evidence_record = None
    speculative_config = runner.vllm_config.speculative_config
    if (
        speculative_config is not None
        and speculative_config.asymspec_evidence_mode == "one_shot"
        and (carrier_path or extra_args.get(EVIDENCE_CARRIER_IN_MEMORY))
    ):
        # Frozen Step-51 captures before bootstrap C1 mutates the seed.  The
        # resulting branch state is fully restored before normal production
        # bootstrap continues.
        evidence_record = AsymSpecEvidenceRollout(full=full, base=base).capture()
    bootstrap_target_token_id = None
    bootstrap_fused_token_id = None
    bootstrap_context_lift = None
    if bootstrap_target_logits is not None:
        # Frozen ``context_causal_bootstrap`` runs after native sampling has
        # produced its ordinary candidate, but before V1 commits that output.
        # Its FULL/BASE rows are the just-prefilled prompt-boundary rows.
        (
            emitted_seed,
            bootstrap_target_token_id,
            bootstrap_fused_token_id,
            bootstrap_context_lift,
        ) = context_causal_bootstrap_token(
            target_logits=bootstrap_target_logits,
            full_logits=full_prefill.logits[0],
            base_logits=base_prefill.logits[0],
        )
        seed_token_id = emitted_seed
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
    # All subsequent TARGET commits enter BASE through its authoritative
    # deferred queue.  The initial prompt and seed above are already
    # canonical in both compressed views.
    if not base_lag:
        base.begin_deferred_base()
    proposal = AsymSpecFullK2Proposer(full).propose_k2()
    base_score = AsymSpecBasePairScorer(base).score_pair(proposal.candidate_token_ids)
    signal = build_asymspec_draft_signal(
        candidate_token_ids=proposal.candidate_token_ids,
        a0=proposal.canonical_result.logits[0],
        b0=base_score.logits_before_a.logits[0],
        a1=proposal.candidate_results[0].logits[0],
        b1=base_score.logits_before_b.logits[0],
    )
    runtime = AsymSpecLiveIterationRuntime(
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
        authoritative_suffix_token_ids=[*preseed_ids, int(seed_token_id)],
        bootstrap_target_token_id=bootstrap_target_token_id,
        bootstrap_fused_token_id=bootstrap_fused_token_id,
        bootstrap_context_lift=bootstrap_context_lift,
        identical_context_views=full_ids == base_ids,
        evidence_carrier_record=evidence_record,
    )
    if carrier_path and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        runtime.write_evidence_carrier_record(carrier_path)
    return runtime
