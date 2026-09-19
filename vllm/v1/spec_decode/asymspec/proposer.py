# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy FULL-only K=2 proposal construction for AsymSpec.

This is deliberately below the scheduler and verifier layers.  It consumes
the current canonical FULL logits for candidate A, executes A through the
existing disposable transaction, then obtains candidate B from A's logits.
The returned transaction remains active for a later caller to roll back or
promote explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base_scorer import AsymSpecBasePairScore, AsymSpecBasePairScorer
from .candidate_transaction import AsymSpecFullCandidateTransaction
from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_forward import (
    AsymSpecDraftForwardResult,
    execute_asymspec_dual_view_forward,
)
from .execution_metadata import build_asymspec_view_execution_metadata
from .views import AsymSpecViewRole


@dataclass(frozen=True)
class AsymSpecFullK2Proposal:
    """A greedy K=2 suffix plus its still-active disposable transaction."""

    candidate_token_ids: tuple[int, int]
    canonical_result: AsymSpecDraftForwardResult
    candidate_results: tuple[AsymSpecDraftForwardResult, AsymSpecDraftForwardResult]
    transaction: AsymSpecFullCandidateTransaction


class AsymSpecFullK2Proposer:
    """Construct a greedy K=2 proposal from one canonical FULL stream."""

    def __init__(self, driver: AsymSpecCanonicalDraftDriver) -> None:
        if driver.role is not AsymSpecViewRole.FULL:
            raise ValueError("AsymSpec K=2 proposer requires the FULL view.")
        self.driver = driver

    @staticmethod
    def _greedy_token(result: AsymSpecDraftForwardResult) -> int:
        logits = result.logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("AsymSpec greedy proposal requires one logit row.")
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def propose_k2(self) -> AsymSpecFullK2Proposal:
        """Produce A/B with exactly two disposable FULL forwards.

        A is the argmax of the already-retained canonical logits, so it needs
        no forward or replay.  Executing A produces the logits for B; B is
        then executed to complete the existing K=2 transaction.  The frozen
        K loop does not truncate at EOS, so this deterministic primitive also
        always constructs both positions.
        """
        canonical = self.driver.last_result
        candidate_a = self._greedy_token(canonical)
        transaction = AsymSpecFullCandidateTransaction(self.driver)
        result_a = transaction.begin_candidate(candidate_a)
        try:
            candidate_b = self._greedy_token(result_a)
            result_b = transaction.complete_candidate(candidate_b)
        except Exception:
            if transaction.active:
                transaction.rollback()
            raise
        if not transaction.ready_to_promote:
            raise AssertionError("AsymSpec K=2 proposal did not complete transaction.")
        return AsymSpecFullK2Proposal(
            candidate_token_ids=(candidate_a, candidate_b),
            canonical_result=canonical,
            candidate_results=(result_a, result_b),
            transaction=transaction,
        )

    @torch.no_grad()
    def propose_k2_dual(
        self, base_driver: AsymSpecCanonicalDraftDriver
    ) -> tuple[AsymSpecFullK2Proposal, AsymSpecBasePairScore]:
        """K=2 proposal with FULL/BASE candidate-A in one model traversal.

        The canonical update remains owned by the two persistent drivers.  At
        candidate A, however, both independently bound views are packed into
        the frozen inner dual-view traversal.  The BASE speculative write is
        restored immediately; the FULL transaction retains its normal
        promotable checkpoint and then executes B exactly as before.
        """
        if base_driver.role is not AsymSpecViewRole.BASE:
            raise ValueError("dual proposal requires the BASE driver")
        candidate_a = self._greedy_token(self.driver.last_result)
        transaction = AsymSpecFullCandidateTransaction(self.driver)
        base_scorer = AsymSpecBasePairScorer(base_driver)
        full_snap = transaction._snapshot_gdn_pages()
        base_snap = base_scorer._snapshot_gdn_pages()
        full_start = self.driver.canonical_len
        base_start = base_driver.canonical_len
        self.driver._ensure_attention_capacity(full_start + 1)
        base_driver._ensure_attention_capacity(base_start + 1)
        full_meta = build_asymspec_view_execution_metadata(
            request_state=self.driver.request_state, views=self.driver.views,
            cache_bindings=self.driver.cache_bindings,
            vllm_config=self.driver.vllm_config, role=AsymSpecViewRole.FULL,
            query_start=full_start, canonical_end=full_start, query_len=1,
            allow_uncommitted_end=True, allow_uncommitted_start=True,
            device=self.driver.device)
        base_meta = build_asymspec_view_execution_metadata(
            request_state=base_driver.request_state, views=base_driver.views,
            cache_bindings=base_driver.cache_bindings,
            vllm_config=base_driver.vllm_config, role=AsymSpecViewRole.BASE,
            query_start=base_start, canonical_end=base_start, query_len=1,
            allow_uncommitted_end=True, allow_uncommitted_start=True,
            device=base_driver.device)
        full_result, base_result = execute_asymspec_dual_view_forward(
            full_input_ids=torch.tensor([candidate_a], dtype=torch.int32,
                                        device=self.driver.device),
            base_input_ids=torch.tensor([candidate_a], dtype=torch.int32,
                                        device=base_driver.device),
            full_metadata=full_meta, base_metadata=base_meta,
            views=self.driver.views, cache_bindings=self.driver.cache_bindings)
        first_full_snap = transaction._snapshot_gdn_pages()
        base_scorer._restore_gdn_pages(base_snap)
        # Leave FULL at A's disposable state for complete_candidate(B).
        transaction._canonical_len = full_start
        transaction._canonical_snapshot = full_snap
        transaction._promotable_snapshots = (first_full_snap,)
        transaction._first_result = full_result
        transaction._first_token_id = candidate_a
        candidate_b = self._greedy_token(full_result)
        result_b = transaction.complete_candidate(candidate_b)
        base_logits_before_a = base_driver.last_result
        score = AsymSpecBasePairScore(
            candidate_token_ids=(candidate_a, candidate_b),
            logits_before_a=base_logits_before_a,
            logits_before_b=base_result,
            score_a=base_scorer._candidate_logit(base_logits_before_a, candidate_a),
            score_b=base_scorer._candidate_logit(base_result, candidate_b),
            catch_up_tokens=0, candidate_scoring_forwards=1)
        return (AsymSpecFullK2Proposal(
            candidate_token_ids=(candidate_a, candidate_b),
            canonical_result=self.driver.last_result,
            candidate_results=(full_result, result_b),
            transaction=transaction), score)
