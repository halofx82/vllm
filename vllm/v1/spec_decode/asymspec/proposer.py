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

from .candidate_transaction import AsymSpecFullCandidateTransaction
from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_forward import AsymSpecDraftForwardResult
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
