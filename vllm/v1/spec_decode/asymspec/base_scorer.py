# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disposable BASE scoring for a supplied FULL K=2 proposal.

BASE never proposes or promotes candidate tokens.  It first consumes any
authoritative deferred suffix, retains the resulting canonical logits for
candidate A, then teacher-forces only A to obtain the logits for B.  The
compact recurrent pages are restored before returning; attention writes in
the uncommitted suffix are deliberately left to be overwritten by a later
authoritative write, matching the frozen SCALE1 path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase

from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_forward import AsymSpecDraftForwardResult
from .views import AsymSpecViewRole


@dataclass
class AsymSpecBasePairScoringCounters:
    """Accounting for BASE evidence work, separate from canonical progress."""

    catch_up_forward_calls: int = 0
    catch_up_tokens_processed: int = 0
    candidate_scoring_forward_calls: int = 0
    candidate_tokens_executed: int = 0
    replayed_historical_tokens: int = 0


@dataclass(frozen=True)
class AsymSpecBasePairScore:
    """Raw BASE evidence for the two supplied FULL candidate tokens."""

    candidate_token_ids: tuple[int, int]
    logits_before_a: AsymSpecDraftForwardResult
    logits_before_b: AsymSpecDraftForwardResult
    score_a: torch.Tensor
    score_b: torch.Tensor
    catch_up_tokens: int
    candidate_scoring_forwards: int


@dataclass(frozen=True)
class _GDNPageSnapshot:
    raw_tensor: torch.Tensor
    state: torch.Tensor


class AsymSpecBasePairScorer:
    """Score a supplied K=2 FULL pair without changing BASE canonical state."""

    def __init__(self, driver: AsymSpecCanonicalDraftDriver) -> None:
        if driver.role is not AsymSpecViewRole.BASE:
            raise ValueError("AsymSpec BASE pair scorer requires the BASE view.")
        self.driver = driver
        self.counters = AsymSpecBasePairScoringCounters()

    def _snapshot_gdn_pages(self) -> tuple[_GDNPageSnapshot, ...]:
        snapshots: list[_GDNPageSnapshot] = []
        for binding in self.driver.cache_bindings.base_bindings:
            if isinstance(binding.module, MambaBase):
                snapshots.append(
                    _GDNPageSnapshot(binding.raw_tensor, binding.raw_tensor.clone())
                )
        if not snapshots:
            raise RuntimeError("AsymSpec BASE pair scorer found no GDN pages.")
        return tuple(snapshots)

    @staticmethod
    def _restore_gdn_pages(snapshots: tuple[_GDNPageSnapshot, ...]) -> None:
        for snapshot in snapshots:
            snapshot.raw_tensor.copy_(snapshot.state)

    @staticmethod
    def _candidate_logit(
        result: AsymSpecDraftForwardResult, token_id: int
    ) -> torch.Tensor:
        logits = result.logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("AsymSpec BASE scoring requires one logit row.")
        return logits[0, int(token_id)].detach()

    @torch.no_grad()
    def score_pair(
        self, candidate_token_ids: tuple[int, int] | list[int]
    ) -> AsymSpecBasePairScore:
        """Score A at canonical BASE state and B after disposable A.

        Deferred BASE work is canonical and therefore committed before the
        disposable snapshot.  Candidate A is the sole speculative BASE
        forward: its output predicts B, while B itself is never executed.
        """
        tokens = tuple(int(token) for token in candidate_token_ids)
        if len(tokens) != 2:
            raise ValueError("AsymSpec BASE pair scoring requires K=2 tokens.")
        if not self.driver._prefilled:
            raise RuntimeError("AsymSpec BASE must be prefilled before scoring.")

        before_catchup_calls = self.driver.counters.catch_up_forward_calls
        before_catchup_tokens = self.driver.counters.catch_up_tokens_processed
        self.driver.catch_up_base()
        self.counters.catch_up_forward_calls += (
            self.driver.counters.catch_up_forward_calls - before_catchup_calls
        )
        catch_up_tokens = (
            self.driver.counters.catch_up_tokens_processed - before_catchup_tokens
        )
        self.counters.catch_up_tokens_processed += catch_up_tokens

        state = self.driver.request_state
        if state.base.canonical_len != state.base.observed_len:
            raise RuntimeError("AsymSpec BASE is not canonical after catch-up.")
        if state.base.pending_token_ids:
            raise AssertionError("AsymSpec BASE pending queue survived catch-up.")

        # ``last_result`` is either the retained canonical result or the
        # just-completed packed catch-up result.  Never replay canonical input
        # merely to recreate position-zero logits.
        logits_before_a = self.driver.last_result
        canonical_len = self.driver.canonical_len
        canonical_snapshot = self._snapshot_gdn_pages()
        try:
            self.driver._ensure_attention_capacity(canonical_len + 1)
            logits_before_b = self.driver._forward(
                torch.tensor([tokens[0]], dtype=torch.int32, device=self.driver.device),
                query_start=canonical_len,
                canonical_end=canonical_len,
                allow_uncommitted_start=True,
            )
        except Exception:
            self._restore_gdn_pages(canonical_snapshot)
            raise
        self._restore_gdn_pages(canonical_snapshot)
        self.counters.candidate_scoring_forward_calls += 1
        self.counters.candidate_tokens_executed += 1
        return AsymSpecBasePairScore(
            candidate_token_ids=(tokens[0], tokens[1]),
            logits_before_a=logits_before_a,
            logits_before_b=logits_before_b,
            score_a=self._candidate_logit(logits_before_a, tokens[0]),
            score_b=self._candidate_logit(logits_before_b, tokens[1]),
            catch_up_tokens=catch_up_tokens,
            candidate_scoring_forwards=1,
        )
