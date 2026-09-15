# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disposable FULL K=2 candidate state for AsymSpec.

The frozen compact-GDN implementation snapshots the active recurrent pages
before candidate execution, saves a promotable copy after every candidate
token, then restores the canonical pages immediately. Attention writes at
positions beyond the canonical end are deliberately left in place: a later
canonical write either exposes an accepted position or overwrites a rejected
suffix. This module preserves that ownership boundary without a proposer or
verifier dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase

from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_forward import AsymSpecDraftForwardResult
from .views import AsymSpecViewRole


@dataclass
class AsymSpecCandidateTransactionCounters:
    """Disposable candidate accounting, separate from canonical AR work."""

    transactions: int = 0
    candidate_tokens_executed: int = 0
    promoted_tokens: int = 0
    rollbacks: int = 0
    replayed_historical_tokens: int = 0


@dataclass(frozen=True)
class _GDNPageSnapshot:
    """A compact view's active recurrent backing bytes at one candidate point."""

    raw_tensor: torch.Tensor
    state: torch.Tensor


class AsymSpecFullCandidateTransaction:
    """One externally supplied, disposable FULL candidate pair.

    Snapshotting the compact committed/speculative pages preserves the frozen
    state protocol. The FULL canonical coordinate advances only on promotion.
    """

    def __init__(self, driver: AsymSpecCanonicalDraftDriver) -> None:
        if driver.role is not AsymSpecViewRole.FULL:
            raise ValueError("AsymSpec candidate transactions require FULL.")
        self.driver = driver
        self.counters = AsymSpecCandidateTransactionCounters()
        self._canonical_len: int | None = None
        self._candidate_token_ids: tuple[int, int] | None = None
        self._canonical_snapshot: tuple[_GDNPageSnapshot, ...] = ()
        self._promotable_snapshots: tuple[tuple[_GDNPageSnapshot, ...], ...] = ()
        self._results: tuple[AsymSpecDraftForwardResult, ...] = ()

    @property
    def active(self) -> bool:
        return self._canonical_len is not None

    @property
    def candidate_token_ids(self) -> tuple[int, int] | None:
        return self._candidate_token_ids

    def _snapshot_gdn_pages(self) -> tuple[_GDNPageSnapshot, ...]:
        """Copy only FULL's compact recurrent pages, never attention KV."""
        snapshots: list[_GDNPageSnapshot] = []
        for binding in self.driver.cache_bindings.full_bindings:
            if isinstance(binding.module, MambaBase):
                # The request-local compact pool is null/committed/spec1/spec2.
                # This raw backing is bounded to those pages for one request.
                snapshots.append(
                    _GDNPageSnapshot(binding.raw_tensor, binding.raw_tensor.clone())
                )
        if not snapshots:
            raise RuntimeError(
                "AsymSpec FULL candidate transaction found no GDN pages."
            )
        return tuple(snapshots)

    @staticmethod
    def _restore_gdn_pages(snapshots: tuple[_GDNPageSnapshot, ...]) -> None:
        for snapshot in snapshots:
            snapshot.raw_tensor.copy_(snapshot.state)

    def _clear(self) -> None:
        self._canonical_len = None
        self._candidate_token_ids = None
        self._canonical_snapshot = ()
        self._promotable_snapshots = ()
        self._results = ()

    def execute_candidate(
        self, candidate_token_ids: tuple[int, int] | list[int]
    ) -> tuple[AsymSpecDraftForwardResult, AsymSpecDraftForwardResult]:
        """Execute a K=2 suffix, capture promotable GDN, restore canonical GDN."""
        if self.active:
            raise RuntimeError("AsymSpec FULL candidate transaction is already active.")
        if not self.driver._prefilled:
            raise RuntimeError("AsymSpec FULL must be prefetched before candidates.")
        tokens = tuple(int(token) for token in candidate_token_ids)
        if len(tokens) != 2:
            raise ValueError("AsymSpec FULL candidate transaction requires K=2 tokens.")
        canonical_len = self.driver.canonical_len
        canonical_snapshot = self._snapshot_gdn_pages()
        results: list[AsymSpecDraftForwardResult] = []
        promotable: list[tuple[_GDNPageSnapshot, ...]] = []
        try:
            for offset, token in enumerate(tokens):
                start = canonical_len + offset
                self.driver._ensure_attention_capacity(start + 1)
                result = self.driver._forward(
                    torch.tensor([token], dtype=torch.int32, device=self.driver.device),
                    query_start=start,
                    canonical_end=start,
                    allow_uncommitted_start=True,
                )
                results.append(result)
                promotable.append(self._snapshot_gdn_pages())
        except Exception:
            self._restore_gdn_pages(canonical_snapshot)
            raise
        self._restore_gdn_pages(canonical_snapshot)
        self._canonical_len = canonical_len
        self._candidate_token_ids = tokens
        self._canonical_snapshot = canonical_snapshot
        self._promotable_snapshots = tuple(promotable)
        self._results = tuple(results)
        self.counters.transactions += 1
        self.counters.candidate_tokens_executed += 2
        return self._results[0], self._results[1]

    def rollback(self) -> None:
        """Discard the active suffix and retain the original canonical state."""
        if not self.active:
            raise RuntimeError("AsymSpec FULL candidate transaction is not active.")
        self._restore_gdn_pages(self._canonical_snapshot)
        self.counters.rollbacks += 1
        self._clear()

    def promote(self, accepted_tokens: int) -> AsymSpecDraftForwardResult | None:
        """Commit zero, one, or two already-executed candidate tokens."""
        if not self.active:
            raise RuntimeError("AsymSpec FULL candidate transaction is not active.")
        if accepted_tokens not in (0, 1, 2):
            raise ValueError(
                "AsymSpec FULL candidate promotion accepts only 0, 1, or 2."
            )
        if accepted_tokens == 0:
            self.rollback()
            return None
        assert self._canonical_len is not None
        if self.driver.canonical_len != self._canonical_len:
            raise RuntimeError(
                "AsymSpec FULL canonical state changed during candidate work."
            )
        self._restore_gdn_pages(self._promotable_snapshots[accepted_tokens - 1])
        self.driver.request_state.advance_full(accepted_tokens)
        self.counters.promoted_tokens += accepted_tokens
        result = self._results[accepted_tokens - 1]
        self._clear()
        return result
