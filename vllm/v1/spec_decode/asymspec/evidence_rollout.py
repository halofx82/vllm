# SPDX-License-Identifier: Apache-2.0
"""Disposable frozen Top-2/L8 evidence carrier rollouts.

This is deliberately independent of K=2 proposal/policy execution.  It uses
the already-bound persistent FULL and BASE drivers, snapshots only their
compact Mamba pages, and leaves canonical progress unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase

from .canonical_driver import AsymSpecCanonicalDraftDriver
from .draft_forward import AsymSpecDraftForwardResult
from .evidence_transfer import ROLLOUT_TOKENS
from .views import AsymSpecViewRole


@dataclass(frozen=True)
class _PageSnapshot:
    raw_tensor: torch.Tensor
    state: torch.Tensor


@dataclass(frozen=True)
class AsymSpecEvidenceCarrierRecord:
    """Worker-side raw record consumed by the external evidence runner."""

    full_top2: tuple[int, int]
    rollouts: tuple[dict[str, object], dict[str, object]]

    def json(self) -> dict[str, object]:
        return {"full_top2": list(self.full_top2), "rollouts": list(self.rollouts)}


class AsymSpecEvidenceRollout:
    """Port frozen disposable branches over native persistent draft views."""

    def __init__(
        self,
        *,
        full: AsymSpecCanonicalDraftDriver,
        base: AsymSpecCanonicalDraftDriver,
    ) -> None:
        if (
            full.role is not AsymSpecViewRole.FULL
            or base.role is not AsymSpecViewRole.BASE
        ):
            raise ValueError("evidence rollouts require FULL and BASE drivers")
        self.full = full
        self.base = base

    @staticmethod
    def _snapshot(driver: AsymSpecCanonicalDraftDriver) -> tuple[_PageSnapshot, ...]:
        bindings = (
            driver.cache_bindings.full_bindings
            if driver.role is AsymSpecViewRole.FULL
            else driver.cache_bindings.base_bindings
        )
        snapshots = tuple(
            _PageSnapshot(binding.raw_tensor, binding.raw_tensor.clone())
            for binding in bindings
            if isinstance(binding.module, MambaBase)
        )
        if not snapshots:
            raise RuntimeError("AsymSpec evidence rollout found no compact GDN pages")
        return snapshots

    @staticmethod
    def _restore(snapshots: tuple[_PageSnapshot, ...]) -> None:
        for snapshot in snapshots:
            snapshot.raw_tensor.copy_(snapshot.state)

    @staticmethod
    def _logprob(row: torch.Tensor, token_id: int) -> float:
        return float(row.float().log_softmax(dim=-1)[int(token_id)].item())

    @staticmethod
    def _greedy(row: torch.Tensor) -> int:
        return int(row.argmax(dim=-1).item())

    def _disposable_forward(
        self,
        driver: AsymSpecCanonicalDraftDriver,
        token_id: int,
        position: int,
        canonical_end: int,
    ) -> AsymSpecDraftForwardResult:
        driver._ensure_attention_capacity(position + 1)
        return driver._forward(
            torch.tensor([token_id], dtype=torch.int32, device=driver.device),
            query_start=position,
            canonical_end=canonical_end,
            allow_uncommitted_start=True,
        )

    @torch.no_grad()
    def capture(self) -> AsymSpecEvidenceCarrierRecord:
        """Capture two forced FULL branches and exact BASE teacher forcing.

        The frozen loop scores a token before consuming it, then extends from
        FULL greedily.  Both compact recurrent page sets are restored after
        every branch and again in ``finally`` for transactionality.
        """
        if not self.full._prefilled or not self.base._prefilled:
            raise RuntimeError("evidence rollout requires canonical prefill")
        full_initial = self.full.last_result.logits[0]
        base_initial = self.base.last_result.logits[0]
        top2 = tuple(int(token) for token in full_initial.topk(2).indices.tolist())
        full_snapshot = self._snapshot(self.full)
        base_snapshot = self._snapshot(self.base)
        full_end = self.full.canonical_len
        base_end = self.base.canonical_len
        records: list[dict[str, object]] = []
        try:
            for seed in top2:
                full_row, base_row = full_initial, base_initial
                token = seed
                tokens: list[int] = []
                full_logprobs: list[float] = []
                base_logprobs: list[float] = []
                for pos in range(ROLLOUT_TOKENS):
                    tokens.append(token)
                    full_logprobs.append(self._logprob(full_row, token))
                    base_logprobs.append(self._logprob(base_row, token))
                    if pos + 1 == ROLLOUT_TOKENS:
                        break
                    full_result = self._disposable_forward(
                        self.full, token, full_end + pos, full_end
                    )
                    base_result = self._disposable_forward(
                        self.base, token, base_end + pos, base_end
                    )
                    full_row, base_row = full_result.logits[0], base_result.logits[0]
                    token = self._greedy(full_row)
                records.append(
                    {
                        "seed": seed,
                        "token_ids": tokens,
                        "full_logprobs": full_logprobs,
                        "base_logprobs": base_logprobs,
                        "context_llr": float(sum(full_logprobs) - sum(base_logprobs)),
                    }
                )
                self._restore(full_snapshot)
                self._restore(base_snapshot)
        finally:
            self._restore(full_snapshot)
            self._restore(base_snapshot)
        return AsymSpecEvidenceCarrierRecord(
            full_top2=(top2[0], top2[1]),
            rollouts=(records[0], records[1]),
        )
