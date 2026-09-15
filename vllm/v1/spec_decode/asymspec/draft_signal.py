# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw FULL-minus-BASE diagnostic signal containers.

This module deliberately has no steering or acceptance policy.  It just
retains the two draft-view rows already produced by the native proposer and
BASE scorer and materializes their FP32 difference for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AsymSpecDraftSignal:
    """Raw K=2 FULL/BASE logits and their FP32 context differences."""

    candidate_token_ids: tuple[int, int]
    a0: torch.Tensor
    b0: torch.Tensor
    d0: torch.Tensor
    a1: torch.Tensor
    b1: torch.Tensor
    d1: torch.Tensor


def build_asymspec_draft_signal(
    *,
    candidate_token_ids: tuple[int, int],
    a0: torch.Tensor,
    b0: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
) -> AsymSpecDraftSignal:
    """Build an immutable diagnostic signal without modifying its inputs."""
    rows = (a0, b0, a1, b1)
    if any(row.ndim != 1 for row in rows):
        raise ValueError("AsymSpec draft signal requires one-dimensional logits.")
    if len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError("AsymSpec FULL/BASE logit shapes must match.")
    if len(candidate_token_ids) != 2:
        raise ValueError("AsymSpec draft signal requires exactly K=2 candidates.")
    return AsymSpecDraftSignal(
        candidate_token_ids=tuple(int(token) for token in candidate_token_ids),
        a0=a0.detach(),
        b0=b0.detach(),
        d0=a0.float() - b0.float(),
        a1=a1.detach(),
        b1=b1.detach(),
        d1=a1.float() - b1.float(),
    )
