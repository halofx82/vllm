# SPDX-License-Identifier: Apache-2.0
"""Frozen SCALE1 K=2 context-causal acceptance policy.

This module is deliberately side-effect free.  It makes one decision from
already-computed verifier and draft rows; TARGET bookkeeping and draft cache
promotion remain owned by the caller.  The equations and ordering are the
production ``context_causal_sample`` path from the frozen implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_K = 2
_LIFT_MIN = 1e-6
_BETA = 1.0
_GAMMA = 0.5


@dataclass(frozen=True)
class AsymSpecPolicyRow:
    """One sequential K=2 accept/reject decision."""

    position: int
    candidate_token_id: int
    target_token_id: int
    emitted_token_id: int
    accepted: bool
    exact_target_match: bool
    jsd: float | None
    gamma_eff: float | None
    cda_passed: bool | None
    context_lift: float | None


@dataclass(frozen=True)
class AsymSpecContextCausalDecision:
    """Policy output consumed once by TARGET, FULL, and BASE coordination."""

    accepted_count: int
    output_token_ids: tuple[int, ...]
    next_seed_token_id: int
    rows: tuple[AsymSpecPolicyRow, ...]
    used_bonus: bool
    bonus_token_id: int | None
    bonus_context_lift: float | None


def _require_row(row: torch.Tensor, name: str) -> torch.Tensor:
    if row.ndim != 1:
        raise ValueError(f"AsymSpec {name} logits must be one-dimensional.")
    if not row.is_floating_point():
        raise ValueError(f"AsymSpec {name} logits must be floating-point.")
    return row.float()


def _jsd(full: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Frozen JSD(full, base), evaluated in FP32 and clamped non-negative."""
    a_log = full.log_softmax(-1)
    b_log = base.log_softmax(-1)
    midpoint = torch.logaddexp(a_log, b_log) - math.log(2.0)
    return (
        (a_log.exp() * (a_log - midpoint)).sum()
        + (b_log.exp() * (b_log - midpoint)).sum()
    ).clamp_min(0.0) / 2.0


def _row_decision(
    *,
    position: int,
    candidate_token_id: int,
    target: torch.Tensor,
    full: torch.Tensor,
    base: torch.Tensor,
    lift_min: float,
) -> AsymSpecPolicyRow:
    target = _require_row(target, "TARGET")
    full = _require_row(full, "FULL")
    base = _require_row(base, "BASE")
    if target.shape != full.shape or target.shape != base.shape:
        raise ValueError("AsymSpec policy rows must share a vocabulary shape.")
    if not 0 <= candidate_token_id < target.numel():
        raise ValueError("AsymSpec candidate token ID is outside the vocabulary.")

    target_id = int(target.argmax().item())
    exact = candidate_token_id == target_id
    if exact:
        return AsymSpecPolicyRow(
            position=position,
            candidate_token_id=candidate_token_id,
            target_token_id=target_id,
            emitted_token_id=candidate_token_id,
            accepted=True,
            exact_target_match=True,
            jsd=None,
            gamma_eff=None,
            cda_passed=None,
            context_lift=None,
        )

    delta = full - base
    divergence = _jsd(full, base)
    gamma_eff = _GAMMA * torch.exp(-divergence)
    p_target = target.softmax(-1)[candidate_token_id]
    p_base = base.log_softmax(-1).exp()[candidate_token_id]
    cda = bool((p_target > gamma_eff * p_base).item())
    lift = delta[candidate_token_id] - delta[target_id]
    accepted = cda and bool((lift > lift_min).item())
    replacement = int((target + _BETA * delta).argmax().item())
    return AsymSpecPolicyRow(
        position=position,
        candidate_token_id=candidate_token_id,
        target_token_id=target_id,
        emitted_token_id=candidate_token_id if accepted else replacement,
        accepted=accepted,
        exact_target_match=False,
        jsd=float(divergence.item()),
        gamma_eff=float(gamma_eff.item()),
        cda_passed=cda,
        context_lift=float(lift.item()),
    )


def _fused_token(
    *, target: torch.Tensor, full: torch.Tensor, base: torch.Tensor, lift_min: float
) -> tuple[int, int, float]:
    """Frozen C1/rejection fusion: retain target unless lift is positive."""
    target = _require_row(target, "TARGET")
    full = _require_row(full, "FULL")
    base = _require_row(base, "BASE")
    if target.shape != full.shape or target.shape != base.shape:
        raise ValueError("AsymSpec fusion rows must share a vocabulary shape.")
    delta = full - base
    target_id = int(target.argmax().item())
    candidate_id = int((target + _BETA * delta).argmax().item())
    lift = float((delta[candidate_id] - delta[target_id]).item())
    emitted = (
        candidate_id if candidate_id == target_id or lift > lift_min else target_id
    )
    return emitted, candidate_id, lift


def decide_context_causal_k2(
    *,
    candidate_token_ids: tuple[int, int] | list[int],
    full_logits: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    base_logits: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    target_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
    bonus_full_logits: torch.Tensor | None = None,
    bonus_base_logits: torch.Tensor | None = None,
    target_only_bonus: bool = False,
    lift_min: float = _LIFT_MIN,
) -> AsymSpecContextCausalDecision:
    """Return frozen JSD+C1 output for one K=2 verifier transaction.

    ``bonus_*`` are required only when both draft positions are accepted. They
    are fresh canonical post-A/B rows, not candidate-position rows.  Keeping
    them explicit makes the policy pure while preserving frozen bonus order.
    """
    if not math.isfinite(lift_min) or lift_min < 0:
        raise ValueError(
            "AsymSpec context lift threshold must be finite and nonnegative."
        )
    candidates = tuple(int(token) for token in candidate_token_ids)
    if len(candidates) != _K or len(full_logits) != _K or len(base_logits) != _K:
        raise ValueError("AsymSpec context-causal policy requires exactly K=2 rows.")
    if len(target_logits) != _K + 1:
        raise ValueError("AsymSpec context-causal policy requires t0, t1, and t2.")

    rows = evaluate_context_causal_prefix_k2(
        candidate_token_ids=candidates,
        full_logits=full_logits,
        base_logits=base_logits,
        target_logits=target_logits[:2],
        lift_min=lift_min,
    )
    for position, row in enumerate(rows):
        if not row.accepted:
            return AsymSpecContextCausalDecision(
                accepted_count=position,
                output_token_ids=(*candidates[:position], row.emitted_token_id),
                next_seed_token_id=row.emitted_token_id,
                rows=rows[: position + 1],
                used_bonus=False,
                bonus_token_id=None,
                bonus_context_lift=None,
            )

    if target_only_bonus:
        bonus = int(_require_row(target_logits[2], "TARGET").argmax().item())
        return AsymSpecContextCausalDecision(
            accepted_count=2,
            output_token_ids=(*candidates, bonus),
            next_seed_token_id=bonus,
            rows=rows,
            used_bonus=True,
            bonus_token_id=bonus,
            bonus_context_lift=None,
        )
    if bonus_full_logits is None or bonus_base_logits is None:
        raise ValueError(
            "AsymSpec full acceptance requires fresh FULL/BASE bonus rows."
        )
    bonus, _candidate, bonus_lift = _fused_token(
        target=target_logits[2],
        full=bonus_full_logits,
        base=bonus_base_logits,
        lift_min=lift_min,
    )
    return AsymSpecContextCausalDecision(
        accepted_count=2,
        output_token_ids=(*candidates, bonus),
        next_seed_token_id=bonus,
        rows=rows,
        used_bonus=True,
        bonus_token_id=bonus,
        bonus_context_lift=bonus_lift,
    )


def evaluate_context_causal_prefix_k2(
    *,
    candidate_token_ids: tuple[int, int] | list[int],
    full_logits: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    base_logits: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    target_logits: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    lift_min: float = _LIFT_MIN,
) -> tuple[AsymSpecPolicyRow, ...]:
    """Evaluate the sequential draft prefix without touching bonus state."""
    if not math.isfinite(lift_min) or lift_min < 0:
        raise ValueError(
            "AsymSpec context lift threshold must be finite and nonnegative."
        )
    candidates = tuple(int(token) for token in candidate_token_ids)
    if len(candidates) != _K or len(full_logits) != _K or len(base_logits) != _K:
        raise ValueError("AsymSpec context-causal policy requires exactly K=2 rows.")
    if len(target_logits) != _K:
        raise ValueError("AsymSpec prefix policy requires t0 and t1.")
    rows: list[AsymSpecPolicyRow] = []
    for position in range(_K):
        row = _row_decision(
            position=position,
            candidate_token_id=candidates[position],
            target=target_logits[position],
            full=full_logits[position],
            base=base_logits[position],
            lift_min=lift_min,
        )
        rows.append(row)
        if not row.accepted:
            break
    return tuple(rows)


def context_causal_bootstrap_token(
    *,
    target_logits: torch.Tensor,
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    lift_min: float = _LIFT_MIN,
) -> tuple[int, int, int, float]:
    """Frozen prompt-boundary C1 token: ``argmax(t + (a-b))`` gated by lift."""
    emitted, candidate, lift = _fused_token(
        target=target_logits, full=full_logits, base=base_logits, lift_min=lift_min
    )
    target_id = int(_require_row(target_logits, "TARGET").argmax().item())
    return emitted, target_id, candidate, lift
