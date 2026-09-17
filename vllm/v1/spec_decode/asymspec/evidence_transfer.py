# SPDX-License-Identifier: Apache-2.0
"""Frozen Step-51 one-shot evidence record helpers.

This module intentionally has no scheduler or request ownership.  The
external evidence runner owns the two-request workflow; workers only emit the
two disposable FULL/BASE rollout records consumed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

MIN_CF = 0.10
MIN_MARGIN = 0.05
ROLLOUT_TOKENS = 8

WRAPPER_PREFIX = (
    "<asymspec_long_context_evidence>\n"
    "The following text was retrieved from a longer context. It may be "
    "fragmentary or source-like. Use it as contextual evidence, then follow "
    "the user's original instruction and requested output format exactly.\n"
)
WRAPPER_SUFFIX = "\n</asymspec_long_context_evidence>"


def longest_exact_span(
    needle: Sequence[int], haystack: Sequence[int]
) -> dict[str, int | bool]:
    """Return frozen provenance metadata for a selected token span."""
    best_start, best_length = -1, 0
    for start in range(len(haystack)):
        length = 0
        limit = min(len(needle), len(haystack) - start)
        while length < limit and needle[length] == haystack[start + length]:
            length += 1
        if length > best_length:
            best_start, best_length = start, length
    return {"found": best_length > 0, "start": best_start, "length": best_length}


@dataclass(frozen=True)
class AsymSpecEvidenceRecord:
    """Frozen-compatible attributed Top-2/L8 carrier output."""

    full_top2: list[int]
    candidates: list[dict[str, Any]]
    selected_index: int
    selected_seed: int
    selected_token_ids: list[int]
    cf_selected: float
    cf_alt: float
    d_ctx: float
    min_cf: float
    min_margin: float
    attribution_pass: bool
    full_provenance: dict[str, int | bool]
    base_provenance: dict[str, int | bool]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def select_evidence(
    rollouts: Sequence[dict[str, Any]],
    *,
    full_top2: Sequence[int],
    full_source_ids: Sequence[int],
    base_source_ids: Sequence[int],
) -> AsymSpecEvidenceRecord:
    """Port frozen winner selection and strict null-calibrated gate."""
    if len(rollouts) != 2 or len(full_top2) != 2:
        raise RuntimeError("one-shot evidence transfer requires exactly FULL Top-2")
    rows = [dict(row) for row in rollouts]
    if any("context_llr" not in row or "token_ids" not in row for row in rows):
        raise RuntimeError("evidence rollout is missing CF/token IDs")
    winner = max(range(2), key=lambda i: float(rows[i]["context_llr"]))
    loser = 1 - winner
    selected_cf = float(rows[winner]["context_llr"])
    alternate_cf = float(rows[loser]["context_llr"])
    if not (isfinite(selected_cf) and isfinite(alternate_cf)):
        raise ValueError("evidence CF values must be finite")
    margin = selected_cf - alternate_cf
    selected_ids = [int(token) for token in rows[winner]["token_ids"]]
    return AsymSpecEvidenceRecord(
        full_top2=[int(token) for token in full_top2],
        candidates=rows,
        selected_index=winner,
        selected_seed=int(rows[winner]["seed"]),
        selected_token_ids=selected_ids,
        cf_selected=selected_cf,
        cf_alt=alternate_cf,
        d_ctx=margin,
        min_cf=MIN_CF,
        min_margin=MIN_MARGIN,
        attribution_pass=(selected_cf > MIN_CF and margin > MIN_MARGIN),
        full_provenance=longest_exact_span(selected_ids, full_source_ids),
        base_provenance=longest_exact_span(selected_ids, base_source_ids),
    )


def evidence_wrapper(text: str) -> str:
    """Return the immutable frozen prior-user evidence envelope."""
    if not text.strip():
        raise ValueError("evidence transport refuses an empty span")
    return WRAPPER_PREFIX + text.strip() + WRAPPER_SUFFIX
