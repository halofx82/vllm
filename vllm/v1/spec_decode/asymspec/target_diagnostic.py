# SPDX-License-Identifier: Apache-2.0
"""Opt-in, non-mutating target-logit capture for AsymSpec diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .context_causal_policy import (
    AsymSpecContextCausalDecision,
    decide_context_causal_k2,
    evaluate_context_causal_prefix_k2,
)
from .verifier_bridge import (
    DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT,
    DIAGNOSTIC_CANDIDATE_TOKEN_IDS,
    DIAGNOSTIC_FIXED_ACCEPTED_COUNT,
    DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS,
    DIAGNOSTIC_LIVE_CONTEXT_CAUSAL_POLICY,
    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_OUTPUT_PATH,
    DIAGNOSTIC_TARGET_CONTROL_OUTPUT_PATH,
    DIAGNOSTIC_VERIFIER_OUTPUT_PATH,
)


@dataclass(frozen=True)
class AsymSpecFixedAcceptanceOutcome:
    """An externally selected, native-shaped K=2 sampler result.

    ``output_token_ids`` intentionally has V1's normal padded speculative
    result layout.  Passing it through the ordinary model-runner and
    scheduler bookkeeping is what commits target cache/recurrent state.
    """

    request_id: str
    accepted_count: int
    next_seed_token_id: int
    output_token_ids: torch.Tensor


@dataclass(frozen=True)
class AsymSpecContextCausalOutcome:
    """One frozen-policy result in native V1 sampler-output layout."""

    request_id: str
    decision: AsymSpecContextCausalDecision
    output_token_ids: torch.Tensor

    @property
    def accepted_count(self) -> int:
        return self.decision.accepted_count

    @property
    def next_seed_token_id(self) -> int:
        return self.decision.next_seed_token_id


def build_asymspec_context_causal_outcome(
    *,
    scheduler_output: SchedulerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    logits: torch.Tensor | None,
    requests: dict[str, CachedRequestState],
    active_live: dict[str, object] | None,
    is_asymspec: bool,
) -> AsymSpecContextCausalOutcome | None:
    """Adapt frozen pure C1/JSD policy to the existing isolated live runtime.

    This is deliberately narrower than generic sampling: only an explicitly
    marked AsymSpec live request can enter it.  The returned padded tensor is
    consumed by unchanged V1 speculative bookkeeping just like the existing
    fixed-count diagnostic output.
    """
    if (
        not is_asymspec
        or spec_decode_metadata is None
        or logits is None
        or not active_live
    ):
        return None
    matches: list[tuple[str, list[int], object]] = []
    for (
        request_id,
        candidate_ids,
    ) in scheduler_output.scheduled_spec_decode_tokens.items():
        state = requests.get(request_id)
        params = None if state is None else state.sampling_params
        extra = None if params is None else params.extra_args
        runtime = active_live.get(request_id)
        if not extra or not extra.get(DIAGNOSTIC_LIVE_CONTEXT_CAUSAL_POLICY):
            continue
        if runtime is None:
            raise RuntimeError("AsymSpec context-causal request has no live runtime.")
        if len(candidate_ids) != 2:
            raise RuntimeError("AsymSpec context-causal policy requires K=2.")
        matches.append((request_id, [int(x) for x in candidate_ids], runtime))
    if not matches:
        return None
    if len(matches) != 1 or spec_decode_metadata.num_draft_tokens != [2]:
        raise RuntimeError("AsymSpec context-causal policy supports one K=2 request.")
    if (
        spec_decode_metadata.target_logits_indices.numel() != 2
        or spec_decode_metadata.bonus_logits_indices.numel() != 1
    ):
        raise RuntimeError(
            "AsymSpec context-causal policy received invalid target rows."
        )

    request_id, candidates, runtime = matches[0]
    signal = runtime.signal
    if tuple(candidates) != tuple(signal.candidate_token_ids):
        raise RuntimeError("AsymSpec target/FULL candidate pair mismatch.")
    t0, t1 = (
        logits[int(spec_decode_metadata.target_logits_indices[i])] for i in range(2)
    )
    t2 = logits[int(spec_decode_metadata.bonus_logits_indices[0])]
    prefix = evaluate_context_causal_prefix_k2(
        candidate_token_ids=tuple(candidates),
        full_logits=(signal.a0, signal.a1),
        base_logits=(signal.b0, signal.b1),
        target_logits=(t0, t1),
    )
    if len(prefix) < 2 or not all(row.accepted for row in prefix):
        # A bonus is irrelevant after rejection; placeholders satisfy the
        # pure function's complete signature without affecting its branch.
        decision = decide_context_causal_k2(
            candidate_token_ids=tuple(candidates),
            full_logits=(signal.a0, signal.a1),
            base_logits=(signal.b0, signal.b1),
            target_logits=(t0, t1, t2),
            bonus_full_logits=signal.a1,
            bonus_base_logits=signal.b1,
        )
    else:
        # Frozen ``advance_context_bonus`` ordering: promote/catch up the
        # accepted pair only after sequential acceptance is established, then
        # make the C1 bonus decision from fresh post-B FULL/BASE rows.
        a_bonus, b_bonus, target_only_bonus = runtime.prepare_full_accept_bonus(t2)
        decision = decide_context_causal_k2(
            candidate_token_ids=tuple(candidates),
            full_logits=(signal.a0, signal.a1),
            base_logits=(signal.b0, signal.b1),
            target_logits=(t0, t1, t2),
            bonus_full_logits=a_bonus,
            bonus_base_logits=b_bonus,
            target_only_bonus=target_only_bonus,
        )
    output = torch.full(
        (1, spec_decode_metadata.max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    output[0, : len(decision.output_token_ids)] = torch.tensor(
        decision.output_token_ids, dtype=torch.int32, device=logits.device
    )
    # Retained only as compact diagnostic data.  The coordinator below remains
    # the sole owner of state mutation and consumes this exact decision once.
    runtime.last_policy_decision = decision
    return AsymSpecContextCausalOutcome(
        request_id=request_id, decision=decision, output_token_ids=output
    )


def force_asymspec_diagnostic_decode_outputs(
    *,
    sampled_token_ids: torch.Tensor,
    scheduler_output: SchedulerOutput,
    requests: dict[str, CachedRequestState],
    req_id_to_index: dict[str, int],
    is_asymspec: bool,
) -> bool:
    """Force requested ordinary sampler outputs for a control request.

    The target forward and native sampler have already run when this helper
    is called.  We replace only the emitted ID before V1's existing hybrid
    state update and scheduler bookkeeping consume it.  This keeps the
    canonical control on ordinary decode transitions rather than constructing
    accepted tokens in prefill or by mutating a Request.
    """
    if not is_asymspec:
        return False

    matches: list[tuple[str, int | None]] = []
    for request_id in scheduler_output.num_scheduled_tokens:
        if request_id in scheduler_output.scheduled_spec_decode_tokens:
            continue
        state = requests.get(request_id)
        params = None if state is None else state.sampling_params
        extra_args = None if params is None else params.extra_args
        if not extra_args or DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS not in extra_args:
            continue
        # The bootstrap sampler must create S itself.  Force only subsequent
        # ordinary decode outputs, after S is canonical-but-uncomputed.
        if not state.output_token_ids:
            continue
        tokens = extra_args[DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS]
        index = getattr(state, "_asymspec_forced_decode_index", 0)
        if not isinstance(tokens, (list, tuple)) or any(
            token is not None and (not isinstance(token, int) or token < 0)
            for token in tokens
        ):
            raise ValueError(
                "AsymSpec forced decode tokens must be non-negative integers or None."
            )
        if index < len(tokens):
            token = tokens[index]
            matches.append((request_id, None if token is None else int(token)))

    if not matches:
        return False
    if len(matches) != 1:
        raise RuntimeError(
            "AsymSpec forced decode control supports one isolated request."
        )
    request_id, token_id = matches[0]
    request_index = req_id_to_index.get(request_id)
    if request_index is None or sampled_token_ids.ndim != 2:
        raise RuntimeError("AsymSpec forced decode received invalid sampler output.")
    if sampled_token_ids.shape[1] != 1:
        raise RuntimeError(
            "AsymSpec forced decode only supports ordinary one-token steps."
        )
    state = requests[request_id]
    state._asymspec_forced_decode_index = (
        getattr(state, "_asymspec_forced_decode_index", 0) + 1
    )
    if token_id is None:
        return False
    sampled_token_ids[request_index, 0] = token_id
    return True


def persist_asymspec_fixed_acceptance_outcome(
    outcome: AsymSpecFixedAcceptanceOutcome,
    requests: dict[str, CachedRequestState],
) -> None:
    """Add compact fixed-outcome facts to the existing live capture bundle."""
    state = requests.get(outcome.request_id)
    params = None if state is None else state.sampling_params
    extra_args = None if params is None else params.extra_args
    if not extra_args or DIAGNOSTIC_LIVE_OUTPUT_PATH not in extra_args:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    path = Path(str(extra_args[DIAGNOSTIC_LIVE_OUTPUT_PATH]))
    if not path.exists():
        return
    capture = torch.load(path, weights_only=False)
    capture.update(
        {
            "fixed_accepted_count": outcome.accepted_count,
            "fixed_next_seed_token_id": outcome.next_seed_token_id,
            "fixed_sampler_output_token_ids": outcome.output_token_ids.detach().cpu(),
        }
    )
    torch.save(capture, path)


def build_asymspec_fixed_acceptance_outcome(
    *,
    scheduler_output: SchedulerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    logits: torch.Tensor | None,
    requests: dict[str, CachedRequestState],
    is_asymspec: bool,
) -> AsymSpecFixedAcceptanceOutcome | None:
    """Build one explicit K=2 outcome without applying acceptance policy.

    The diagnostic is deliberately restricted to the isolated live verifier
    shape.  It chooses ``R`` greedily from t0/t1/t2 after an externally
    supplied accepted count, then returns the exact padded representation a
    normal rejection sampler would hand to V1 bookkeeping.
    """
    if not is_asymspec or spec_decode_metadata is None or logits is None:
        return None

    matches: list[tuple[str, int, list[int]]] = []
    for (
        request_id,
        candidate_ids,
    ) in scheduler_output.scheduled_spec_decode_tokens.items():
        state = requests.get(request_id)
        params = None if state is None else state.sampling_params
        extra_args = None if params is None else params.extra_args
        if not extra_args or DIAGNOSTIC_FIXED_ACCEPTED_COUNT not in extra_args:
            continue
        requested_counts = extra_args[DIAGNOSTIC_FIXED_ACCEPTED_COUNT]
        outcome_index = getattr(state, "_asymspec_fixed_acceptance_index", 0)
        if isinstance(requested_counts, (list, tuple)):
            if outcome_index >= len(requested_counts):
                continue
            accepted_count = requested_counts[outcome_index]
        else:
            if outcome_index > 0:
                continue
            accepted_count = requested_counts
        if not isinstance(accepted_count, int) or accepted_count not in (0, 1, 2):
            raise ValueError(
                "AsymSpec fixed acceptance count must be one of 0, 1, or 2."
            )
        if len(candidate_ids) != 2:
            raise RuntimeError("AsymSpec fixed acceptance requires a K=2 pair.")
        matches.append(
            (request_id, accepted_count, [int(token) for token in candidate_ids])
        )

    if not matches:
        return None
    if len(matches) != 1 or len(spec_decode_metadata.num_draft_tokens) != 1:
        raise RuntimeError(
            "AsymSpec fixed acceptance diagnostic supports one isolated K=2 request."
        )
    request_id, accepted_count, candidate_ids = matches[0]
    if spec_decode_metadata.num_draft_tokens != [2]:
        raise RuntimeError("AsymSpec fixed acceptance received non-K=2 metadata.")
    if (
        spec_decode_metadata.target_logits_indices.numel() != 2
        or spec_decode_metadata.bonus_logits_indices.numel() != 1
    ):
        raise RuntimeError(
            "AsymSpec fixed acceptance received unexpected verifier rows."
        )

    row_index = (
        spec_decode_metadata.target_logits_indices[accepted_count]
        if accepted_count < 2
        else spec_decode_metadata.bonus_logits_indices[0]
    )
    next_seed = int(logits[int(row_index)].argmax().item())
    output_token_ids = torch.full(
        (1, spec_decode_metadata.max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    committed = candidate_ids[:accepted_count] + [next_seed]
    output_token_ids[0, : len(committed)] = torch.tensor(
        committed, dtype=torch.int32, device=logits.device
    )
    return AsymSpecFixedAcceptanceOutcome(
        request_id=request_id,
        accepted_count=accepted_count,
        next_seed_token_id=next_seed,
        output_token_ids=output_token_ids,
    )


if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass(frozen=True)
class AsymSpecTargetRowCapture:
    """Persist one ordinary sampler input row and leave it unmodified.

    This is intentionally a SamplingParams logits processor rather than a
    model-runner hook.  Consequently it exercises the ordinary V1 target KV
    request/cache path and is absent from all requests that do not explicitly
    install it.  The processor has no effect on sampling: it returns the
    original tensor object unchanged.
    """

    output_path: str

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        writer = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        if writer:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            torch.save(
                {
                    "token_ids": list(token_ids),
                    "logits": logits.detach().cpu().to(torch.bfloat16),
                },
                path,
            )
        return logits


def capture_asymspec_verifier_rows(
    *,
    scheduler_output: SchedulerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    logits: torch.Tensor | None,
    requests: dict[str, CachedRequestState],
    is_asymspec: bool,
) -> bool:
    """Persist t0/t1/t2 selected by the normal V1 verifier metadata."""
    if not is_asymspec or spec_decode_metadata is None or logits is None:
        return False
    diagnostic: list[tuple[str, list[int], str]] = []
    for request_id, scheduled in scheduler_output.scheduled_spec_decode_tokens.items():
        state = requests.get(request_id)
        params = None if state is None else state.sampling_params
        extra_args = None if params is None else params.extra_args
        if not extra_args:
            continue
        is_live = DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS in extra_args
        is_target_control = DIAGNOSTIC_TARGET_CONTROL_OUTPUT_PATH in extra_args
        if (
            not is_live
            and DIAGNOSTIC_CANDIDATE_TOKEN_IDS not in extra_args
            and DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT not in extra_args
        ):
            continue
        expected = (
            [int(token) for token in scheduled]
            if is_live or DIAGNOSTIC_CANDIDATE_TOKEN_IDS not in extra_args
            else [int(token) for token in extra_args[DIAGNOSTIC_CANDIDATE_TOKEN_IDS]]
        )
        if list(scheduled) != expected:
            raise RuntimeError(
                "Scheduled verifier tokens differ from supplied FULL pair."
            )
        output_key = (
            DIAGNOSTIC_LIVE_OUTPUT_PATH
            if is_live
            else (
                DIAGNOSTIC_TARGET_CONTROL_OUTPUT_PATH
                if is_target_control
                else DIAGNOSTIC_VERIFIER_OUTPUT_PATH
            )
        )
        if output_key not in extra_args:
            raise ValueError("AsymSpec verifier diagnostic is missing an output path.")
        diagnostic.append((request_id, expected, str(extra_args[output_key])))
    if not diagnostic:
        return False
    if len(diagnostic) != 1 or spec_decode_metadata.num_draft_tokens.count(2) != 1:
        raise RuntimeError("AsymSpec verifier diagnostic supports one K=2 request.")

    request_id, tokens, output_path = diagnostic[0]
    state = requests[request_id]
    is_follow_up = bool(
        getattr(state, "_asymspec_fixed_acceptance_consumed", False)
        or getattr(state, "_asymspec_context_causal_outcome_consumed", False)
    )
    target_indices = spec_decode_metadata.target_logits_indices.detach().cpu()
    bonus_indices = spec_decode_metadata.bonus_logits_indices.detach().cpu()
    if target_indices.numel() != 2 or bonus_indices.numel() != 1:
        raise RuntimeError(
            "AsymSpec verifier diagnostic received unexpected row counts."
        )
    writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if writer:
        if logits.is_cuda:
            torch.cuda.synchronize(logits.device)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row_prefix = "next_" if is_follow_up else ""
        capture = {
            "request_id": request_id,
            # Read-only state facts for the ordinary-decode control.  These
            # are captured before the disposable verifier is finished, never
            # used to drive request state.
            f"{row_prefix}output_token_ids": list(
                getattr(state, "output_token_ids", ())
            ),
            f"{row_prefix}num_computed_tokens": getattr(
                state, "num_computed_tokens", None
            ),
            f"{row_prefix}candidate_token_ids": tokens,
            f"{row_prefix}scheduled_spec_decode_tokens": tokens,
            f"{row_prefix}target_logits_indices": target_indices.tolist(),
            f"{row_prefix}bonus_logits_indices": bonus_indices.tolist(),
            f"{row_prefix}t0": logits[int(target_indices[0])]
            .detach()
            .cpu()
            .to(torch.bfloat16),
            f"{row_prefix}t1": logits[int(target_indices[1])]
            .detach()
            .cpu()
            .to(torch.bfloat16),
            f"{row_prefix}t2": logits[int(bonus_indices[0])]
            .detach()
            .cpu()
            .to(torch.bfloat16),
        }
        # The live draft coordinator writes a/b before V1 schedules the
        # verifier.  Merge rather than replace that one diagnostic bundle.
        if path.exists():
            existing = torch.load(path, weights_only=False)
            if not is_follow_up and tuple(
                existing.get("candidate_token_ids", ())
            ) != tuple(tokens):
                raise RuntimeError("Live draft/target candidate pairs disagree.")
            existing.update(capture)
            capture = existing
        extra_args = state.sampling_params.extra_args
        requested_counts = extra_args.get(DIAGNOSTIC_FIXED_ACCEPTED_COUNT)
        chain_index = getattr(state, "_asymspec_fixed_acceptance_index", 0)
        has_pending_fixed_outcome = requested_counts is not None and (
            (
                isinstance(requested_counts, (list, tuple))
                and chain_index < len(requested_counts)
            )
            or (not isinstance(requested_counts, (list, tuple)) and chain_index == 0)
        )
        if has_pending_fixed_outcome:
            capture.setdefault("fixed_chain_rows", []).append(
                {
                    "candidate_token_ids": list(tokens),
                    "t0": logits[int(target_indices[0])]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16),
                    "t1": logits[int(target_indices[1])]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16),
                    "t2": logits[int(bonus_indices[0])]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16),
                }
            )
        torch.save(capture, path)
    return True
