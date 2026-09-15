# SPDX-License-Identifier: Apache-2.0
"""Opt-in, non-mutating target-logit capture for AsymSpec diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .verifier_bridge import (
    DIAGNOSTIC_CANDIDATE_TOKEN_IDS,
    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_OUTPUT_PATH,
    DIAGNOSTIC_VERIFIER_OUTPUT_PATH,
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
        writer = (not torch.distributed.is_initialized()
                  or torch.distributed.get_rank() == 0)
        if writer:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            torch.save({
                "token_ids": list(token_ids),
                "logits": logits.detach().cpu().to(torch.bfloat16),
            }, path)
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
        if not is_live and DIAGNOSTIC_CANDIDATE_TOKEN_IDS not in extra_args:
            continue
        expected = (
            [int(token) for token in scheduled]
            if is_live
            else [int(token) for token in extra_args[DIAGNOSTIC_CANDIDATE_TOKEN_IDS]]
        )
        if list(scheduled) != expected:
            raise RuntimeError(
                "Scheduled verifier tokens differ from supplied FULL pair."
            )
        output_key = (
            DIAGNOSTIC_LIVE_OUTPUT_PATH if is_live else DIAGNOSTIC_VERIFIER_OUTPUT_PATH
        )
        if output_key not in extra_args:
            raise ValueError("AsymSpec verifier diagnostic is missing an output path.")
        diagnostic.append(
            (request_id, expected, str(extra_args[output_key]))
        )
    if not diagnostic:
        return False
    if len(diagnostic) != 1 or spec_decode_metadata.num_draft_tokens.count(2) != 1:
        raise RuntimeError("AsymSpec verifier diagnostic supports one K=2 request.")

    request_id, tokens, output_path = diagnostic[0]
    target_indices = spec_decode_metadata.target_logits_indices.detach().cpu()
    bonus_indices = spec_decode_metadata.bonus_logits_indices.detach().cpu()
    if target_indices.numel() != 2 or bonus_indices.numel() != 1:
        raise RuntimeError(
            "AsymSpec verifier diagnostic received unexpected row counts."
        )
    writer = (not torch.distributed.is_initialized()
              or torch.distributed.get_rank() == 0)
    if writer:
        if logits.is_cuda:
            torch.cuda.synchronize(logits.device)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        capture = {
                "request_id": request_id,
                "candidate_token_ids": tokens,
                "scheduled_spec_decode_tokens": tokens,
                "target_logits_indices": target_indices.tolist(),
                "bonus_logits_indices": bonus_indices.tolist(),
                "t0": logits[int(target_indices[0])].detach().cpu().to(torch.bfloat16),
                "t1": logits[int(target_indices[1])].detach().cpu().to(torch.bfloat16),
                "t2": logits[int(bonus_indices[0])].detach().cpu().to(torch.bfloat16),
        }
        # The live draft coordinator writes a/b before V1 schedules the
        # verifier.  Merge rather than replace that one diagnostic bundle.
        if path.exists():
            existing = torch.load(path, weights_only=False)
            if tuple(existing.get("candidate_token_ids", ())) != tuple(tokens):
                raise RuntimeError("Live draft/target candidate pairs disagree.")
            existing.update(capture)
            capture = existing
        torch.save(capture, path)
    return True
