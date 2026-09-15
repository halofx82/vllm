# SPDX-License-Identifier: Apache-2.0
"""Opt-in, non-mutating target-logit capture for AsymSpec diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


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
