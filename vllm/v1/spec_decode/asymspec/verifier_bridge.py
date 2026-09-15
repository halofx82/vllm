# SPDX-License-Identifier: Apache-2.0
"""AsymSpec-gated transport for a disposable V1 verifier diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.request import Request


DIAGNOSTIC_CANDIDATE_TOKEN_IDS = "asymspec_diagnostic_candidate_token_ids"
DIAGNOSTIC_VERIFIER_OUTPUT_PATH = "asymspec_diagnostic_verifier_output_path"


def register_asymspec_diagnostic_spec_tokens(
    request: "Request", speculative_config: "SpeculativeConfig | None"
) -> bool:
    """Register one explicit FULL K=2 pair for normal V1 scheduling.

    The pair is held until the target has completed its compressed prompt.
    V1 then transports and clears ``Request.spec_token_ids`` through its
    ordinary scheduler path.
    """
    if speculative_config is None or speculative_config.method != "asymspec":
        return False
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    if not extra_args or DIAGNOSTIC_CANDIDATE_TOKEN_IDS not in extra_args:
        return False
    if request.spec_token_ids or hasattr(
        request, "_asymspec_diagnostic_pending_spec_token_ids"
    ):
        raise RuntimeError(
            "AsymSpec diagnostic verifier cannot overwrite speculative token IDs."
        )
    tokens = extra_args[DIAGNOSTIC_CANDIDATE_TOKEN_IDS]
    if not isinstance(tokens, (list, tuple)) or len(tokens) != 2:
        raise ValueError("AsymSpec diagnostic verifier requires exactly K=2 tokens.")
    if not all(isinstance(token, int) and token >= 0 for token in tokens):
        raise ValueError("AsymSpec diagnostic verifier token IDs must be non-negative ints.")
    if DIAGNOSTIC_VERIFIER_OUTPUT_PATH not in extra_args:
        raise ValueError("AsymSpec diagnostic verifier requires an output path.")
    request._asymspec_diagnostic_pending_spec_token_ids = [
        int(token) for token in tokens
    ]
    return True


def activate_asymspec_diagnostic_spec_tokens(
    request: "Request", speculative_config: "SpeculativeConfig | None"
) -> bool:
    """Move a registered pair into V1's normal request-side spec field."""
    if speculative_config is None or speculative_config.method != "asymspec":
        return False
    pending = getattr(request, "_asymspec_diagnostic_pending_spec_token_ids", None)
    if pending is None or request.spec_token_ids:
        return False
    if request.num_computed_tokens < request.num_prompt_tokens:
        return False
    request.spec_token_ids = pending
    del request._asymspec_diagnostic_pending_spec_token_ids
    return True
