# SPDX-License-Identifier: Apache-2.0
"""AsymSpec-gated transport for a disposable V1 verifier diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.request import Request


DIAGNOSTIC_CANDIDATE_TOKEN_IDS = "asymspec_diagnostic_candidate_token_ids"
DIAGNOSTIC_VERIFIER_OUTPUT_PATH = "asymspec_diagnostic_verifier_output_path"
DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS = "asymspec_live_full_prompt_token_ids"
DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS = "asymspec_live_base_prompt_token_ids"
DIAGNOSTIC_LIVE_OUTPUT_PATH = "asymspec_live_output_path"
DIAGNOSTIC_LIVE_BASE_LAG_TOKENS = "asymspec_live_base_lag_tokens"
# Internal integration mode for the frozen C1/JSD policy.  Capture-only LIVE2
# requests intentionally omit this key and retain their sampler bypass.
DIAGNOSTIC_LIVE_CONTEXT_CAUSAL_POLICY = "asymspec_live_context_causal_policy"
DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS = (
    "asymspec_live_preseed_committed_token_ids"
)
# Test-only: selects the already-produced verifier row that supplies the
# canonical-but-uncomputed token for the next V1 iteration.  This is not an
# AsymSpec acceptance policy; callers explicitly provide 0, 1, or 2.
DIAGNOSTIC_FIXED_ACCEPTED_COUNT = "asymspec_diagnostic_fixed_accepted_count"
DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS = "asymspec_diagnostic_next_spec_token_ids"
DIAGNOSTIC_TARGET_CONTROL_OUTPUT_PATH = "asymspec_diagnostic_target_control_output_path"
DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT = "asymspec_diagnostic_arm_after_output_count"
# Test-only ordinary-decode control.  Each ID replaces exactly one ordinary
# sampler output *after* its real model forward and before normal V1
# bookkeeping.  It must never be used on a speculative verifier step.
DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS = (
    "asymspec_diagnostic_forced_decode_token_ids"
)
# Frozen Step-51 is an external two-request workflow.  The carrier runner
# supplies this rank-zero artifact destination; workers never own messages or
# construct the final target-only request.
EVIDENCE_CARRIER_OUTPUT_PATH = "asymspec_evidence_carrier_output_path"
ASYMSPEC_EXECUTION_MODE = "asymspec_execution_mode"
ASYMSPEC_TARGET_ONLY_EXECUTION_MODE = "target_only"


def is_asymspec_target_only_request(request: Request) -> bool:
    """Whether frozen's external evidence runner requested final TARGET-only work."""
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    return bool(
        extra_args
        and extra_args.get(ASYMSPEC_EXECUTION_MODE)
        == ASYMSPEC_TARGET_ONLY_EXECUTION_MODE
    )


def arm_asymspec_diagnostic_next_spec_tokens(
    request: Request, speculative_config: SpeculativeConfig | None
) -> bool:
    """Arm one test-only follow-up K=2 pair after a fixed V1 outcome.

    The next scheduler pass consumes the ordinary V1 output token (R) and
    this pair as ``[R, C, D]``.  This is only a state-parity diagnostic.
    """
    if speculative_config is None or speculative_config.method != "asymspec":
        return False
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    if not extra_args or DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS not in extra_args:
        return False
    tokens = extra_args[DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS]
    if (not isinstance(tokens, (list, tuple)) or len(tokens) != 2
            or not all(isinstance(token, int) and token >= 0 for token in tokens)):
        raise ValueError("AsymSpec next diagnostic pair requires two non-negative IDs.")
    if request.spec_token_ids:
        raise RuntimeError(
            "AsymSpec next diagnostic pair cannot overwrite spec tokens."
        )
    request.spec_token_ids = [int(token) for token in tokens]
    return True


def arm_asymspec_diagnostic_control_spec_tokens(
    request: Request, speculative_config: SpeculativeConfig | None
) -> bool:
    """Arm a target-control pair after ordinary V1 has produced its seed.

    This is solely the independent canonical control for the fixed-outcome
    state diagnostic: prefix prefill produces S, ordinary decode of S
    produces R, and the following verifier sees ``[R, C, D]``.  It never
    participates in regular AsymSpec or ordinary V1 requests.
    """
    if speculative_config is None or speculative_config.method != "asymspec":
        return False
    params = request.sampling_params
    extra_args = None if params is None else params.extra_args
    if not extra_args or DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT not in extra_args:
        return False
    if request.spec_token_ids:
        return False
    after = extra_args[DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT]
    if not isinstance(after, int) or after < 1:
        raise ValueError("AsymSpec target control needs a positive output count.")
    if request.num_output_tokens != after:
        return False
    return arm_asymspec_diagnostic_next_spec_tokens(request, speculative_config)


def register_asymspec_diagnostic_spec_tokens(
    request: Request, speculative_config: SpeculativeConfig | None
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
    if not extra_args:
        return False
    # A live pair is generated inside the TP workers only after V1 has
    # sampled its canonical-but-uncomputed seed.  It is installed from the
    # ModelRunnerOutput in ``arm_asymspec_live_spec_tokens`` below.
    if DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS in extra_args:
        required = (
            DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
            DIAGNOSTIC_LIVE_OUTPUT_PATH,
        )
        if any(key not in extra_args for key in required):
            raise ValueError("AsymSpec live diagnostic is missing required metadata.")
        return True
    if DIAGNOSTIC_CANDIDATE_TOKEN_IDS not in extra_args:
        return DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT in extra_args
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
        raise ValueError(
            "AsymSpec diagnostic verifier token IDs must be non-negative ints."
        )
    if (DIAGNOSTIC_VERIFIER_OUTPUT_PATH not in extra_args
            and DIAGNOSTIC_TARGET_CONTROL_OUTPUT_PATH not in extra_args):
        raise ValueError("AsymSpec diagnostic verifier requires an output path.")
    request._asymspec_diagnostic_pending_spec_token_ids = [
        int(token) for token in tokens
    ]
    return True


def arm_asymspec_live_spec_tokens(
    request: Request,
    speculative_config: SpeculativeConfig | None,
    candidate_token_ids: tuple[int, int] | list[int] | None,
) -> bool:
    """Install a TP-produced pair at V1's native uncomputed-seed boundary.

    The pair remains ordinary ``Request.spec_token_ids``: the scheduler owns
    all later scheduling, rollback, output-limit, and finish behavior.  This
    is the production bridge for ``method='asymspec'``; diagnostics only
    control how the pair was obtained or captured.
    """
    if speculative_config is None or speculative_config.method != "asymspec":
        return False
    if candidate_token_ids is None:
        raise RuntimeError("AsymSpec live request did not return a K=2 pair.")
    tokens = [int(token) for token in candidate_token_ids]
    if len(tokens) != 2 or any(token < 0 for token in tokens):
        raise ValueError("AsymSpec live request requires exactly two token IDs.")
    if request.spec_token_ids:
        raise RuntimeError("AsymSpec live request cannot overwrite spec tokens.")
    # Bootstrap has one output; later fixed-outcome rounds have an accepted
    # suffix plus R.  In either case V1 must leave exactly that final seed
    # canonical-but-uncomputed when the next pair is armed.
    if request.num_output_tokens < 1:
        raise RuntimeError("AsymSpec live request is missing its seed output.")
    expected_computed = (
        request.num_prompt_tokens + request.num_output_tokens - 1
    )
    if request.num_computed_tokens != expected_computed:
        raise RuntimeError(
            "AsymSpec live request must retain exactly one uncomputed seed: "
            f"computed={request.num_computed_tokens} expected={expected_computed} "
            f"prompt={request.num_prompt_tokens} outputs={request.num_output_tokens}."
        )
    request.spec_token_ids = tokens
    request._asymspec_live_seed_token_id = request._output_token_ids[-1]
    return True


def activate_asymspec_diagnostic_spec_tokens(
    request: Request, speculative_config: SpeculativeConfig | None
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
