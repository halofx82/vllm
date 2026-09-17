# SPDX-License-Identifier: Apache-2.0
"""Frozen SCALE1 C1/JSD policy parity cases, without cache/runtime state."""

import importlib.util
from pathlib import Path

import pytest
import torch

from vllm.v1.spec_decode.asymspec.context_causal_policy import (
    context_causal_bootstrap_token,
    decide_context_causal_k2,
)
from vllm.v1.spec_decode.asymspec.target_diagnostic import (
    build_asymspec_context_causal_outcome,
)
from vllm.v1.spec_decode.asymspec.verifier_bridge import (
    DIAGNOSTIC_LIVE_CONTEXT_CAUSAL_POLICY,
)


def _frozen_sampler_module():
    """Load the frozen executable policy without importing its runtime tree."""
    path = (
        Path(__file__).resolve().parents[4]
        / "AsymSpec-0.28.0"
        / "vllm_specsteer"
        / "vllm_0_28"
        / "specsteer_sampler.py"
    )
    if not path.is_file():
        pytest.skip("frozen SCALE1 executable is unavailable")
    spec = importlib.util.spec_from_file_location("frozen_context_causal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(*, candidates=(0, 1), target=None, full=None, base=None, bonus=None):
    target = target or [
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([0.0, 0.0, 5.0]),
    ]
    full = full or [target[0].clone(), target[1].clone()]
    base = base or [target[0].clone(), target[1].clone()]
    bonus = bonus or (target[2].clone(), target[2].clone())
    return decide_context_causal_k2(
        candidate_token_ids=candidates,
        full_logits=full,
        base_logits=base,
        target_logits=target,
        bonus_full_logits=bonus[0],
        bonus_base_logits=bonus[1],
    )


def test_null_delta_does_not_steer_non_target_candidate():
    target = [
        torch.tensor([1.0, 1.5, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
    ]
    equal = torch.tensor([1.5, 1.0, 0.0])
    decision = _decision(
        candidates=(0, 1),
        target=target,
        full=[equal, target[1]],
        base=[equal, target[1]],
    )
    assert decision.accepted_count == 0
    assert decision.next_seed_token_id == 1
    assert decision.rows[0].context_lift == 0.0


def test_exact_target_match_bypasses_cda_and_reaches_bonus():
    decision = _decision()
    assert decision.accepted_count == 2
    assert [row.exact_target_match for row in decision.rows] == [True, True]
    assert decision.used_bonus
    assert decision.output_token_ids == (0, 1, 2)


def test_context_supported_non_target_is_accepted():
    target = [
        torch.tensor([0.0, 5.0, 4.9]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
    ]
    full = [torch.tensor([20.0, 0.0, 0.0]), target[1]]
    base = [torch.tensor([-20.0, 0.0, 0.0]), target[1]]
    decision = _decision(candidates=(0, 1), target=target, full=full, base=base)
    assert decision.accepted_count == 2
    assert decision.rows[0].cda_passed
    assert decision.rows[0].context_lift > 1e-6


def test_lift_failure_replaces_with_fused_target_safe_token():
    target = [
        torch.tensor([5.0, 4.9, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
    ]
    # Candidate one has adequate target probability, but no *relative*
    # FULL-vs-BASE context lift and therefore cannot steer.
    full = [torch.tensor([0.0, 0.0, 0.0]), target[1]]
    base = [torch.tensor([0.0, 0.0, 0.0]), target[1]]
    decision = _decision(candidates=(1, 1), target=target, full=full, base=base)
    assert decision.accepted_count == 0
    assert decision.next_seed_token_id == 0


def test_full_accept_uses_context_causal_bonus_not_raw_target_argmax():
    target = [
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([5.0, 4.9, 0.0]),
    ]
    bonus_full = torch.tensor([0.0, 20.0, 0.0])
    bonus_base = torch.tensor([0.0, -20.0, 0.0])
    decision = _decision(target=target, bonus=(bonus_full, bonus_base))
    assert decision.accepted_count == 2
    assert decision.bonus_token_id == 1
    assert decision.next_seed_token_id == 1
    assert decision.bonus_context_lift > 1e-6


def test_bootstrap_null_delta_retains_target():
    target = torch.tensor([1.0, 5.0, 4.0])
    emitted, target_id, candidate, lift = context_causal_bootstrap_token(
        target_logits=target,
        full_logits=torch.tensor([3.0, 0.0, 9.0]),
        base_logits=torch.tensor([3.0, 0.0, 9.0]),
    )
    assert (emitted, target_id, candidate, lift) == (1, 1, 1, 0.0)


def test_bootstrap_frozen_replace_and_retain_fixtures():
    # Exact vectors from frozen ``test_context_causal_sampler``.  These are
    # prompt-boundary C1 cases, so no K=2 transaction is involved.
    retain = context_causal_bootstrap_token(
        target_logits=torch.tensor([5.0, 4.0]),
        full_logits=torch.tensor([0.0, 1.0]),
        base_logits=torch.zeros(2),
        lift_min=1.1,
    )
    replace = context_causal_bootstrap_token(
        target_logits=torch.tensor([5.0, 4.0]),
        full_logits=torch.tensor([0.0, 3.0]),
        base_logits=torch.zeros(2),
    )
    assert retain == (0, 0, 0, 0.0)
    assert replace[:3] == (1, 0, 1)
    assert replace[3] > 1e-6


def test_frozen_partial_k2_rejection_emits_only_accepted_prefix_and_fusion():
    target = [
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([0.0, 0.0, 5.0]),
    ]
    # A is exact. B disagrees with no delta evidence, so frozen sequential
    # sampling accepts only A and emits target-safe fusion token 1.
    decision = _decision(
        candidates=(0, 2),
        target=target,
        full=[target[0], torch.tensor([0.0, 0.0, 0.0])],
        base=[target[0], torch.tensor([0.0, 0.0, 0.0])],
    )
    assert decision.accepted_count == 1
    assert decision.output_token_ids == (0, 1)
    assert [row.accepted for row in decision.rows] == [True, False]
    assert not decision.used_bonus


def test_frozen_full_accept_bonus_fixture_uses_fresh_rows_only():
    target = [
        torch.tensor([5.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 5.0, 0.0]),
    ]
    decision = _decision(
        candidates=(0, 1),
        target=target,
        full=[target[0], target[1]],
        base=[target[0], target[1]],
        bonus=(torch.tensor([0.0, 0.0, 0.0, 20.0]), torch.zeros(4)),
    )
    assert decision.accepted_count == 2
    assert decision.output_token_ids == (0, 1, 3)
    assert decision.used_bonus


def test_frozen_target_only_bonus_retains_target_row_without_delta_fusion():
    target = [
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([0.0, 5.0, 4.9]),
    ]
    decision = decide_context_causal_k2(
        candidate_token_ids=(0, 1),
        full_logits=(target[0], target[1]),
        base_logits=(target[0], target[1]),
        target_logits=target,
        target_only_bonus=True,
    )
    assert decision.output_token_ids == (0, 1, 1)
    assert decision.bonus_context_lift is None


@pytest.mark.parametrize(
    ("candidates", "target", "full", "base", "bonus"),
    [
        (
            (0, 2),
            [
                torch.tensor([5.0, 0.0, 0.0]),
                torch.tensor([0.0, 5.0, 0.0]),
                torch.tensor([0.0, 0.0, 5.0]),
            ],
            [torch.tensor([5.0, 0.0, 0.0]), torch.zeros(3)],
            [torch.tensor([5.0, 0.0, 0.0]), torch.zeros(3)],
            (torch.zeros(3), torch.zeros(3)),
        ),
        (
            (0, 1),
            [
                torch.tensor([0.0, 5.0, 4.9]),
                torch.tensor([0.0, 5.0, 0.0]),
                torch.tensor([0.0, 5.0, 0.0]),
            ],
            [torch.tensor([20.0, 0.0, 0.0]), torch.tensor([0.0, 5.0, 0.0])],
            [torch.tensor([-20.0, 0.0, 0.0]), torch.tensor([0.0, 5.0, 0.0])],
            (torch.tensor([0.0, 5.0, 0.0]), torch.tensor([0.0, 5.0, 0.0])),
        ),
        (
            (0, 1),
            [
                torch.tensor([5.0, 0.0, 0.0]),
                torch.tensor([0.0, 5.0, 0.0]),
                torch.tensor([5.0, 4.9, 0.0]),
            ],
            [torch.tensor([5.0, 0.0, 0.0]), torch.tensor([0.0, 5.0, 0.0])],
            [torch.tensor([5.0, 0.0, 0.0]), torch.tensor([0.0, 5.0, 0.0])],
            (torch.tensor([0.0, 20.0, 0.0]), torch.tensor([0.0, -20.0, 0.0])),
        ),
    ],
)
def test_native_pure_policy_matches_frozen_executable_fixtures(
    candidates, target, full, base, bonus
):
    """Direct parity against frozen ``context_causal_sample`` branch output."""
    frozen = _frozen_sampler_module()
    frozen_output = frozen.context_causal_sample(
        torch.tensor(candidates),
        [2],
        2,
        torch.stack(target[:2]),
        torch.stack(full),
        torch.stack(base),
        target[2].unsqueeze(0),
        lambda _drafts, _target_bonus=None: (
            bonus[0].unsqueeze(0),
            bonus[1].unsqueeze(0),
        ),
        1e-6,
        1.0,
        0.5,
        None,
    )
    native = decide_context_causal_k2(
        candidate_token_ids=candidates,
        full_logits=full,
        base_logits=base,
        target_logits=target,
        bonus_full_logits=bonus[0],
        bonus_base_logits=bonus[1],
    )
    assert native.output_token_ids == tuple(
        int(token) for token in frozen_output[0].tolist() if token >= 0
    )


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_invalid_lift_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        _decision() if threshold == 1e-6 else decide_context_causal_k2(
            candidate_token_ids=(0, 1),
            full_logits=[torch.zeros(3), torch.zeros(3)],
            base_logits=[torch.zeros(3), torch.zeros(3)],
            target_logits=[torch.zeros(3), torch.zeros(3), torch.zeros(3)],
            bonus_full_logits=torch.zeros(3),
            bonus_base_logits=torch.zeros(3),
            lift_min=threshold,
        )


class _PolicyRuntime:
    def __init__(self, *, candidates, a0, b0, a1, b1, bonus=None):
        self.signal = type(
            "Signal",
            (),
            {
                "candidate_token_ids": candidates,
                "a0": a0,
                "b0": b0,
                "a1": a1,
                "b1": b1,
            },
        )()
        self.bonus = bonus
        self.bonus_calls = 0

    def prepare_full_accept_bonus(self, target_bonus_logits):
        self.bonus_calls += 1
        assert self.bonus is not None
        return (*self.bonus, False)


def _policy_outcome(*, runtime, target):
    request_id = "policy"
    return build_asymspec_context_causal_outcome(
        scheduler_output=type(
            "SchedulerOutput",
            (),
            {"scheduled_spec_decode_tokens": {request_id: [0, 1]}},
        )(),
        spec_decode_metadata=type(
            "Metadata",
            (),
            {
                "num_draft_tokens": [2],
                "max_spec_len": 2,
                "target_logits_indices": torch.tensor([0, 1]),
                "bonus_logits_indices": torch.tensor([2]),
            },
        )(),
        logits=torch.stack(target),
        requests={
            request_id: type(
                "Request",
                (),
                {
                    "sampling_params": type(
                        "Params",
                        (),
                        {"extra_args": {DIAGNOSTIC_LIVE_CONTEXT_CAUSAL_POLICY: True}},
                    )()
                },
            )()
        },
        active_live={request_id: runtime},
        is_asymspec=True,
    )


def test_live_policy_rejection_never_prepares_bonus_state():
    target = [
        torch.tensor([1.0, 1.5, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([5.0, 0.0, 0.0]),
    ]
    runtime = _PolicyRuntime(
        candidates=(0, 1),
        a0=torch.tensor([1.5, 1.0, 0.0]),
        b0=torch.tensor([1.5, 1.0, 0.0]),
        a1=target[1],
        b1=target[1],
    )
    outcome = _policy_outcome(runtime=runtime, target=target)
    assert outcome is not None
    assert outcome.accepted_count == 0
    assert outcome.output_token_ids.tolist() == [[1, -1, -1]]
    assert runtime.bonus_calls == 0


def test_live_policy_full_accept_prepares_fresh_bonus_once():
    target = [
        torch.tensor([5.0, 0.0, 0.0]),
        torch.tensor([0.0, 5.0, 0.0]),
        torch.tensor([5.0, 4.9, 0.0]),
    ]
    runtime = _PolicyRuntime(
        candidates=(0, 1),
        a0=target[0],
        b0=target[0],
        a1=target[1],
        b1=target[1],
        bonus=(torch.tensor([0.0, 20.0, 0.0]), torch.tensor([0.0, -20.0, 0.0])),
    )
    outcome = _policy_outcome(runtime=runtime, target=target)
    assert outcome is not None
    assert outcome.accepted_count == 2
    assert outcome.next_seed_token_id == 1
    assert outcome.output_token_ids.tolist() == [[0, 1, 1]]
    assert runtime.bonus_calls == 1
