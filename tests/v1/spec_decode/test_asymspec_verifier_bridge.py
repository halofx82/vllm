# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.asymspec.live_iteration import _live_prompt_ids
from vllm.v1.spec_decode.asymspec.target_diagnostic import (
    build_asymspec_fixed_acceptance_outcome,
    capture_asymspec_verifier_rows,
    force_asymspec_diagnostic_decode_outputs,
)
from vllm.v1.spec_decode.asymspec.verifier_bridge import (
    DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT,
    DIAGNOSTIC_CANDIDATE_TOKEN_IDS,
    DIAGNOSTIC_FIXED_ACCEPTED_COUNT,
    DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS,
    DIAGNOSTIC_LIVE_BASE_LAG_TOKENS,
    DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS,
    DIAGNOSTIC_LIVE_OUTPUT_PATH,
    DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS,
    DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS,
    DIAGNOSTIC_VERIFIER_OUTPUT_PATH,
    arm_asymspec_diagnostic_control_spec_tokens,
    arm_asymspec_live_spec_tokens,
    register_asymspec_diagnostic_spec_tokens,
)


def _request(*, extra_args=None, spec_token_ids=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=extra_args),
        spec_token_ids=[] if spec_token_ids is None else spec_token_ids,
    )


def test_asymspec_bridge_injects_exactly_two_tokens():
    request = _request(
        extra_args={
            DIAGNOSTIC_CANDIDATE_TOKEN_IDS: [13, 17],
            DIAGNOSTIC_VERIFIER_OUTPUT_PATH: "/tmp/rows.pt",
        }
    )
    config = SimpleNamespace(method="asymspec")

    assert register_asymspec_diagnostic_spec_tokens(request, config)
    assert request.spec_token_ids == []


def test_non_asymspec_bridge_is_a_noop():
    request = _request(
        extra_args={
            DIAGNOSTIC_CANDIDATE_TOKEN_IDS: [13, 17],
            DIAGNOSTIC_VERIFIER_OUTPUT_PATH: "/tmp/rows.pt",
        }
    )

    assert not register_asymspec_diagnostic_spec_tokens(
        request, SimpleNamespace(method="draft_model")
    )
    assert request.spec_token_ids == []


def test_live_bridge_arms_only_at_the_uncomputed_seed_boundary():
    request = _request(
        extra_args={
            DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1, 2],
            DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1, 2],
            DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
        }
    )
    request._output_token_ids = [31]
    request.num_output_tokens = 1
    request.num_computed_tokens = 2
    request.num_prompt_tokens = 2

    assert arm_asymspec_live_spec_tokens(
        request, SimpleNamespace(method="asymspec"), (13, 17)
    )
    assert request.spec_token_ids == [13, 17]
    assert request._asymspec_live_seed_token_id == 31


def test_live_bridge_needs_no_diagnostic_request_fields():
    request = _request()
    request._output_token_ids = [31]
    request.num_output_tokens = 1
    request.num_computed_tokens = 2
    request.num_prompt_tokens = 2

    assert arm_asymspec_live_spec_tokens(
        request, SimpleNamespace(method="asymspec"), (13, 17)
    )
    assert request.spec_token_ids == [13, 17]


def test_live_bridge_rejects_a_computed_or_missing_seed():
    request = _request(
        extra_args={
            DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1],
            DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1],
            DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
        }
    )
    request._output_token_ids = [31]
    request.num_output_tokens = 1
    request.num_prompt_tokens = 1
    request.num_computed_tokens = 2
    with pytest.raises(RuntimeError, match="uncomputed"):
        arm_asymspec_live_spec_tokens(
            request, SimpleNamespace(method="asymspec"), (13, 17)
        )


def test_live_bridge_arms_again_after_fixed_outcome_with_one_seed_left():
    request = _request(
        extra_args={
            DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1, 2],
            DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1, 2],
            DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
        }
    )
    # Prompt + accepted suffix + R; only R is intentionally uncomputed.
    request._output_token_ids = [31, 13, 17, 41]
    request.num_output_tokens = 4
    request.num_prompt_tokens = 2
    request.num_computed_tokens = 5

    assert arm_asymspec_live_spec_tokens(
        request, SimpleNamespace(method="asymspec"), (43, 47)
    )
    assert request.spec_token_ids == [43, 47]


def test_target_control_arms_only_after_ordinary_seed_and_decode():
    request = _request(
        extra_args={
            DIAGNOSTIC_ARM_AFTER_OUTPUT_COUNT: 2,
            DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS: [13, 17],
        }
    )
    request.num_output_tokens = 1
    config = SimpleNamespace(method="asymspec")
    assert not arm_asymspec_diagnostic_control_spec_tokens(request, config)
    request.num_output_tokens = 2
    assert arm_asymspec_diagnostic_control_spec_tokens(request, config)
    assert request.spec_token_ids == [13, 17]


def test_forced_decode_replaces_only_post_seed_ordinary_output():
    state = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS: [13, 17]}
        ),
        output_token_ids=[31],
    )
    sampled = torch.tensor([[41]], dtype=torch.int32)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"control": 1},
        scheduled_spec_decode_tokens={},
    )
    assert force_asymspec_diagnostic_decode_outputs(
        sampled_token_ids=sampled,
        scheduler_output=scheduler_output,
        requests={"control": state},
        req_id_to_index={"control": 0},
        is_asymspec=True,
    )
    assert sampled.tolist() == [[13]]
    assert state._asymspec_forced_decode_index == 1


def test_forced_decode_does_not_touch_seed_or_speculative_step():
    state = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS: [13]}
        ),
        output_token_ids=[],
    )
    sampled = torch.tensor([[41]], dtype=torch.int32)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"control": 1},
        scheduled_spec_decode_tokens={},
    )
    assert not force_asymspec_diagnostic_decode_outputs(
        sampled_token_ids=sampled,
        scheduler_output=scheduler_output,
        requests={"control": state},
        req_id_to_index={"control": 0},
        is_asymspec=True,
    )
    state.output_token_ids = [31]
    scheduler_output.scheduled_spec_decode_tokens = {"control": [13, 17]}
    assert not force_asymspec_diagnostic_decode_outputs(
        sampled_token_ids=sampled,
        scheduler_output=scheduler_output,
        requests={"control": state},
        req_id_to_index={"control": 0},
        is_asymspec=True,
    )
    assert sampled.tolist() == [[41]]


def test_forced_decode_none_leaves_greedy_output_but_consumes_control_slot():
    state = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={DIAGNOSTIC_FORCED_DECODE_TOKEN_IDS: [None, 17]}
        ),
        output_token_ids=[31],
    )
    sampled = torch.tensor([[41]], dtype=torch.int32)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"control": 1},
        scheduled_spec_decode_tokens={},
    )
    assert not force_asymspec_diagnostic_decode_outputs(
        sampled_token_ids=sampled,
        scheduler_output=scheduler_output,
        requests={"control": state},
        req_id_to_index={"control": 0},
        is_asymspec=True,
    )
    assert sampled.tolist() == [[41]]
    assert state._asymspec_forced_decode_index == 1


def test_verifier_capture_records_control_state_without_mutation(tmp_path):
    request_id = "control"
    state = SimpleNamespace(
        output_token_ids=[31, 13, 41],
        num_computed_tokens=9,
        sampling_params=SimpleNamespace(
            extra_args={
                DIAGNOSTIC_CANDIDATE_TOKEN_IDS: [43, 47],
                DIAGNOSTIC_VERIFIER_OUTPUT_PATH: str(tmp_path / "rows.pt"),
            }
        ),
    )
    metadata = SimpleNamespace(
        num_draft_tokens=[2],
        target_logits_indices=torch.tensor([0, 1]),
        bonus_logits_indices=torch.tensor([2]),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={request_id: [43, 47]}
    )
    assert capture_asymspec_verifier_rows(
        scheduler_output=scheduler_output,
        spec_decode_metadata=metadata,
        logits=torch.randn(3, 64),
        requests={request_id: state},
        is_asymspec=True,
    )
    capture = torch.load(tmp_path / "rows.pt", weights_only=False)
    assert capture["output_token_ids"] == [31, 13, 41]
    assert capture["num_computed_tokens"] == 9
    assert state.output_token_ids == [31, 13, 41]


def test_live_preseed_commits_are_the_only_deferred_base_range():
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1, 2],
                DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1, 2],
                DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
                DIAGNOSTIC_LIVE_PRESEED_COMMITTED_TOKEN_IDS: [3, 4, 5],
                DIAGNOSTIC_LIVE_BASE_LAG_TOKENS: 2,
            }
        )
    )
    assert _live_prompt_ids(request) == (
        [1, 2],
        [1, 2],
        [3, 4, 5],
        "/tmp/live.pt",
        2,
    )


def test_live_prompt_defaults_to_normal_request_tokens_without_diagnostics():
    request = SimpleNamespace(
        prompt_token_ids=[11, 13, 17],
        sampling_params=SimpleNamespace(extra_args=None),
    )

    assert _live_prompt_ids(request) == ([11, 13, 17], [11, 13, 17], [], "", 0)


def test_live_prompt_uses_server_owned_full_view_only_for_full_driver():
    request = SimpleNamespace(
        prompt_token_ids=[11, 13, 17],
        sampling_params=SimpleNamespace(
            extra_args={"specsteer_aug_prompt_ids": [2, 3, 5, 7, 11, 13, 17]}
        ),
    )

    assert _live_prompt_ids(request) == (
        [2, 3, 5, 7, 11, 13, 17],
        [11, 13, 17],
        [],
        "",
        0,
    )


def test_server_owned_full_view_cannot_be_shorter_than_compressed_view():
    request = SimpleNamespace(
        prompt_token_ids=[11, 13, 17],
        sampling_params=SimpleNamespace(extra_args={"specsteer_aug_prompt_ids": [11]}),
    )

    with pytest.raises(ValueError, match="cannot be shorter"):
        _live_prompt_ids(request)


def test_scheduler_transports_asymspec_pair_through_normal_v1_path():
    scheduler = create_scheduler(num_speculative_tokens=2)
    assert scheduler.vllm_config.speculative_config is not None
    scheduler.vllm_config.speculative_config.method = "asymspec"
    request = create_requests(num_requests=1, num_tokens=8)[0]
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        DIAGNOSTIC_CANDIDATE_TOKEN_IDS: [13, 17],
        DIAGNOSTIC_VERIFIER_OUTPUT_PATH: "/tmp/rows.pt",
    }

    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.request_id not in output.scheduled_spec_decode_tokens
    # Simulate completion of the normal first prompt forward. The next V1
    # schedule activates the pending pair and consumes it normally.
    request.num_computed_tokens = request.num_prompt_tokens
    output = scheduler.schedule()
    assert output.scheduled_spec_decode_tokens[request.request_id] == [13, 17]
    # The generic scheduler consumed the request-side transport field.
    assert request.spec_token_ids == []


def test_scheduler_finishes_live_capture_without_sampling():
    scheduler = create_scheduler(num_speculative_tokens=2)
    assert scheduler.vllm_config.speculative_config is not None
    scheduler.vllm_config.speculative_config.method = "asymspec"
    request = create_requests(num_requests=1, num_tokens=8)[0]
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1, 2],
        DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1, 2],
        DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
    }
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    req_id = request.request_id
    assert req_id in scheduler.requests

    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[]],
            asymspec_live_capture_complete={req_id},
        ),
    )

    assert req_id not in scheduler.requests
    assert req_id in scheduler.finished_req_ids


def test_live_capture_marker_does_not_finish_non_asymspec_request():
    scheduler = create_scheduler(num_speculative_tokens=2)
    request = create_requests(num_requests=1, num_tokens=8)[0]
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1, 2],
        DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1, 2],
        DIAGNOSTIC_LIVE_OUTPUT_PATH: "/tmp/live.pt",
    }
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    req_id = request.request_id
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[]],
            asymspec_live_capture_complete={req_id},
        ),
    )

    assert req_id in scheduler.requests


@pytest.mark.parametrize(
    ("accepted_count", "expected_tokens"),
    [(0, [31]), (1, [13, 41]), (2, [13, 17, 59])],
)
def test_fixed_acceptance_uses_the_matching_native_verifier_row(
    accepted_count, expected_tokens
):
    request_id = "fixed"
    requests = {
        request_id: SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={DIAGNOSTIC_FIXED_ACCEPTED_COUNT: accepted_count}
            )
        )
    }
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={request_id: [13, 17]}
    )
    metadata = SimpleNamespace(
        num_draft_tokens=[2],
        max_spec_len=2,
        target_logits_indices=torch.tensor([0, 1]),
        bonus_logits_indices=torch.tensor([2]),
    )
    logits = torch.zeros(3, 64)
    logits[0, 31] = 1
    logits[1, 41] = 1
    logits[2, 59] = 1

    outcome = build_asymspec_fixed_acceptance_outcome(
        scheduler_output=scheduler_output,
        spec_decode_metadata=metadata,
        logits=logits,
        requests=requests,
        is_asymspec=True,
    )

    assert outcome is not None
    assert outcome.accepted_count == accepted_count
    assert outcome.next_seed_token_id == expected_tokens[-1]
    assert outcome.output_token_ids.tolist() == [
        expected_tokens + [-1] * (3 - len(expected_tokens))
    ]


@pytest.mark.parametrize("accepted_count", [0, 1, 2])
def test_scheduler_applies_fixed_native_spec_outcome(accepted_count):
    scheduler = create_scheduler(num_speculative_tokens=2)
    assert scheduler.vllm_config.speculative_config is not None
    scheduler.vllm_config.speculative_config.method = "asymspec"
    request = create_requests(num_requests=1, num_tokens=8)[0]
    scheduler.add_request(request)
    scheduler.schedule()
    # Model the native seed boundary: S is canonical but remains uncomputed.
    seed = 29
    request.append_output_token_ids(seed)
    request.num_computed_tokens = request.num_prompt_tokens
    request.spec_token_ids = [13, 17]
    scheduled = scheduler.schedule()
    req_id = request.request_id
    assert scheduled.scheduled_spec_decode_tokens[req_id] == [13, 17]
    assert request.num_computed_tokens == request.num_prompt_tokens + 3

    next_seed = 41 + accepted_count
    generated = [13, 17][:accepted_count] + [next_seed]
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[generated],
        ),
    )

    assert list(request.output_token_ids) == [seed, *generated]
    assert request.num_computed_tokens == request.num_prompt_tokens + 1 + accepted_count
    assert request.num_tokens == request.num_computed_tokens + 1


def test_fixed_outcome_can_arm_one_follow_up_seed_verification():
    scheduler = create_scheduler(num_speculative_tokens=2)
    assert scheduler.vllm_config.speculative_config is not None
    scheduler.vllm_config.speculative_config.method = "asymspec"
    request = create_requests(num_requests=1, num_tokens=8)[0]
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        DIAGNOSTIC_NEXT_SPEC_TOKEN_IDS: [43, 47],
    }
    scheduler.add_request(request)
    scheduler.schedule()
    request.append_output_token_ids(29)
    request.num_computed_tokens = request.num_prompt_tokens
    request.spec_token_ids = [13, 17]
    scheduled = scheduler.schedule()
    req_id = request.request_id
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[13, 41]],
            asymspec_fixed_acceptance_counts={req_id: 1},
        ),
    )

    assert list(request.output_token_ids) == [29, 13, 41]
    assert request.num_computed_tokens == request.num_tokens - 1
    assert request.spec_token_ids == [43, 47]


@pytest.mark.parametrize("tokens", [[13], [13, 17, 19], [13, -1]])
def test_asymspec_bridge_rejects_invalid_tokens(tokens):
    request = _request(
        extra_args={
            DIAGNOSTIC_CANDIDATE_TOKEN_IDS: tokens,
            DIAGNOSTIC_VERIFIER_OUTPUT_PATH: "/tmp/rows.pt",
        }
    )
    with pytest.raises(ValueError):
        register_asymspec_diagnostic_spec_tokens(
            request, SimpleNamespace(method="asymspec")
        )


def test_verifier_capture_uses_v1_metadata_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    output_path = tmp_path / "rows.pt"
    request_id = "diagnostic"
    requests = {
        request_id: SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={
                    DIAGNOSTIC_CANDIDATE_TOKEN_IDS: [13, 17],
                    DIAGNOSTIC_VERIFIER_OUTPUT_PATH: str(output_path),
                }
            )
        )
    }
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={request_id: [13, 17]}
    )
    metadata = SimpleNamespace(
        num_draft_tokens=[2],
        target_logits_indices=torch.tensor([1, 3]),
        bonus_logits_indices=torch.tensor([4]),
    )
    logits = torch.arange(30, dtype=torch.float32).reshape(5, 6)

    assert capture_asymspec_verifier_rows(
        scheduler_output=scheduler_output,
        spec_decode_metadata=metadata,
        logits=logits,
        requests=requests,
        is_asymspec=True,
    )
    saved = torch.load(output_path, weights_only=False)
    assert saved["candidate_token_ids"] == [13, 17]
    assert saved["target_logits_indices"] == [1, 3]
    assert saved["bonus_logits_indices"] == [4]
    assert torch.equal(saved["t0"], logits[1].to(torch.bfloat16))
    assert torch.equal(saved["t1"], logits[3].to(torch.bfloat16))
    assert torch.equal(saved["t2"], logits[4].to(torch.bfloat16))


def test_verifier_capture_is_inert_for_non_asymspec():
    assert not capture_asymspec_verifier_rows(
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
        spec_decode_metadata=None,
        logits=None,
        requests={},
        is_asymspec=False,
    )


def test_live_verifier_capture_merges_draft_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    output_path = tmp_path / "live.pt"
    torch.save({"candidate_token_ids": (13, 17), "a0": torch.ones(6)}, output_path)
    request_id = "live"
    requests = {
        request_id: SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={
                    DIAGNOSTIC_LIVE_FULL_PROMPT_TOKEN_IDS: [1],
                    DIAGNOSTIC_LIVE_BASE_PROMPT_TOKEN_IDS: [1],
                    DIAGNOSTIC_LIVE_OUTPUT_PATH: str(output_path),
                }
            )
        )
    }
    assert capture_asymspec_verifier_rows(
        scheduler_output=SimpleNamespace(
            scheduled_spec_decode_tokens={request_id: [13, 17]}
        ),
        spec_decode_metadata=SimpleNamespace(
            num_draft_tokens=[2],
            target_logits_indices=torch.tensor([0, 1]),
            bonus_logits_indices=torch.tensor([2]),
        ),
        logits=torch.arange(18, dtype=torch.float32).reshape(3, 6),
        requests=requests,
        is_asymspec=True,
    )
    saved = torch.load(output_path, weights_only=False)
    assert torch.equal(saved["a0"], torch.ones(6))
    assert saved["candidate_token_ids"] == [13, 17]
    assert saved["t0"].shape == (6,)
