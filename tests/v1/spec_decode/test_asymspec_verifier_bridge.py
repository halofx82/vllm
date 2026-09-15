# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.spec_decode.asymspec.target_diagnostic import (
    capture_asymspec_verifier_rows,
)
from vllm.v1.spec_decode.asymspec.verifier_bridge import (
    DIAGNOSTIC_CANDIDATE_TOKEN_IDS,
    DIAGNOSTIC_VERIFIER_OUTPUT_PATH,
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
