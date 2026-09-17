# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for AsymSpec's post-verifier draft synchronization."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.asymspec.live_iteration as live_iteration
from vllm.v1.spec_decode.asymspec.live_iteration import AsymSpecLiveIterationRuntime


class _Full:
    def __init__(self):
        self.canonical_len = 11
        self.committed = []

    def commit_token(self, token):
        self.committed.append(token)
        self.canonical_len += 1


class _Base:
    def __init__(self, state):
        self.canonical_len = 11
        self.state = state
        self.observed = []

    def observe_committed_token(self, token):
        self.observed.append(token)
        self.state.base.pending_token_ids.append(token)

    def catch_up_base(self):
        self.canonical_len += len(self.state.base.pending_token_ids)
        self.state.base.canonical_len = self.canonical_len
        self.state.base.observed_len = self.canonical_len
        self.state.base.pending_token_ids.clear()


class _Transaction:
    def __init__(self, full, tokens):
        self.full = full
        self.candidate_token_ids = tokens
        self.active = True
        self.promoted = None

    def promote(self, count):
        self.promoted = count
        self.full.canonical_len += count
        self.active = False

    def rollback(self):
        self.promote(0)


class _Proposal:
    def __init__(self, full, tokens):
        self.candidate_token_ids = tokens
        self.transaction = _Transaction(full, tokens)
        self.canonical_result = SimpleNamespace(logits=torch.zeros(1, 4))
        self.candidate_results = (
            SimpleNamespace(logits=torch.zeros(1, 4)),
            SimpleNamespace(logits=torch.zeros(1, 4)),
        )


@pytest.mark.parametrize("accepted_count", [0, 1, 2])
def test_live_outcome_promotes_full_and_observes_base(monkeypatch, accepted_count):
    state = SimpleNamespace(
        base=SimpleNamespace(canonical_len=11, observed_len=11, pending_token_ids=[]),
        full=SimpleNamespace(augmentation_offset=0),
    )
    full = _Full()
    base = _Base(state)
    initial = _Proposal(full, (101, 103))
    runtime = AsymSpecLiveIterationRuntime(
        request_id="r",
        seed_token_id=7,
        request_state=state,
        physical=None,
        logical=None,
        bindings=None,
        full=full,
        base=base,
        proposal=initial,
        base_score=None,
        signal=None,
        output_path="/tmp/unused.pt",
        authoritative_suffix_token_ids=[7],
    )

    next_proposal = _Proposal(full, (107, 109))
    monkeypatch.setattr(
        live_iteration,
        "AsymSpecFullK2Proposer",
        lambda driver: SimpleNamespace(propose_k2=lambda: next_proposal),
    )
    monkeypatch.setattr(
        live_iteration,
        "AsymSpecBasePairScorer",
        lambda driver: SimpleNamespace(
            score_pair=lambda tokens: SimpleNamespace(
                logits_before_a=SimpleNamespace(logits=torch.zeros(1, 4)),
                logits_before_b=SimpleNamespace(logits=torch.zeros(1, 4)),
            )
        ),
    )
    monkeypatch.setattr(
        live_iteration,
        "build_asymspec_draft_signal",
        lambda **_: SimpleNamespace(),
    )

    assert runtime.apply_target_outcome(
        accepted_count=accepted_count, next_seed_token_id=113
    ) == (
        107,
        109,
    )
    accepted = [101, 103][:accepted_count]
    assert initial.transaction.promoted == accepted_count
    assert full.committed == [113]
    assert base.observed == [*accepted, 113]
    assert state.base.pending_token_ids == []
    assert runtime.authoritative_suffix_token_ids == [7, *accepted, 113]
    assert runtime.last_accepted_token_ids == tuple(accepted)
    assert full.canonical_len == base.canonical_len == 12 + accepted_count
