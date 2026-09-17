# SPDX-License-Identifier: Apache-2.0
"""Frozen Step-51 external evidence-record parity tests."""

from vllm.v1.spec_decode.asymspec.evidence_transfer import (
    MIN_CF,
    MIN_MARGIN,
    evidence_wrapper,
    select_evidence,
)


def test_select_evidence_uses_frozen_strict_cf_and_margin_gate():
    record = select_evidence(
        [
            {"seed": 11, "token_ids": [7, 8], "context_llr": 0.24},
            {"seed": 12, "token_ids": [9, 10], "context_llr": 0.12},
        ],
        full_top2=[11, 12],
        full_source_ids=[1, 7, 8, 2],
        base_source_ids=[1, 9, 10, 2],
    )
    assert record.selected_seed == 11
    assert record.attribution_pass
    assert record.cf_selected == 0.24
    assert record.d_ctx == 0.12
    assert record.full_provenance == {"found": True, "start": 1, "length": 2}
    assert not record.base_provenance["found"]


def test_exact_frozen_thresholds_fail_closed():
    record = select_evidence(
        [
            {"seed": 11, "token_ids": [7], "context_llr": MIN_CF},
            {"seed": 12, "token_ids": [8], "context_llr": MIN_CF - MIN_MARGIN},
        ],
        full_top2=[11, 12],
        full_source_ids=[7],
        base_source_ids=[],
    )
    assert not record.attribution_pass


def test_wrapper_is_the_immutable_frozen_prior_user_envelope():
    assert evidence_wrapper("field request_key") == (
        "<asymspec_long_context_evidence>\n"
        "The following text was retrieved from a longer context. It may be "
        "fragmentary or source-like. Use it as contextual evidence, then follow "
        "the user's original instruction and requested output format exactly.\n"
        "field request_key\n"
        "</asymspec_long_context_evidence>"
    )
