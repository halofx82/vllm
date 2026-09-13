# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-only coverage for the native AsymSpec registration."""

from types import SimpleNamespace

import pytest

import vllm.config.speculative as speculative_module
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


class _FakeModelConfig:
    def __init__(self, max_model_len: int = 32768):
        self.model = "test-model"
        self.tokenizer = "test-tokenizer"
        self.tokenizer_mode = "auto"
        self.trust_remote_code = False
        self.allowed_local_media_path = ""
        self.allowed_media_domains = None
        self.dtype = "bfloat16"
        self.seed = 0
        self.revision = None
        self.code_revision = None
        self.tokenizer_revision = None
        self.max_model_len = max_model_len
        self.quantization = None
        self.enforce_eager = True
        self.max_logprobs = 20
        self.hf_overrides = None
        self.config_format = "auto"
        self.hf_config = SimpleNamespace(model_type="llama")
        self.architectures = ["LlamaForCausalLM"]

    def get_vocab_size(self) -> int:
        return 32000

    def verify_with_parallel_config(self, parallel_config) -> None:
        del parallel_config


@pytest.fixture
def asymspec_config_dependencies(monkeypatch):
    target = _FakeModelConfig()
    draft = _FakeModelConfig()
    monkeypatch.setattr(speculative_module, "ModelConfig", lambda **_: draft)
    return {
        "model": "test-draft",
        "target_model_config": target,
        "target_parallel_config": ParallelConfig(),
    }


def test_non_asymspec_draft_model_config_remains_valid(
    asymspec_config_dependencies,
):
    config = SpeculativeConfig(
        method="draft_model",
        num_speculative_tokens=3,
        **asymspec_config_dependencies,
    )

    assert config.method == "draft_model"
    assert config.uses_draft_model()


def test_asymspec_minimal_config_is_valid(asymspec_config_dependencies):
    config = SpeculativeConfig(
        method="asymspec",
        num_speculative_tokens=2,
        asymspec_compressed_max_model_len=8192,
        **asymspec_config_dependencies,
    )

    assert config.method == "asymspec"
    assert config.uses_draft_model()
    assert config.asymspec_compressed_max_model_len == 8192
    assert config.asymspec_full_prefill_chunk_tokens == 8192
    assert config.asymspec_evidence_mode == "off"


def test_asymspec_rejects_k_other_than_two(asymspec_config_dependencies):
    with pytest.raises(ValueError, match="num_speculative_tokens=2"):
        SpeculativeConfig(
            method="asymspec",
            num_speculative_tokens=3,
            asymspec_compressed_max_model_len=8192,
            **asymspec_config_dependencies,
        )


def test_asymspec_rejects_invalid_evidence_mode(asymspec_config_dependencies):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="asymspec",
            num_speculative_tokens=2,
            asymspec_compressed_max_model_len=8192,
            asymspec_evidence_mode="experimental",
            **asymspec_config_dependencies,
        )


@pytest.mark.parametrize("compressed_max_model_len", [0, 32769])
def test_asymspec_rejects_invalid_compressed_max_model_len(
    asymspec_config_dependencies, compressed_max_model_len
):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="asymspec",
            num_speculative_tokens=2,
            asymspec_compressed_max_model_len=compressed_max_model_len,
            **asymspec_config_dependencies,
        )


def test_asymspec_rejects_invalid_full_prefill_chunk_size(
    asymspec_config_dependencies,
):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="asymspec",
            num_speculative_tokens=2,
            asymspec_compressed_max_model_len=8192,
            asymspec_full_prefill_chunk_tokens=0,
            **asymspec_config_dependencies,
        )
