# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit coverage for the pre-cache AsymSpec shared-draft foundation."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.v1.spec_decode.asymspec.views as views_module
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.spec_decode.asymspec.views import (
    AsymSpecDraftViews,
    AsymSpecViewRole,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class _ParallelConfig:
    rank: int = 0


@dataclass
class _VllmConfig:
    speculative_config: object
    parallel_config: _ParallelConfig
    model_config: object
    quant_config: object | None = None


def _make_config(method: str = "asymspec") -> _VllmConfig:
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(model="test-draft"),
        draft_parallel_config=_ParallelConfig(),
        draft_load_config=None,
        num_speculative_tokens=2,
    )
    return _VllmConfig(speculative_config, _ParallelConfig(), object())


def test_full_and_base_are_distinct_views_of_one_physical_model():
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    physical_model = nn.Linear(4, 4, bias=False)

    views.bind_model(physical_model)

    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    assert full is not base
    assert full.role is AsymSpecViewRole.FULL
    assert base.role is AsymSpecViewRole.BASE
    assert full.state is not base.state
    assert full.model is base.model is physical_model
    assert full.model.weight.data_ptr() == base.model.weight.data_ptr()


def test_loads_one_physical_model_then_binds_both_views(monkeypatch):
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    physical_model = nn.Linear(4, 4, bias=False)
    loads: list[object] = []

    def load_once(**kwargs):
        loads.append(kwargs)
        return physical_model

    monkeypatch.setattr(views_module, "get_model", load_once)
    monkeypatch.setattr(views_module, "set_model_tag", lambda _: nullcontext())

    views.load_model()

    assert len(loads) == 1
    assert views.full is not None and views.base is not None
    assert views.full.model is views.base.model is physical_model
    with pytest.raises(RuntimeError, match="already been loaded"):
        views.load_model()


def test_rejects_non_asymspec_or_missing_draft_configuration():
    with pytest.raises(ValueError, match="method='asymspec'"):
        AsymSpecDraftViews(_make_config(method="draft_model"), torch.device("cpu"))

    config = _make_config()
    config.speculative_config.draft_model_config = None
    with pytest.raises(ValueError, match="draft model config"):
        AsymSpecDraftViews(config, torch.device("cpu"))


def test_model_runner_selects_native_asymspec_view_holder():
    runner = SimpleNamespace(vllm_config=_make_config(), device=torch.device("cpu"))

    GPUModelRunner._initialize_asymspec_draft_views(runner)

    assert isinstance(runner.asymspec_draft_views, AsymSpecDraftViews)


class _AttentionLayer(nn.Module, AttentionLayerBase):
    def get_attn_backend(self):
        return object

    def get_kv_cache_spec(self, vllm_config):
        del vllm_config
        return None


class _GDNLayer(nn.Module, MambaBase):
    def get_state_shape(self):
        return ((4, 8), (2, 3, 5))

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.GDN_ATTN


class _HybridQwenLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_index in range(2):
            layer = nn.Module()
            if layer_index == 0:
                layer.self_attn = _AttentionLayer()
            else:
                layer.linear_attn = _GDNLayer()
            self.layers.append(layer)


def test_hybrid_specs_are_role_owned_and_allocation_free():
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    model = _HybridQwenLikeModel()
    views.bind_model(model)

    views.initialize_state_specs()

    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    full_spec = full.state.hybrid_spec
    base_spec = base.state.hybrid_spec
    assert full_spec is not None and base_spec is not None
    assert full_spec is not base_spec
    assert full_spec.recurrent is not base_spec.recurrent
    assert full_spec.attention[0].layer_name == "layers.0.self_attn"
    assert full_spec.recurrent[0].layer_name == "layers.1.linear_attn"
    assert full_spec.recurrent[0].shapes == ((4, 8), (2, 3, 5))
    assert full_spec.recurrent[0].dtypes == (torch.bfloat16, torch.float32)
    assert full_spec.compact_recurrent_state_slots == 3
    assert full.model is base.model is model
    assert full.state.kv_state is base.state.kv_state is None
    assert full.state.recurrent_state is base.state.recurrent_state is None
    assert full.state.position_state is base.state.position_state is None
    assert not hasattr(full_spec, "cache_group_id")


def test_non_hybrid_draft_is_rejected_before_any_state_allocation():
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = _AttentionLayer()
    views.bind_model(model)

    with pytest.raises(ValueError, match="hybrid Qwen3.5"):
        views.initialize_state_specs()

    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    assert full.state.hybrid_spec is None
    assert base.state.hybrid_spec is None
