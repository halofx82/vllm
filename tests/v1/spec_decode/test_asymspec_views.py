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
