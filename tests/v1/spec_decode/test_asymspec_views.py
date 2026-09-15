# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit coverage for the pre-cache AsymSpec shared-draft foundation."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.v1.spec_decode.asymspec.canonical_driver as canonical_driver_module
import vllm.v1.spec_decode.asymspec.execution_metadata as execution_metadata_module
import vllm.v1.spec_decode.asymspec.views as views_module
import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.spec_decode.asymspec.base_scorer import AsymSpecBasePairScorer
from vllm.v1.spec_decode.asymspec.cache_binding import (
    bind_asymspec_draft_caches,
)
from vllm.v1.spec_decode.asymspec.cache_plan import (
    AsymSpecCacheDomain,
    build_asymspec_domain_allocation_plans,
    characterize_asymspec_allocator_compatibility,
    compose_asymspec_global_cache_plan,
)
from vllm.v1.spec_decode.asymspec.candidate_transaction import (
    AsymSpecFullCandidateTransaction,
)
from vllm.v1.spec_decode.asymspec.canonical_driver import (
    AsymSpecCanonicalDraftDriver,
)
from vllm.v1.spec_decode.asymspec.draft_forward import (
    execute_asymspec_draft_forward,
    qwen3_5_text_positions,
)
from vllm.v1.spec_decode.asymspec.execution_metadata import (
    build_asymspec_view_execution_metadata,
    recurrent_page_ids,
)
from vllm.v1.spec_decode.asymspec.logical_cache import (
    AsymSpecLogicalCacheGroup,
    allocate_synthetic_attention_blocks,
    build_asymspec_logical_cache_plan,
    instantiate_asymspec_logical_block_pools,
)
from vllm.v1.spec_decode.asymspec.physical_cache import (
    allocate_asymspec_physical_cache_tensors,
    build_asymspec_physical_cache_plan,
)
from vllm.v1.spec_decode.asymspec.proposer import AsymSpecFullK2Proposer
from vllm.v1.spec_decode.asymspec.request_state import (
    create_asymspec_request_state,
)
from vllm.v1.spec_decode.asymspec.views import (
    AsymSpecDraftViews,
    AsymSpecViewRole,
    share_model_state,
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
    cache_config: object | None = None
    scheduler_config: object | None = None
    kv_transfer_config: object | None = None


def _make_config(method: str = "asymspec") -> _VllmConfig:
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(model="test-draft"),
        draft_parallel_config=_ParallelConfig(),
        draft_load_config=None,
        num_speculative_tokens=2,
        asymspec_compressed_max_model_len=128,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        mamba_block_size=256,
        mamba_page_size_padded=None,
        mamba_cache_mode="none",
        num_gpu_blocks_override=None,
    )
    scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    return _VllmConfig(
        speculative_config,
        _ParallelConfig(),
        SimpleNamespace(max_model_len=1024),
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )


def test_full_and_base_are_distinct_trees_with_shared_model_storage():
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    full_model = nn.Linear(4, 4, bias=False)
    base_model = nn.Linear(4, 4, bias=False)

    views.bind_models(full_model, base_model)

    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    assert full is not base
    assert full.role is AsymSpecViewRole.FULL
    assert base.role is AsymSpecViewRole.BASE
    assert full.state is not base.state
    assert full.model is full_model
    assert base.model is base_model
    assert full.model is not base.model
    assert full.model.weight.data_ptr() == base.model.weight.data_ptr()


def test_shared_state_leaves_runtime_attributes_tree_local():
    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.register_buffer("persistent", torch.ones(2))
            self.kv_cache = object()
            self.runtime_state = {"position": 0}

    full = _Tree()
    base = _Tree()
    parameter_bytes, buffer_bytes = share_model_state(full, base)

    assert parameter_bytes == full.proj.weight.numel() * full.proj.weight.element_size()
    assert buffer_bytes == full.persistent.numel() * full.persistent.element_size()
    assert full is not base
    assert full.proj is not base.proj
    assert full.proj.weight is base.proj.weight
    assert full.proj.weight.data_ptr() == base.proj.weight.data_ptr()
    assert full.persistent is base.persistent
    assert full.kv_cache is not base.kv_cache
    assert full.runtime_state is not base.runtime_state


def test_loads_one_checkpoint_model_then_builds_shared_storage_base(monkeypatch):
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    physical_model = nn.Linear(4, 4, bias=False)
    loads: list[object] = []

    def load_once(**kwargs):
        loads.append(kwargs)
        return physical_model

    base_model = nn.Linear(4, 4, bias=False, device="meta")

    monkeypatch.setattr(views_module, "get_model", load_once)
    monkeypatch.setattr(views_module, "initialize_model", lambda **_: base_model)
    monkeypatch.setattr(views_module, "set_model_tag", lambda _: nullcontext())
    monkeypatch.setattr(
        views_module, "set_default_torch_dtype", lambda _: nullcontext()
    )

    views.load_model()

    assert len(loads) == 1
    assert loads[0]["prefix"] == "asymspec_draft"
    assert views.full is not None and views.base is not None
    assert views.full.model is physical_model
    assert views.base.model is base_model
    assert views.full.model is not views.base.model
    assert views.full.model.weight.data_ptr() == views.base.model.weight.data_ptr()
    # Loading weights establishes only the shared physical model and logical
    # view identities.  Cache geometry is not valid until the platform has
    # normalized hybrid block/page sizes.
    assert views.full.state.cache_plan is None
    assert views.base.state.cache_plan is None
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


class _TestAttentionBackend:
    @staticmethod
    def indexes_kv_by_block_stride():
        return False

    @staticmethod
    def customize_spec(spec):
        return spec

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2, 3)


class _AttentionLayer(nn.Module, AttentionLayerBase):
    def get_attn_backend(self):
        return _TestAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.bfloat16,
        )


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


class _ManyLayerHybridModel(nn.Module):
    """Small structural stand-in for the real 8-attention/24-GDN draft."""

    def __init__(self, attention_layers: int, recurrent_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        total_layers = attention_layers + recurrent_layers
        attention_indices = set(range(attention_layers))
        for layer_index in range(total_layers):
            layer = nn.Module()
            if layer_index in attention_indices:
                layer.self_attn = _AttentionLayer()
            else:
                layer.linear_attn = _GDNLayer()
            self.layers.append(layer)


class _ForwardModel(nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(bias))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids, positions, inputs_embeds=None):
        del inputs_embeds
        self.calls.append((input_ids, positions))
        return input_ids.to(torch.float32).unsqueeze(-1) + self.weight

    def compute_logits(self, hidden_states):
        return hidden_states


def test_draft_dispatcher_selects_only_the_requested_tree(monkeypatch):
    """Role-local metadata is installed only around its selected tree call."""
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    full_model = _ForwardModel(1.0)
    base_model = _ForwardModel(2.0)
    views.bind_models(full_model, base_model)
    marker = SimpleNamespace(runtime=object())
    views.draft_cache_bindings = marker
    metadata = SimpleNamespace(
        role=AsymSpecViewRole.FULL,
        query_len=2,
        positions=torch.tensor([3, 4]),
        layer_metadata={"full": object()},
        layer_slot_mapping={"full": torch.tensor([3, 4])},
    )
    contexts = []

    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    def record_context(attn_metadata, vllm_config, **kwargs):
        contexts.append((attn_metadata, vllm_config, kwargs))
        return _Context()

    import vllm.v1.spec_decode.asymspec.draft_forward as forward_module

    monkeypatch.setattr(forward_module, "set_forward_context", record_context)
    result = execute_asymspec_draft_forward(
        role=AsymSpecViewRole.FULL,
        input_ids=torch.tensor([9, 10], dtype=torch.int32),
        metadata=metadata,
        views=views,
        cache_bindings=marker.runtime,
    )

    assert result.role is AsymSpecViewRole.FULL
    assert result.logits.tolist() == [[11.0]]
    assert len(full_model.calls) == 1
    assert not base_model.calls
    assert full_model.calls[0][1].tolist() == [[3, 4], [3, 4], [3, 4]]
    assert contexts == [
        (
            metadata.layer_metadata,
            config,
            {"num_tokens": 2, "slot_mapping": metadata.layer_slot_mapping},
        )
    ]


def test_qwen3_5_text_positions_rejects_non_vector_coordinates():
    assert qwen3_5_text_positions(torch.tensor([1, 2])).tolist() == [
        [1, 2],
        [1, 2],
        [1, 2],
    ]
    with pytest.raises(ValueError, match="one-dimensional"):
        qwen3_5_text_positions(torch.ones((1, 2), dtype=torch.int64))


def _make_canonical_driver_test_state():
    config, views, _, logical_runtime, bindings = _bind_metadata_test_runtime(
        full_max_model_len=64,
        compressed_max_model_len=32,
    )
    state = create_asymspec_request_state(
        request_id="canonical-driver",
        compressed_prompt_len=4,
        full_prompt_len=4,
        augmentation_offset=0,
        canonical_prompt_processed=False,
        logical_pools=logical_runtime,
    )
    return config, views, bindings, state


def test_canonical_driver_commits_only_successful_role_local_forwards(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    built = []

    def build_metadata(**kwargs):
        built.append((kwargs["role"], kwargs["query_start"], kwargs["query_len"]))
        return SimpleNamespace(role=kwargs["role"], query_len=kwargs["query_len"])

    def execute(**kwargs):
        return SimpleNamespace(role=kwargs["role"], logits=torch.tensor([[1.0]]))

    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )

    full = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    full.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    full.commit_token(5)
    assert built[:2] == [(AsymSpecViewRole.FULL, 0, 4), (AsymSpecViewRole.FULL, 4, 1)]
    assert state.full.canonical_len == 5
    assert state.base.canonical_len == state.compressed.canonical_len == 0
    assert full.counters.prefill_tokens_processed == 4
    assert full.counters.incremental_tokens_processed == 1
    assert full.counters.replayed_historical_tokens == 0

    base = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.BASE,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    base.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    base.commit_token(torch.tensor(5, dtype=torch.int64))
    assert built[2:] == [(AsymSpecViewRole.BASE, 0, 4), (AsymSpecViewRole.BASE, 4, 1)]
    assert state.base.canonical_len == state.base.observed_len == 5
    assert state.compressed.canonical_len == 0
    assert state.full.canonical_len == 5
    assert base.counters.replayed_historical_tokens == 0
    state.release()


def test_canonical_driver_does_not_advance_state_on_forward_failure(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )

    def fail(**kwargs):
        raise RuntimeError("draft forward failed")

    monkeypatch.setattr(canonical_driver_module, "execute_asymspec_draft_forward", fail)
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    with pytest.raises(RuntimeError, match="draft forward failed"):
        driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert state.full.canonical_len == 0
    assert driver.counters.total_forward_calls == 0
    assert driver.counters.prefill_tokens_processed == 0
    state.release()


def test_full_candidate_transaction_is_disposable_then_promotable(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    calls = []
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )

    def build_metadata(**kwargs):
        calls.append(
            (
                kwargs["query_start"],
                kwargs["canonical_end"],
                kwargs["allow_uncommitted_start"],
            )
        )
        return SimpleNamespace(role=kwargs["role"], query_len=kwargs["query_len"])

    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "execute_asymspec_draft_forward",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([[1.0]])
        ),
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    transaction = AsymSpecFullCandidateTransaction(driver)
    monkeypatch.setattr(transaction, "_snapshot_gdn_pages", lambda: ())
    transaction.execute_candidate((5, 6))
    assert state.full.canonical_len == 4
    assert calls == [(0, 0, False), (4, 4, True), (5, 5, True)]
    transaction.promote(1)
    assert state.full.canonical_len == 5
    assert transaction.counters.promoted_tokens == 1
    assert not transaction.active
    state.release()


def test_full_candidate_transaction_rolls_back_and_rejects_double_begin(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "execute_asymspec_draft_forward",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([[1.0]])
        ),
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    transaction = AsymSpecFullCandidateTransaction(driver)
    monkeypatch.setattr(transaction, "_snapshot_gdn_pages", lambda: ())
    transaction.execute_candidate((5, 6))
    with pytest.raises(RuntimeError, match="already active"):
        transaction.execute_candidate((5, 6))
    transaction.rollback()
    assert state.full.canonical_len == 4
    assert transaction.counters.rollbacks == 1
    transaction.execute_candidate((5, 6))
    assert transaction.promote(0) is None
    assert state.full.canonical_len == 4
    assert transaction.counters.rollbacks == 2
    with pytest.raises(RuntimeError, match="not active"):
        transaction.rollback()
    state.release()


def test_full_k2_proposer_reuses_canonical_logits_and_leaves_transaction_active(
    monkeypatch,
):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )
    calls: list[list[int]] = []

    def execute(**kwargs):
        token_ids = kwargs["input_ids"].tolist()
        calls.append(token_ids)
        # prompt -> A=7; consuming A -> B=9; consuming B -> arbitrary.
        logits = {
            (1, 2, 3, 4): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            (7,): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }.get(tuple(token_ids), [1.0])
        return SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([logits])
        )

    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    proposal = AsymSpecFullK2Proposer(driver).propose_k2()
    assert proposal.candidate_token_ids == (7, 9)
    assert proposal.transaction.active
    assert proposal.transaction.ready_to_promote
    assert state.full.canonical_len == 4
    # No forward was spent obtaining A; only A and B were executed.
    assert calls == [[1, 2, 3, 4], [7], [9]]
    proposal.transaction.promote(1)
    assert driver.last_result is proposal.candidate_results[0]
    assert state.full.canonical_len == 5
    state.release()


def test_full_k2_proposer_always_constructs_two_positions_for_eos_like_argmax(
    monkeypatch,
):
    """Frozen K=2 behavior does not short-circuit the inner loop at EOS."""
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )
    calls = []

    def execute(**kwargs):
        calls.append(kwargs["input_ids"].tolist())
        return SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([[1.0, 0.0]])
        )

    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    proposal = AsymSpecFullK2Proposer(driver).propose_k2()
    assert proposal.candidate_token_ids == (0, 0)
    assert calls == [[1, 2, 3, 4], [0], [0]]
    proposal.transaction.rollback()
    state.release()


def test_full_candidate_transaction_restores_state_after_forward_exception(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "execute_asymspec_draft_forward",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([[1.0]])
        ),
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.FULL,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    transaction = AsymSpecFullCandidateTransaction(driver)
    snapshots = (SimpleNamespace(),)
    restored = []
    monkeypatch.setattr(transaction, "_snapshot_gdn_pages", lambda: snapshots)
    monkeypatch.setattr(
        transaction, "_restore_gdn_pages", lambda value: restored.append(value)
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "execute_asymspec_draft_forward",
        lambda **_: (_ for _ in ()).throw(RuntimeError("candidate forward failed")),
    )
    with pytest.raises(RuntimeError, match="candidate forward failed"):
        transaction.execute_candidate((5, 6))
    assert restored == [snapshots]
    assert not transaction.active
    assert state.full.canonical_len == 4
    state.release()


def test_deferred_base_observes_then_catches_up_once(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    calls = []

    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"],
            query_len=kwargs["query_len"],
            query_start=kwargs["query_start"],
        ),
    )

    def execute(**kwargs):
        calls.append((kwargs["input_ids"].tolist(), kwargs["metadata"].query_start))
        return SimpleNamespace(role=kwargs["role"], logits=torch.tensor([[1.0]]))

    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.BASE,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    driver.begin_deferred_base()
    driver.observe_committed_token(5)
    driver.observe_committed_token(6)
    assert state.compressed.canonical_len == state.base.observed_len == 6
    assert state.base.canonical_len == 4
    assert state.base.pending_token_ids == [5, 6]

    driver.catch_up_base()
    assert calls == [([1, 2, 3, 4], 0), ([5, 6], 4)]
    assert state.base.canonical_len == state.base.observed_len == 6
    assert state.base.pending_token_ids == []
    assert driver.counters.catch_up_forward_calls == 1
    assert driver.counters.catch_up_tokens_processed == 2
    assert driver.counters.replayed_historical_tokens == 0
    state.release()


def test_deferred_base_catch_up_is_metadata_transactional(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_len=kwargs["query_len"]
        ),
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "execute_asymspec_draft_forward",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], logits=torch.tensor([[1.0]])
        ),
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.BASE,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    driver.begin_deferred_base()
    driver.observe_committed_token(5)
    original_forward = canonical_driver_module.execute_asymspec_draft_forward

    def fail(**kwargs):
        if kwargs["input_ids"].numel() == 1:
            raise RuntimeError("catch-up failed")
        return original_forward(**kwargs)

    monkeypatch.setattr(canonical_driver_module, "execute_asymspec_draft_forward", fail)
    with pytest.raises(RuntimeError, match="catch-up failed"):
        driver.catch_up_base()
    assert state.base.canonical_len == 4
    assert state.base.observed_len == state.compressed.canonical_len == 5
    assert state.base.pending_token_ids == [5]
    assert driver.counters.catch_up_tokens_processed == 0
    state.release()


def test_deferred_base_long_lag_crosses_attention_blocks():
    _, _, _, logical_runtime, _ = _bind_metadata_test_runtime(
        full_max_model_len=4096,
        compressed_max_model_len=4096,
    )
    state = create_asymspec_request_state(
        request_id="deferred-long-lag",
        compressed_prompt_len=800,
        full_prompt_len=800,
        augmentation_offset=0,
        logical_pools=logical_runtime,
    )
    state.advance_target(list(range(900)))
    assert state.base.canonical_len == 800
    assert state.base.observed_len == state.compressed.canonical_len == 1700
    assert (
        len(
            state.block_tables.attention_block_ids(
                AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
            )
        )
        == 3
    )
    assert state.catch_up_base() == tuple(range(900))
    assert state.base.canonical_len == 1700
    assert state.base.pending_token_ids == []
    state.release()


def test_base_pair_scorer_reuses_canonical_logits_and_restores_gdn(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_start=kwargs["query_start"]
        ),
    )
    calls = []

    def execute(**kwargs):
        ids = kwargs["input_ids"].tolist()
        calls.append((ids, kwargs["metadata"].query_start))
        logits = [0.0] * 12
        logits[7 if len(ids) > 1 else 9] = 1.0
        return SimpleNamespace(role=kwargs["role"], logits=torch.tensor([logits]))

    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.BASE,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    scorer = AsymSpecBasePairScorer(driver)
    snapshots = (SimpleNamespace(),)
    restored = []
    monkeypatch.setattr(scorer, "_snapshot_gdn_pages", lambda: snapshots)
    monkeypatch.setattr(
        scorer,
        "_restore_gdn_pages",
        lambda value: restored.append(value),
    )
    score = scorer.score_pair((7, 9))
    assert calls == [([1, 2, 3, 4], 0), ([7], 4)]
    assert score.candidate_token_ids == (7, 9)
    assert float(score.score_a) == 1.0
    assert float(score.score_b) == 1.0
    assert score.candidate_scoring_forwards == 1
    assert scorer.counters.candidate_tokens_executed == 1
    assert state.base.canonical_len == 4
    assert restored == [snapshots]
    state.release()


def test_base_pair_scorer_catches_up_before_disposable_candidate(monkeypatch):
    config, views, bindings, state = _make_canonical_driver_test_state()
    monkeypatch.setattr(
        canonical_driver_module,
        "initialize_fresh_asymspec_view_state",
        lambda **_: None,
    )
    monkeypatch.setattr(
        canonical_driver_module,
        "build_asymspec_view_execution_metadata",
        lambda **kwargs: SimpleNamespace(
            role=kwargs["role"], query_start=kwargs["query_start"]
        ),
    )
    calls = []

    def execute(**kwargs):
        ids = kwargs["input_ids"].tolist()
        calls.append((ids, kwargs["metadata"].query_start))
        return SimpleNamespace(role=kwargs["role"], logits=torch.tensor([[1.0]]))

    monkeypatch.setattr(
        canonical_driver_module, "execute_asymspec_draft_forward", execute
    )
    driver = AsymSpecCanonicalDraftDriver(
        role=AsymSpecViewRole.BASE,
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
    )
    driver.prefill(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    driver.begin_deferred_base()
    driver.observe_committed_token(5)
    driver.observe_committed_token(6)
    scorer = AsymSpecBasePairScorer(driver)
    monkeypatch.setattr(scorer, "_snapshot_gdn_pages", lambda: ())
    score = scorer.score_pair((0, 0))
    assert calls == [([1, 2, 3, 4], 0), ([5, 6], 4), ([0], 6)]
    assert score.catch_up_tokens == 2
    assert scorer.counters.catch_up_tokens_processed == 2
    assert state.base.canonical_len == state.base.observed_len == 6
    assert state.base.pending_token_ids == []
    state.release()


def _make_kv_spec_runner(config, views):
    return SimpleNamespace(
        vllm_config=config,
        speculative_config=config.speculative_config,
        asymspec_draft_views=views,
        shared_kv_cache_layers={},
        kv_cache_dtype=torch.bfloat16,
    )


def test_asymspec_target_cache_specs_exclude_draft_by_module_ownership(monkeypatch):
    """TARGET sees only target modules; FULL/BASE own the shared draft tree."""
    config = _make_config()
    draft_model = _ManyLayerHybridModel(attention_layers=8, recurrent_layers=24)
    target_model = _ManyLayerHybridModel(attention_layers=16, recurrent_layers=48)
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(draft_model)
    runner = _make_kv_spec_runner(config, views)

    static_layers = {
        **{
            f"target.{name}": module
            for name, module in target_model.named_modules()
            if isinstance(module, AttentionLayerBase)
        },
        **{
            f"draft.{name}": module
            for name, module in draft_model.named_modules()
            if isinstance(module, AttentionLayerBase)
        },
    }
    assert len(static_layers) == 96
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_layers_from_vllm_config",
        lambda *_: static_layers,
    )

    target_specs = GPUModelRunner.get_kv_cache_spec(runner)

    assert len(target_specs) == 64
    assert all(name.startswith("target.") for name in target_specs)
    assert (
        sum(isinstance(spec, FullAttentionSpec) for spec in target_specs.values()) == 16
    )
    assert sum(isinstance(spec, MambaSpec) for spec in target_specs.values()) == 48
    assert all(
        not views.owns_physical_module(module)
        for name, module in static_layers.items()
        if name in target_specs
    )

    # The physical target is represented once, while the same physical draft
    # tree is represented once per logical role: 64 + 32 + 32 = 128.
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )
    assert len(full_plan.layer_specs) == 32
    assert len(base_plan.layer_specs) == 32
    assert len(global_plan.merged_specs) == 128


def test_non_asymspec_target_cache_spec_collection_is_unfiltered(monkeypatch):
    """Ordinary methods retain the exact static-context collection behavior."""
    config = _make_config(method="draft_model")
    target_layer = _AttentionLayer()
    draft_layer = _AttentionLayer()
    static_layers = {"target.attn": target_layer, "draft.attn": draft_layer}
    runner = SimpleNamespace(
        vllm_config=config,
        speculative_config=config.speculative_config,
        # Deliberately present: this must not affect a non-AsymSpec method.
        asymspec_draft_views=SimpleNamespace(
            owns_physical_module=lambda _: (_ for _ in ()).throw(AssertionError())
        ),
        shared_kv_cache_layers={},
        kv_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_layers_from_vllm_config",
        lambda *_: static_layers,
    )

    specs = GPUModelRunner.get_kv_cache_spec(runner)

    assert set(specs) == set(static_layers)


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
    assert full.model is model
    assert full.model is not base.model
    assert full.model.layers[0].self_attn is not base.model.layers[0].self_attn
    assert full.model.layers[1].linear_attn is not base.model.layers[1].linear_attn
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


def test_hybrid_cache_plans_are_role_owned_and_allocation_free():
    views = AsymSpecDraftViews(_make_config(), torch.device("cpu"))
    model = _HybridQwenLikeModel()
    views.bind_model(model)
    views.initialize_state_specs()

    views.initialize_cache_plans()

    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    full_plan = full.state.cache_plan
    base_plan = base.state.cache_plan
    assert full_plan is not None and base_plan is not None
    assert full_plan is not base_plan
    assert full_plan.role is AsymSpecViewRole.FULL
    assert base_plan.role is AsymSpecViewRole.BASE
    assert full_plan.max_model_len == 1024
    assert base_plan.max_model_len == 128
    assert set(full_plan.layer_specs) == {
        "layers.0.self_attn",
        "layers.1.linear_attn",
    }
    assert set(base_plan.layer_specs) == set(full_plan.layer_specs)
    assert isinstance(full_plan.layer_specs["layers.0.self_attn"], FullAttentionSpec)
    assert isinstance(full_plan.layer_specs["layers.1.linear_attn"], MambaSpec)
    assert len(full_plan.cache_groups) == len(base_plan.cache_groups) > 0
    assert full_plan.cache_groups is not base_plan.cache_groups
    assert all(
        group is not other
        for group, other in zip(full_plan.cache_groups, base_plan.cache_groups)
    )
    assert full.model is model
    assert full.model is not base.model
    assert full.model.layers[0].self_attn is not base.model.layers[0].self_attn
    assert full.model.layers[1].linear_attn is not base.model.layers[1].linear_attn
    assert full.state.kv_state is base.state.kv_state is None
    assert full.state.recurrent_state is base.state.recurrent_state is None
    assert full.state.position_state is base.state.position_state is None
    assert not hasattr(full_plan, "cache_group_id")


def test_cache_plans_consume_post_normalization_geometry_only():
    config = _make_config()
    assert config.cache_config is not None
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()

    # This is the state immediately after model load: plan ownership is
    # pending, because backend normalization has not run yet.
    full = views.view(AsymSpecViewRole.FULL)
    base = views.view(AsymSpecViewRole.BASE)
    assert full.state.cache_plan is None
    assert base.state.cache_plan is None

    # Model the normalized geometry supplied by
    # Platform.update_block_size_for_backend().  The planner must consume
    # these values rather than reconstructing pre-normalization geometry.
    config.cache_config.block_size = 800
    config.cache_config.mamba_page_size_padded = 51_200
    views.initialize_cache_plans()

    full_plan = full.state.cache_plan
    base_plan = base.state.cache_plan
    assert full_plan is not None and base_plan is not None
    for plan in (full_plan, base_plan):
        attention_spec = plan.layer_specs["layers.0.self_attn"]
        mamba_spec = plan.layer_specs["layers.1.linear_attn"]
        assert isinstance(attention_spec, FullAttentionSpec)
        assert isinstance(mamba_spec, MambaSpec)
        assert attention_spec.block_size == 800
        assert attention_spec.page_size_bytes == 51_200
        assert mamba_spec.page_size_bytes == 51_200


def test_model_runner_finalizes_only_existing_asymspec_views():
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    runner = SimpleNamespace(asymspec_draft_views=views)

    GPUModelRunner.initialize_asymspec_cache_plans(runner)
    assert views.view(AsymSpecViewRole.FULL).state.cache_plan is not None

    ordinary_runner = SimpleNamespace()
    # The model-runner hook is a no-op when no AsymSpec views exist; executor
    # paths additionally avoid calling it for ordinary speculative methods.
    GPUModelRunner.initialize_asymspec_cache_plans(ordinary_runner)


def test_cache_plan_rejects_noncompact_mamba_metadata():
    config = _make_config()
    assert config.cache_config is not None
    config.cache_config.mamba_cache_mode = "align"
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()

    with pytest.raises(ValueError, match="Mamba cache mode 'none'"):
        views.initialize_cache_plans()


def test_global_cache_namespace_is_reversible_and_role_aware():
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None

    target_specs = {
        "target.layers.0.self_attn": FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.bfloat16,
        ),
        "target.layers.1.linear_attn": MambaSpec(
            block_size=256,
            shapes=((4, 8), (2, 3, 5)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="none",
            num_speculative_blocks=2,
        ),
    }
    full_layer_names_before = set(full_plan.layer_specs)
    base_layer_names_before = set(base_plan.layer_specs)
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )

    native_target_groups = get_kv_cache_groups(config, dict(target_specs))
    target_indices = global_plan.group_indices_for_domain(AsymSpecCacheDomain.TARGET)
    composed_target_layer_names = [
        global_plan.cache_groups[index].layer_names for index in target_indices
    ]
    assert composed_target_layer_names == [
        group.layer_names for group in native_target_groups
    ]
    composed_target_specs = [
        global_plan.cache_groups[index].kv_cache_spec for index in target_indices
    ]
    assert composed_target_specs == [
        group.kv_cache_spec for group in native_target_groups
    ]

    full_name = global_plan.registry.global_name_for(
        AsymSpecViewRole.FULL, "layers.0.self_attn"
    )
    base_name = global_plan.registry.global_name_for(
        AsymSpecViewRole.BASE, "layers.0.self_attn"
    )
    assert set(target_specs).issubset(global_plan.merged_specs)
    assert full_name != base_name
    assert full_name not in target_specs and base_name not in target_specs
    assert global_plan.registry.binding_for(full_name).role is AsymSpecViewRole.FULL
    assert global_plan.registry.binding_for(base_name).role is AsymSpecViewRole.BASE
    assert (
        global_plan.registry.binding_for(full_name).physical_layer_name
        == "layers.0.self_attn"
    )
    assert (
        global_plan.merged_specs[full_name]
        is full_plan.layer_specs["layers.0.self_attn"]
    )
    assert (
        global_plan.merged_specs[base_name]
        is base_plan.layer_specs["layers.0.self_attn"]
    )
    assert len(global_plan.merged_specs) == (
        len(target_specs) + len(full_plan.layer_specs) + len(base_plan.layer_specs)
    )
    assert set(full_plan.layer_specs) == full_layer_names_before
    assert set(base_plan.layer_specs) == base_layer_names_before
    assert not hasattr(global_plan.registry, "cache_group_id")
    with pytest.raises(ValueError, match="reserved AsymSpec namespace"):
        compose_asymspec_global_cache_plan(
            vllm_config=_make_config(),
            target_specs={full_name: target_specs["target.layers.0.self_attn"]},
            full_plan=full_plan,
            base_plan=base_plan,
        )


def test_global_cache_groups_are_role_separated_and_native_within_domains():
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        f"target.{name}": spec for name, spec in full_plan.layer_specs.items()
    }

    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )

    assert len(global_plan.cache_groups) == (
        len(global_plan.group_indices_for_domain(AsymSpecCacheDomain.TARGET))
        + len(global_plan.group_indices_for_domain(AsymSpecCacheDomain.FULL))
        + len(global_plan.group_indices_for_domain(AsymSpecCacheDomain.BASE))
    )
    assert global_plan.group_domains == tuple(
        sorted(
            global_plan.group_domains,
            key=lambda domain: {
                AsymSpecCacheDomain.TARGET: 0,
                AsymSpecCacheDomain.FULL: 1,
                AsymSpecCacheDomain.BASE: 2,
            }[domain],
        )
    )

    def domain_for_layer(name: str) -> AsymSpecCacheDomain:
        binding = global_plan.registry.global_to_binding.get(name)
        if binding is None:
            return AsymSpecCacheDomain.TARGET
        return (
            AsymSpecCacheDomain.FULL
            if binding.role is AsymSpecViewRole.FULL
            else AsymSpecCacheDomain.BASE
        )

    seen_layers: set[str] = set()
    for index, group in enumerate(global_plan.cache_groups):
        assert {domain_for_layer(name) for name in group.layer_names} == {
            global_plan.domain_for_group(index)
        }
        seen_layers.update(group.layer_names)
    assert seen_layers == set(global_plan.merged_specs)
    for name in seen_layers:
        binding = global_plan.registry.global_to_binding.get(name)
        if binding is not None:
            assert (
                global_plan.registry.global_name_for(
                    binding.role, binding.physical_layer_name
                )
                == name
            )
    for domain, original_groups in (
        (AsymSpecCacheDomain.FULL, full_plan.cache_groups),
        (AsymSpecCacheDomain.BASE, base_plan.cache_groups),
    ):
        composed_groups = [
            global_plan.cache_groups[index]
            for index in global_plan.group_indices_for_domain(domain)
        ]
        assert [
            [
                global_plan.registry.binding_for(name).physical_layer_name
                for name in group.layer_names
            ]
            for group in composed_groups
        ] == [group.layer_names for group in original_groups]
    assert all(
        group.kv_cache_spec is original.kv_cache_spec
        for group, original in zip(
            global_plan.cache_groups[
                len(global_plan.group_indices_for_domain(AsymSpecCacheDomain.TARGET)) :
            ],
            full_plan.cache_groups + base_plan.cache_groups,
        )
    )

    compatibility = characterize_asymspec_allocator_compatibility(
        global_plan=global_plan,
        vllm_config=config,
        available_memory=1 << 30,
    )
    # The generic allocator either builds metadata or reports a precise pure
    # compatibility error; it never receives a cache tensor or block pool here.
    if compatibility.config_error is None:
        assert compatibility.num_blocks is not None
        assert compatibility.tensor_shares_domains is True
    else:
        assert compatibility.num_blocks is None


def test_domain_allocation_plan_uses_independent_budgets_and_compact_mamba():
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        f"target.{name}": spec for name, spec in full_plan.layer_specs.items()
    }
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )

    plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=131072,
        compressed_max_model_len=8192,
        available_memory_bytes=1 << 40,
    )

    assert plans.full.max_tokens == 131072
    assert plans.target.max_tokens == 8192
    assert plans.base.max_tokens == 8192
    assert plans.target.domain is AsymSpecCacheDomain.TARGET
    assert plans.full.domain is AsymSpecCacheDomain.FULL
    assert plans.base.domain is AsymSpecCacheDomain.BASE
    assert {plan.backing_pool_key for plan in plans.domains} == {
        AsymSpecCacheDomain.TARGET,
        AsymSpecCacheDomain.FULL,
        AsymSpecCacheDomain.BASE,
    }

    for domain_plan in plans.domains:
        assert domain_plan.minimum_bytes == sum(
            group.minimum_bytes for group in domain_plan.groups
        )
        assert {group.page_size_bytes for group in domain_plan.groups} == {
            domain_plan.page_size_bytes
        }
        assert all(group.minimum_bytes > 0 for group in domain_plan.groups)
        assert all(not hasattr(group, "block_pool") for group in domain_plan.groups)

    full_attention = next(
        group
        for group in plans.full.groups
        if isinstance(group.kv_cache_spec, FullAttentionSpec)
    )
    full_mamba = next(
        group
        for group in plans.full.groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    )
    target_attention = next(
        group
        for group in plans.target.groups
        if isinstance(group.kv_cache_spec, FullAttentionSpec)
    )
    target_mamba = next(
        group
        for group in plans.target.groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    )
    assert full_attention.min_blocks_per_request == 8193
    assert target_attention.min_blocks_per_request == 513
    # Compact mode is bounded by committed + K state slots plus BlockPool's
    # physical null block, rather than by 131072 / 4096 token positions.
    assert full_mamba.request_block_table_entries == 3
    assert full_mamba.min_blocks_per_request == 4
    assert target_mamba.min_blocks_per_request == 4
    assert plans.minimum_bytes == sum(plan.minimum_bytes for plan in plans.domains)
    assert plans.has_minimum_capacity is True
    assert plans.remaining_unassigned_bytes == (1 << 40) - plans.minimum_bytes


def test_domain_allocation_plan_permits_different_pages_across_domains():
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        "target.attn": FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=8, dtype=torch.bfloat16
        ),
        "target.mamba": MambaSpec(
            block_size=256,
            shapes=((4, 8),),
            dtypes=(torch.bfloat16,),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="none",
            num_speculative_blocks=2,
        ),
    }
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )
    plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=1024,
        compressed_max_model_len=128,
    )
    assert plans.target.page_size_bytes != plans.full.page_size_bytes
    assert plans.full.page_size_bytes == plans.base.page_size_bytes
    assert {group.page_size_bytes for group in plans.target.groups} == {
        plans.target.page_size_bytes
    }


def test_physical_cache_plan_is_one_independent_tensor_per_logical_layer():
    """Physical ownership follows plans; it never uses generic sharing."""
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    model = _HybridQwenLikeModel()
    views.bind_model(model)
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        f"target.{name}": spec for name, spec in full_plan.layer_specs.items()
    }
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )
    domain_plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=1024,
        compressed_max_model_len=128,
    )

    physical_plan = build_asymspec_physical_cache_plan(
        global_plan=global_plan, domain_plans=domain_plans
    )
    assert len(physical_plan.tensors) == len(global_plan.merged_specs) == 6
    assert physical_plan.total_bytes == domain_plans.minimum_bytes
    assert all(
        allocation.kv_cache_tensor.shared_by == [allocation.global_layer_name]
        for allocation in physical_plan.tensors
    )
    assert {allocation.domain for allocation in physical_plan.tensors} == set(
        AsymSpecCacheDomain
    )
    for domain in AsymSpecCacheDomain:
        assert (
            physical_plan.bytes_for_domain(domain)
            == getattr(domain_plans, domain.value).minimum_bytes
        )

    full_attn = next(
        allocation
        for allocation in physical_plan.tensors_for_domain(AsymSpecCacheDomain.FULL)
        if isinstance(allocation.kv_cache_spec, FullAttentionSpec)
    )
    base_attn = next(
        allocation
        for allocation in physical_plan.tensors_for_domain(AsymSpecCacheDomain.BASE)
        if isinstance(allocation.kv_cache_spec, FullAttentionSpec)
    )
    assert full_attn.physical_layer_name == base_attn.physical_layer_name
    assert full_attn.global_layer_name != base_attn.global_layer_name
    assert full_attn.num_pages == 65
    assert base_attn.num_pages == 9
    assert all(
        not hasattr(allocation, "block_pool") for allocation in physical_plan.tensors
    )


def test_raw_physical_allocator_keeps_full_and_base_storage_distinct_on_cpu():
    """The allocation primitive is unbound and has no generic cache side effect."""
    config = _make_config()
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(_HybridQwenLikeModel())
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs={
            f"target.{name}": spec for name, spec in full_plan.layer_specs.items()
        },
        full_plan=full_plan,
        base_plan=base_plan,
    )
    domain_plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=128,
        compressed_max_model_len=64,
    )
    physical_plan = build_asymspec_physical_cache_plan(
        global_plan=global_plan, domain_plans=domain_plans
    )

    runtime = allocate_asymspec_physical_cache_tensors(
        plan=physical_plan, device=torch.device("cpu")
    )
    assert set(runtime.raw_tensors) == {
        allocation.global_layer_name for allocation in physical_plan.tensors
    }
    assert runtime.total_bytes == physical_plan.total_bytes
    pointers = {
        tensor.untyped_storage().data_ptr() for tensor in runtime.raw_tensors.values()
    }
    assert len(pointers) == len(runtime.raw_tensors)
    full_name = next(
        allocation.global_layer_name
        for allocation in physical_plan.tensors_for_domain(AsymSpecCacheDomain.FULL)
        if isinstance(allocation.kv_cache_spec, FullAttentionSpec)
    )
    base_name = next(
        allocation.global_layer_name
        for allocation in physical_plan.tensors_for_domain(AsymSpecCacheDomain.BASE)
        if isinstance(allocation.kv_cache_spec, FullAttentionSpec)
    )
    assert (
        runtime.raw_tensors[full_name].data_ptr()
        != runtime.raw_tensors[base_name].data_ptr()
    )
    # This standalone object is deliberately not a normal model-runner cache.
    assert not hasattr(runtime, "kv_caches")


def _build_large_asymspec_physical_plan():
    """Build a CPU-only stand-in for the real 64/32/32 cache inventory."""
    config = _make_config()
    draft_model = _ManyLayerHybridModel(attention_layers=8, recurrent_layers=24)
    target_model = _ManyLayerHybridModel(attention_layers=16, recurrent_layers=48)
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(draft_model)
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        name: layer.get_kv_cache_spec(config)
        for name, layer in target_model.named_modules()
        if isinstance(layer, (AttentionLayerBase, MambaBase))
    }
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )
    domain_plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=1024,
        compressed_max_model_len=128,
    )
    return build_asymspec_physical_cache_plan(
        global_plan=global_plan, domain_plans=domain_plans
    )


def test_logical_cache_plan_reproduces_eight_semantic_pool_topology():
    physical_plan = _build_large_asymspec_physical_plan()
    logical_plan = build_asymspec_logical_cache_plan(physical_plan)

    assert [group.semantic_group for group in logical_plan.groups] == [
        AsymSpecLogicalCacheGroup.TARGET_MAMBA_A,
        AsymSpecLogicalCacheGroup.TARGET_MAMBA_B,
        AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
        AsymSpecLogicalCacheGroup.FULL_ATTENTION,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_A,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_A,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_B,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_B,
    ]
    groups = {group.semantic_group: group for group in logical_plan.groups}
    assert (
        len(groups[AsymSpecLogicalCacheGroup.TARGET_MAMBA_A].member_layer_names) == 24
    )
    assert (
        len(groups[AsymSpecLogicalCacheGroup.TARGET_MAMBA_B].member_layer_names) == 24
    )
    assert (
        len(groups[AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION].member_layer_names)
        == 24
    )
    assert len(groups[AsymSpecLogicalCacheGroup.FULL_ATTENTION].member_layer_names) == 8
    assert len(groups[AsymSpecLogicalCacheGroup.BASE_MAMBA_A].member_layer_names) == 12
    assert len(groups[AsymSpecLogicalCacheGroup.BASE_MAMBA_B].member_layer_names) == 12
    assert len(groups[AsymSpecLogicalCacheGroup.FULL_MAMBA_A].member_layer_names) == 12
    assert len(groups[AsymSpecLogicalCacheGroup.FULL_MAMBA_B].member_layer_names) == 12

    compressed = groups[AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION]
    full_attention = groups[AsymSpecLogicalCacheGroup.FULL_ATTENTION]
    assert compressed.domains == {
        AsymSpecCacheDomain.TARGET,
        AsymSpecCacheDomain.BASE,
    }
    assert compressed.intentionally_shared_coordinates is True
    assert full_attention.domains == {AsymSpecCacheDomain.FULL}
    assert full_attention.intentionally_shared_coordinates is False
    assert set(logical_plan.layer_to_group) == {
        allocation.global_layer_name for allocation in physical_plan.tensors
    }


def test_logical_block_pools_share_compressed_coordinates_not_storage():
    physical_plan = _build_large_asymspec_physical_plan()
    logical_plan = build_asymspec_logical_cache_plan(physical_plan)
    runtime = instantiate_asymspec_logical_block_pools(logical_plan)

    assert len(runtime.pools) == 8
    assert len({id(pool) for pool in runtime.pools.values()}) == 8
    compressed_pool = runtime.pools[AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION]
    full_pool = runtime.pools[AsymSpecLogicalCacheGroup.FULL_ATTENTION]
    assert compressed_pool is not full_pool
    assert compressed_pool.enable_caching is False
    assert compressed_pool.null_block.block_id == 0

    compressed_group = next(
        group
        for group in logical_plan.groups
        if group.semantic_group is AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
    )
    target_attention_name = next(
        name
        for name in compressed_group.member_layer_names
        if logical_plan.physical_tensors_by_layer[name].domain
        is AsymSpecCacheDomain.TARGET
    )
    base_attention_name = next(
        name
        for name in compressed_group.member_layer_names
        if logical_plan.physical_tensors_by_layer[name].domain
        is AsymSpecCacheDomain.BASE
    )
    assert runtime.pool_for_layer(target_attention_name) is compressed_pool
    assert runtime.pool_for_layer(base_attention_name) is compressed_pool

    physical_runtime = allocate_asymspec_physical_cache_tensors(
        plan=physical_plan, device=torch.device("cpu")
    )
    assert (
        physical_runtime.raw_tensors[target_attention_name].data_ptr()
        != physical_runtime.raw_tensors[base_attention_name].data_ptr()
    )

    compressed_ids = allocate_synthetic_attention_blocks(
        runtime, AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION, 128
    )
    full_ids = allocate_synthetic_attention_blocks(
        runtime, AsymSpecLogicalCacheGroup.FULL_ATTENTION, 1024
    )
    assert compressed_ids == tuple(range(1, 9))
    assert full_ids == tuple(range(1, 65))
    compressed_pool.free_blocks(
        compressed_pool.blocks[block_id] for block_id in compressed_ids
    )
    full_pool.free_blocks(full_pool.blocks[block_id] for block_id in full_ids)

    recurrent_pools = [
        runtime.pools[group]
        for group in (
            AsymSpecLogicalCacheGroup.TARGET_MAMBA_A,
            AsymSpecLogicalCacheGroup.TARGET_MAMBA_B,
            AsymSpecLogicalCacheGroup.BASE_MAMBA_A,
            AsymSpecLogicalCacheGroup.BASE_MAMBA_B,
            AsymSpecLogicalCacheGroup.FULL_MAMBA_A,
            AsymSpecLogicalCacheGroup.FULL_MAMBA_B,
        )
    ]
    assert all(pool.num_gpu_blocks == 4 for pool in recurrent_pools)
    assert all(pool.null_block.block_id == 0 for pool in recurrent_pools)
    for pool in recurrent_pools:
        blocks = pool.get_new_blocks(3)
        assert [block.block_id for block in blocks] == [1, 2, 3]
        pool.free_blocks(blocks)


def test_request_state_keeps_full_and_deferred_base_coordinates_independent():
    """Request-local tables retain frozen FULL/BASE coordinate ownership."""
    physical_plan = _build_large_asymspec_physical_plan()
    logical_runtime = instantiate_asymspec_logical_block_pools(
        build_asymspec_logical_cache_plan(physical_plan)
    )
    state = create_asymspec_request_state(
        request_id="synthetic-request",
        compressed_prompt_len=96,
        full_prompt_len=960,
        augmentation_offset=864,
        logical_pools=logical_runtime,
    )

    tables = state.block_tables
    assert tables.target_attention_table is tables.base_attention_table
    assert tables.full_attention_table is not tables.compressed_attention_table
    assert state.compressed.canonical_len == state.base.canonical_len == 96
    assert state.base.observed_len == 96
    assert state.full.canonical_len == 960
    assert state.full.position_for_compressed(96) == 960

    compressed_ids = tables.attention_block_ids(
        AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
    )
    full_ids = tables.attention_block_ids(AsymSpecLogicalCacheGroup.FULL_ATTENTION)
    assert compressed_ids == tuple(range(1, 7))
    assert full_ids == tuple(range(1, 61))

    assert list(state.advance_target([101, 102, 103, 104])) == list(range(96, 100))
    assert state.compressed.canonical_len == 100
    assert state.base.canonical_len == 96
    assert state.base.observed_len == 100
    assert state.base.pending_token_ids == [101, 102, 103, 104]
    assert list(state.advance_full(4)) == list(range(960, 964))
    assert state.full.canonical_len == 964
    assert state.compressed.canonical_len == 100
    assert tables.target_attention_table is tables.base_attention_table

    assert state.catch_up_base() == (101, 102, 103, 104)
    assert state.base.canonical_len == state.compressed.canonical_len == 100
    assert state.base.pending_token_ids == []

    assert set(tables.recurrent_slots) == {
        AsymSpecLogicalCacheGroup.TARGET_MAMBA_A,
        AsymSpecLogicalCacheGroup.TARGET_MAMBA_B,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_A,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_B,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_A,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_B,
    }
    for group, slots in tables.recurrent_slots.items():
        assert slots.null_block_id == 0
        assert slots.committed_block_id == 1
        assert slots.speculative_block_ids == (2, 3)
        assert tables.table_by_group[group].slot_mapping_mode.name == "NONE"
        assert tables.table_by_group[group].block_table.np[0, :3].tolist() == [
            1,
            2,
            3,
        ]

    state.release()
    assert all(
        pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
        for pool in logical_runtime.pools.values()
    )


def test_request_state_requires_a_valid_augmented_full_coordinate_mapping():
    physical_plan = _build_large_asymspec_physical_plan()
    logical_runtime = instantiate_asymspec_logical_block_pools(
        build_asymspec_logical_cache_plan(physical_plan)
    )
    with pytest.raises(ValueError, match="augmentation offset"):
        create_asymspec_request_state(
            request_id="bad-offset",
            compressed_prompt_len=128,
            full_prompt_len=1024,
            augmentation_offset=895,
            logical_pools=logical_runtime,
        )


def _build_bindable_draft_cache_runtime(
    *,
    full_max_model_len: int = 1024,
    compressed_max_model_len: int = 128,
    attention_block_size: int | None = None,
):
    """Build the real 8/24 draft shape without a GPU or model forward."""
    config = _make_config()
    config.model_config.max_model_len = full_max_model_len
    config.speculative_config.asymspec_compressed_max_model_len = (
        compressed_max_model_len
    )
    if attention_block_size is not None:
        config.cache_config.block_size = attention_block_size
    draft_model = _ManyLayerHybridModel(attention_layers=8, recurrent_layers=24)
    target_model = _ManyLayerHybridModel(attention_layers=16, recurrent_layers=48)
    views = AsymSpecDraftViews(config, torch.device("cpu"))
    views.bind_model(draft_model)
    views.initialize_state_specs()
    views.initialize_cache_plans()
    full_plan = views.view(AsymSpecViewRole.FULL).state.cache_plan
    base_plan = views.view(AsymSpecViewRole.BASE).state.cache_plan
    assert full_plan is not None and base_plan is not None
    target_specs = {
        name: layer.get_kv_cache_spec(config)
        for name, layer in target_model.named_modules()
        if isinstance(layer, (AttentionLayerBase, MambaBase))
    }
    global_plan = compose_asymspec_global_cache_plan(
        vllm_config=config,
        target_specs=target_specs,
        full_plan=full_plan,
        base_plan=base_plan,
    )
    domain_plans = build_asymspec_domain_allocation_plans(
        global_plan=global_plan,
        full_max_model_len=full_max_model_len,
        compressed_max_model_len=compressed_max_model_len,
    )
    physical_plan = build_asymspec_physical_cache_plan(
        global_plan=global_plan, domain_plans=domain_plans
    )
    physical_runtime = allocate_asymspec_physical_cache_tensors(
        plan=physical_plan, device=torch.device("cpu")
    )
    logical_runtime = instantiate_asymspec_logical_block_pools(
        build_asymspec_logical_cache_plan(physical_plan)
    )
    return config, views, global_plan, physical_runtime, logical_runtime


def test_draft_cache_binder_permanently_binds_distinct_full_and_base_trees():
    config, views, global_plan, physical_runtime, logical_runtime = (
        _build_bindable_draft_cache_runtime()
    )
    runtime = bind_asymspec_draft_caches(
        views=views,
        physical_runtime=physical_runtime,
        global_plan=global_plan,
        logical_pools=logical_runtime,
        vllm_config=config,
    )

    assert len(runtime.bindings) == 64
    assert len(runtime.full_bindings) == 32
    assert len(runtime.base_bindings) == 32
    assert runtime.logical_pools is logical_runtime
    assert views.full.state.kv_state is runtime
    assert views.base.state.kv_state is runtime

    full_attention = next(
        binding
        for binding in runtime.full_bindings
        if isinstance(binding.module, _AttentionLayer)
    )
    base_attention = next(
        binding
        for binding in runtime.base_bindings
        if binding.physical_layer_name == full_attention.physical_layer_name
    )
    assert full_attention.module is not base_attention.module
    assert full_attention.raw_tensor.data_ptr() != base_attention.raw_tensor.data_ptr()
    assert full_attention.module.kv_cache.untyped_storage().data_ptr() == (
        full_attention.raw_tensor.untyped_storage().data_ptr()
    )
    assert base_attention.module.kv_cache.untyped_storage().data_ptr() == (
        base_attention.raw_tensor.untyped_storage().data_ptr()
    )

    full_mamba = next(
        binding
        for binding in runtime.full_bindings
        if isinstance(binding.module, _GDNLayer)
    )
    base_mamba = next(
        binding
        for binding in runtime.base_bindings
        if binding.physical_layer_name == full_mamba.physical_layer_name
    )
    assert full_mamba.module is not base_mamba.module
    assert all(
        state.untyped_storage().data_ptr()
        == full_mamba.raw_tensor.untyped_storage().data_ptr()
        for state in full_mamba.module.kv_cache
    )
    assert all(
        state.untyped_storage().data_ptr()
        == base_mamba.raw_tensor.untyped_storage().data_ptr()
        for state in base_mamba.module.kv_cache
    )
    assert full_mamba.raw_tensor.data_ptr() != base_mamba.raw_tensor.data_ptr()

    # TARGET allocations are deliberately not passed to any draft module.
    target_names = {
        allocation.global_layer_name
        for allocation in physical_runtime.plan.tensors
        if allocation.domain is AsymSpecCacheDomain.TARGET
    }
    assert not target_names.intersection(runtime.bindings)
    assert (
        bind_asymspec_draft_caches(
            views=views,
            physical_runtime=physical_runtime,
            global_plan=global_plan,
            logical_pools=logical_runtime,
            vllm_config=config,
        )
        is runtime
    )


def test_draft_cache_binder_rejects_a_second_mismatched_runtime():
    config, views, global_plan, physical_runtime, logical_runtime = (
        _build_bindable_draft_cache_runtime()
    )
    bind_asymspec_draft_caches(
        views=views,
        physical_runtime=physical_runtime,
        global_plan=global_plan,
        logical_pools=logical_runtime,
        vllm_config=config,
    )
    other_runtime = allocate_asymspec_physical_cache_tensors(
        plan=physical_runtime.plan, device=torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="already permanently bound"):
        bind_asymspec_draft_caches(
            views=views,
            physical_runtime=other_runtime,
            global_plan=global_plan,
            logical_pools=logical_runtime,
            vllm_config=config,
        )


class _MetadataOnlyBuilder:
    """Avoid backend-specific execution in allocation-free metadata tests."""

    def build(self, *, common_attn_metadata, **kwargs):
        return common_attn_metadata


def _bind_metadata_test_runtime(
    *,
    full_max_model_len: int = 3200,
    compressed_max_model_len: int = 1602,
):
    config, views, global_plan, physical_runtime, logical_runtime = (
        _build_bindable_draft_cache_runtime(
            full_max_model_len=full_max_model_len,
            compressed_max_model_len=compressed_max_model_len,
            attention_block_size=800,
        )
    )
    bindings = bind_asymspec_draft_caches(
        views=views,
        physical_runtime=physical_runtime,
        global_plan=global_plan,
        logical_pools=logical_runtime,
        vllm_config=config,
    )
    return config, views, physical_runtime, logical_runtime, bindings


def test_execution_metadata_keeps_full_and_base_resources_role_local(monkeypatch):
    monkeypatch.setattr(
        execution_metadata_module,
        "_new_metadata_builder",
        lambda **_: _MetadataOnlyBuilder(),
    )
    config, views, physical_runtime, logical_runtime, bindings = (
        _bind_metadata_test_runtime()
    )
    state = create_asymspec_request_state(
        request_id="metadata-role-local",
        compressed_prompt_len=1602,
        full_prompt_len=3200,
        augmentation_offset=1598,
        logical_pools=logical_runtime,
    )
    storage_before = {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in physical_runtime.raw_tensors.items()
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("metadata construction must not execute a draft model")

    monkeypatch.setattr(views.full.model, "forward", fail_if_called)
    monkeypatch.setattr(views.base.model, "forward", fail_if_called)

    full = build_asymspec_view_execution_metadata(
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
        role=AsymSpecViewRole.FULL,
        query_start=799,
        query_len=3,
        device=torch.device("cpu"),
    )
    base = build_asymspec_view_execution_metadata(
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
        role=AsymSpecViewRole.BASE,
        query_start=799,
        query_len=3,
        device=torch.device("cpu"),
    )

    assert set(full.groups) == {
        AsymSpecLogicalCacheGroup.FULL_ATTENTION,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_A,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_B,
    }
    assert set(base.groups) == {
        AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_A,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_B,
    }
    assert (
        full.groups[AsymSpecLogicalCacheGroup.FULL_ATTENTION].block_table.data_ptr()
        == state.block_tables.full_attention_table.block_table.gpu.data_ptr()
    )
    assert (
        base.groups[
            AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
        ].block_table.data_ptr()
        == state.block_tables.compressed_attention_table.block_table.gpu.data_ptr()
    )
    assert full.positions.tolist() == [799, 800, 801]
    assert base.positions.tolist() == [799, 800, 801]
    assert full.attention_block_ids.tolist() == [1, 2, 2]
    assert base.attention_block_ids.tolist() == [1, 2, 2]
    assert full.attention_slot_mapping.tolist() == [1599, 1600, 1601]
    assert base.attention_slot_mapping.tolist() == [1599, 1600, 1601]
    assert all(
        recurrent_page_ids(group.recurrent_slots) == (0, 1, 2, 3)
        for group in full.groups.values()
        if group.recurrent_slots is not None
    )
    assert all(
        recurrent_page_ids(group.recurrent_slots) == (0, 1, 2, 3)
        for group in base.groups.values()
        if group.recurrent_slots is not None
    )
    # Frozen compact GDN semantics anchor state metadata at the canonical
    # boundary, not at the end of a temporary query span.
    assert all(
        group.metadata.seq_lens.tolist() == [799]
        for group in full.groups.values()
        if group.recurrent_slots is not None
    )
    assert all(
        group.metadata.seq_lens.tolist() == [799]
        for group in base.groups.values()
        if group.recurrent_slots is not None
    )
    assert storage_before == {
        name: tensor.untyped_storage().data_ptr()
        for name, tensor in physical_runtime.raw_tensors.items()
    }
    state.release()


@pytest.mark.parametrize(
    ("query_start", "expected_blocks", "expected_slots"),
    [
        (799, [1, 2, 2], [1599, 1600, 1601]),
        (1599, [2, 3, 3], [2399, 2400, 2401]),
    ],
)
def test_execution_metadata_uses_native_attention_block_boundaries(
    monkeypatch, query_start, expected_blocks, expected_slots
):
    monkeypatch.setattr(
        execution_metadata_module,
        "_new_metadata_builder",
        lambda **_: _MetadataOnlyBuilder(),
    )
    config, views, _, logical_runtime, bindings = _bind_metadata_test_runtime()
    state = create_asymspec_request_state(
        request_id=f"metadata-boundary-{query_start}",
        compressed_prompt_len=1602,
        full_prompt_len=3200,
        augmentation_offset=1598,
        logical_pools=logical_runtime,
    )
    metadata = build_asymspec_view_execution_metadata(
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
        role=AsymSpecViewRole.BASE,
        query_start=query_start,
        query_len=3,
        device=torch.device("cpu"),
    )
    assert metadata.attention_block_ids.tolist() == expected_blocks
    assert metadata.attention_slot_mapping.tolist() == expected_slots
    state.release()


def test_execution_metadata_represents_deferred_base_catchup(monkeypatch):
    monkeypatch.setattr(
        execution_metadata_module,
        "_new_metadata_builder",
        lambda **_: _MetadataOnlyBuilder(),
    )
    config, views, _, logical_runtime, bindings = _bind_metadata_test_runtime(
        full_max_model_len=1024,
        compressed_max_model_len=128,
    )
    state = create_asymspec_request_state(
        request_id="metadata-base-lag",
        compressed_prompt_len=96,
        full_prompt_len=1024,
        augmentation_offset=928,
        logical_pools=logical_runtime,
    )
    state.advance_target([11, 12, 13, 14])
    assert state.base.canonical_len == 96
    assert state.base.observed_len == 100
    full_before = state.full.canonical_len
    metadata = build_asymspec_view_execution_metadata(
        request_state=state,
        views=views,
        cache_bindings=bindings,
        vllm_config=config,
        role=AsymSpecViewRole.BASE,
        query_start=96,
        query_len=4,
        canonical_end=96,
        device=torch.device("cpu"),
    )
    assert metadata.positions.tolist() == [96, 97, 98, 99]
    assert metadata.canonical_end == 96
    assert metadata.attention_block_ids.tolist() == [1, 1, 1, 1]
    assert state.full.canonical_len == full_before
    assert state.base.pending_token_ids == [11, 12, 13, 14]
    state.release()
