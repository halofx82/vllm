# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-weight draft model trees and logical AsymSpec views.

This module deliberately contains no cache allocation or proposal logic. It
only establishes the ownership boundary required by later AsymSpec stages:
one checkpoint-backed FULL draft tree plus a structurally independent BASE
tree whose parameters and persistent buffers alias FULL storage.
"""

from dataclasses import dataclass, field
from enum import Enum
from copy import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

from .hybrid import AsymSpecHybridStateSpec, describe_qwen3_5_hybrid_state

if TYPE_CHECKING:
    from .cache_plan import AsymSpecCachePlan


class AsymSpecViewRole(str, Enum):
    """Semantic identity of an AsymSpec logical draft view."""

    FULL = "full"
    BASE = "base"


@dataclass
class AsymSpecViewState:
    """Future per-view state attachment point.

    Each view owns a distinct instance. KV, recurrent, and position state are
    intentionally not allocated or populated until the corresponding runtime
    stages are introduced.
    """

    kv_state: object | None = None
    recurrent_state: object | None = None
    position_state: object | None = None
    hybrid_spec: AsymSpecHybridStateSpec | None = None
    cache_plan: "AsymSpecCachePlan | None" = None


@dataclass
class AsymSpecView:
    """A logical AsymSpec view backed by its own draft module tree."""

    role: AsymSpecViewRole
    model: nn.Module
    state: AsymSpecViewState = field(default_factory=AsymSpecViewState)


@dataclass(frozen=True)
class AsymSpecDraftLoadMemory:
    """Per-worker memory checkpoints for the shared-weight construction."""

    before_full_load: int
    after_full_load: int
    after_base_tree: int
    after_state_aliasing: int


def _module_and_leaf(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = qualified_name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), leaf


def share_model_state(full: nn.Module, base: nn.Module) -> tuple[int, int]:
    """Alias BASE parameters/buffers to FULL without sharing module objects.

    ``base`` is created on ``meta`` during real loading, so assigning FULL's
    tensors is its only model-state materialization. Runtime/cache attributes
    are not parameters or persistent buffers and deliberately remain local to
    each module tree.
    """
    full_params = dict(full.named_parameters(remove_duplicate=False))
    base_params = dict(base.named_parameters(remove_duplicate=False))
    if full_params.keys() != base_params.keys():
        mismatch = sorted(full_params.keys() ^ base_params.keys())[:16]
        raise RuntimeError("AsymSpec FULL/BASE parameter layouts differ: "
                           f"{mismatch!r}")

    parameter_bytes = 0
    seen_params: set[int] = set()
    for name, full_param in full_params.items():
        parent, leaf = _module_and_leaf(base, name)
        parent._parameters[leaf] = full_param
        if id(full_param) not in seen_params:
            parameter_bytes += full_param.numel() * full_param.element_size()
            seen_params.add(id(full_param))

    full_buffers = dict(full.named_buffers(remove_duplicate=False))
    base_buffers = dict(base.named_buffers(remove_duplicate=False))
    if full_buffers.keys() != base_buffers.keys():
        mismatch = sorted(full_buffers.keys() ^ base_buffers.keys())[:16]
        raise RuntimeError("AsymSpec FULL/BASE buffer layouts differ: "
                           f"{mismatch!r}")

    buffer_bytes = 0
    seen_buffers: set[int] = set()
    for name, full_buffer in full_buffers.items():
        parent, leaf = _module_and_leaf(base, name)
        parent._buffers[leaf] = full_buffer
        if id(full_buffer) not in seen_buffers:
            buffer_bytes += full_buffer.numel() * full_buffer.element_size()
            seen_buffers.add(id(full_buffer))

    rebound_params = dict(base.named_parameters(remove_duplicate=False))
    rebound_buffers = dict(base.named_buffers(remove_duplicate=False))
    for name, full_param in full_params.items():
        if rebound_params[name] is not full_param:
            raise RuntimeError(f"AsymSpec parameter sharing failed: {name}")
    for name, full_buffer in full_buffers.items():
        if rebound_buffers[name] is not full_buffer:
            raise RuntimeError(f"AsymSpec buffer sharing failed: {name}")
    return parameter_bytes, buffer_bytes


class AsymSpecDraftViews:
    """Own shared-weight FULL/BASE trees with independent runtime identity."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "asymspec":
            raise ValueError("AsymSpecDraftViews requires method='asymspec'.")
        if speculative_config.draft_model_config is None:
            raise ValueError("AsymSpec requires an initialized draft model config.")

        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = speculative_config
        # ``model`` remains the FULL tree for existing draft-model accessors.
        self.model: nn.Module | None = None
        self.full: AsymSpecView | None = None
        self.base: AsymSpecView | None = None
        self.load_memory: AsymSpecDraftLoadMemory | None = None
        # Bound only by the later AsymSpec-local cache-binding stage. This is
        # deliberately not a GPUModelRunner ``kv_caches`` entry.
        self.draft_cache_bindings: object | None = None

    def _allocated_memory(self) -> int:
        return (
            torch.cuda.memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Create the normal external-draft loading config once."""
        spec = self.speculative_config
        # TARGET uses aligned Mamba state so its verifier can select the
        # accepted S/A/B checkpoint after one K=2 forward.  FULL and BASE
        # intentionally retain the independently validated compact state
        # representation; only the mode differs, while backend-normalized
        # block geometry remains shared.
        # ``CacheConfig`` is a dataclass in production, while lightweight
        # structural fixtures may provide a namespace with the same field.
        # A shallow copy is sufficient: only this scalar mode is role-local.
        draft_cache_config = copy(self.vllm_config.cache_config)
        draft_cache_config.mamba_cache_mode = "none"
        # Build the draft's standalone VllmConfig without recursively applying
        # the TARGET-only AsymSpec align policy. Validation of compact Mamba
        # mode requires its conventional max-length block size; the later
        # cache-plan construction still observes the worker-normalized
        # geometry through a fresh config view.
        if isinstance(self.vllm_config, VllmConfig):
            draft_cache_config.mamba_block_size = getattr(
                self.speculative_config.draft_model_config,
                "max_model_len",
                self.vllm_config.model_config.max_model_len,
            )
        draft_vllm_config = replace(
            self.vllm_config,
            quant_config=None,
            cache_config=draft_cache_config,
            # Real VllmConfig construction must avoid recursively applying the
            # TARGET-only align policy. Synthetic fixtures retain their
            # speculative metadata because their mock Mamba specs use K.
            speculative_config=(
                None
                if isinstance(self.vllm_config, VllmConfig)
                else self.speculative_config
            ),
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )
        if isinstance(self.vllm_config, VllmConfig):
            # Construction above deliberately omitted AsymSpec so the target
            # align policy cannot affect the draft tree. Restore its K=2
            # metadata after validation; MambaSpec uses it to reserve the
            # compact committed/speculative slots. Cache-plan callers run
            # after backend normalization and must also see its resolved
            # Mamba block geometry.
            draft_vllm_config.speculative_config = self.speculative_config
            draft_vllm_config.cache_config.mamba_block_size = (
                self.vllm_config.cache_config.mamba_block_size
            )
        return draft_vllm_config

    def load_model(self) -> None:
        """Load FULL once and construct a shared-storage BASE module tree."""
        if self.model is not None:
            raise RuntimeError("AsymSpec draft model has already been loaded.")

        draft_vllm_config = self._create_draft_vllm_config()
        before_full_load = self._allocated_memory()
        with set_model_tag("asymspec_draft"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
                prefix="asymspec_draft",
            )
        after_full_load = self._allocated_memory()
        with set_model_tag("asymspec_base"):
            with set_default_torch_dtype(
                    getattr(draft_vllm_config.model_config, "dtype", torch.bfloat16)):
                with torch.device("meta"):
                    base_model = initialize_model(
                        vllm_config=draft_vllm_config,
                        model_config=self.speculative_config.draft_model_config,
                        prefix="asymspec_base",
                    )
        after_base_tree = self._allocated_memory()
        self.bind_models(model, base_model)
        self.load_memory = AsymSpecDraftLoadMemory(
            before_full_load=before_full_load,
            after_full_load=after_full_load,
            after_base_tree=after_base_tree,
            after_state_aliasing=self._allocated_memory(),
        )

    def bind_model(self, model: nn.Module) -> None:
        """Test-only convenience constructor for a structural BASE clone.

        Real loading uses :meth:`bind_models` with a meta-constructed BASE
        tree; this helper keeps synthetic unit fixtures concise.
        """
        import copy

        self.bind_models(model, copy.deepcopy(model))

    def bind_models(self, full_model: nn.Module, base_model: nn.Module) -> None:
        """Bind distinct FULL/BASE trees after aliasing their model state."""
        if self.model is not None:
            raise RuntimeError("AsymSpec draft model has already been bound.")

        share_model_state(full_model, base_model)
        self.model = full_model
        self.full = AsymSpecView(AsymSpecViewRole.FULL, full_model)
        self.base = AsymSpecView(AsymSpecViewRole.BASE, base_model)
        assert self.full is not self.base
        assert self.full.model is self.model
        assert self.full.model is not self.base.model

    def owns_physical_module(self, module: nn.Module) -> bool:
        """Return whether ``module`` belongs to either draft module tree.

        The compilation static-forward context intentionally retains both the
        target and draft modules.  AsymSpec uses this identity boundary when
        collecting TARGET cache specs: draft modules belong exclusively to the
        logical FULL and BASE cache plans, never to TARGET.
        """
        if self.model is None or self.base is None:
            raise RuntimeError("AsymSpec draft model is unavailable before model load.")
        return any(
            candidate is module
            for model in (self.model, self.base.model)
            for candidate in model.modules()
        )

    def initialize_state_specs(self) -> None:
        """Attach independent, allocation-free hybrid state descriptions."""
        if self.model is None or self.full is None or self.base is None:
            raise RuntimeError(
                "AsymSpec draft views are unavailable before model load."
            )

        # Build descriptors from separate module trees. Weight storage is
        # shared; all future KV/GDN/position state remains role-local.
        self.full.state.hybrid_spec = describe_qwen3_5_hybrid_state(
            self.full.model, self.speculative_config.num_speculative_tokens
        )
        self.base.state.hybrid_spec = describe_qwen3_5_hybrid_state(
            self.base.model, self.speculative_config.num_speculative_tokens
        )
        assert self.full.state.hybrid_spec is not self.base.state.hybrid_spec

    def initialize_cache_plans(self) -> None:
        """Attach allocation-free native KV-cache plans to both views."""
        if self.model is None or self.full is None or self.base is None:
            raise RuntimeError(
                "AsymSpec draft views are unavailable before model load."
            )
        if self.full.state.hybrid_spec is None or self.base.state.hybrid_spec is None:
            raise RuntimeError(
                "AsymSpec hybrid state specs must be initialized before cache plans."
            )

        # Local import avoids a circular dependency: cache plans use the role
        # enum defined in this module.
        from .cache_plan import build_asymspec_cache_plan

        compressed_max_model_len = (
            self.speculative_config.asymspec_compressed_max_model_len
        )
        if compressed_max_model_len is None:
            raise ValueError("AsymSpec requires a compressed max model length.")

        draft_vllm_config = self._create_draft_vllm_config()
        self.full.state.cache_plan = build_asymspec_cache_plan(
            role=AsymSpecViewRole.FULL,
            max_model_len=self.vllm_config.model_config.max_model_len,
            model=self.full.model,
            hybrid_spec=self.full.state.hybrid_spec,
            draft_vllm_config=draft_vllm_config,
        )
        self.base.state.cache_plan = build_asymspec_cache_plan(
            role=AsymSpecViewRole.BASE,
            max_model_len=compressed_max_model_len,
            model=self.base.model,
            hybrid_spec=self.base.state.hybrid_spec,
            draft_vllm_config=draft_vllm_config,
        )
        assert self.full.state.cache_plan is not self.base.state.cache_plan

    def view(self, role: AsymSpecViewRole) -> AsymSpecView:
        """Return a logical view by semantic role, never by cache-group ID."""
        if self.full is None or self.base is None:
            raise RuntimeError(
                "AsymSpec draft views are unavailable before model load."
            )
        return self.full if role is AsymSpecViewRole.FULL else self.base
