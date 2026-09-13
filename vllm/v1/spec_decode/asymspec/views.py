# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared physical draft model and logical AsymSpec views.

This module deliberately contains no cache allocation or proposal logic. It
only establishes the ownership boundary required by later AsymSpec stages:
one loaded draft model, with independent FULL and BASE runtime identities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model

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
    """A logical view over the one physical AsymSpec draft model."""

    role: AsymSpecViewRole
    model: nn.Module
    state: AsymSpecViewState = field(default_factory=AsymSpecViewState)


class AsymSpecDraftViews:
    """Own one physical draft model and expose FULL and BASE logical views."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "asymspec":
            raise ValueError("AsymSpecDraftViews requires method='asymspec'.")
        if speculative_config.draft_model_config is None:
            raise ValueError("AsymSpec requires an initialized draft model config.")

        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = speculative_config
        self.model: nn.Module | None = None
        self.full: AsymSpecView | None = None
        self.base: AsymSpecView | None = None

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Create the normal external-draft loading config once."""
        spec = self.speculative_config
        return replace(
            self.vllm_config,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    def load_model(self) -> None:
        """Load the physical draft weights once, then bind both logical views."""
        if self.model is not None:
            raise RuntimeError("AsymSpec draft model has already been loaded.")

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("asymspec_draft"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
                prefix="asymspec_draft",
            )
        self.bind_model(model)

    def bind_model(self, model: nn.Module) -> None:
        """Bind one physical model to distinct FULL and BASE view identities."""
        if self.model is not None:
            raise RuntimeError("AsymSpec draft model has already been bound.")

        self.model = model
        self.full = AsymSpecView(AsymSpecViewRole.FULL, model)
        self.base = AsymSpecView(AsymSpecViewRole.BASE, model)
        assert self.full is not self.base
        assert self.full.model is self.base.model is self.model

    def initialize_state_specs(self) -> None:
        """Attach independent, allocation-free hybrid state descriptions."""
        if self.model is None or self.full is None or self.base is None:
            raise RuntimeError(
                "AsymSpec draft views are unavailable before model load."
            )

        # Build separate descriptor objects: the physical layers remain shared,
        # while each logical role owns its future KV/GDN/position state.
        self.full.state.hybrid_spec = describe_qwen3_5_hybrid_state(
            self.model, self.speculative_config.num_speculative_tokens
        )
        self.base.state.hybrid_spec = describe_qwen3_5_hybrid_state(
            self.model, self.speculative_config.num_speculative_tokens
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
            model=self.model,
            hybrid_spec=self.full.state.hybrid_spec,
            draft_vllm_config=draft_vllm_config,
        )
        self.base.state.cache_plan = build_asymspec_cache_plan(
            role=AsymSpecViewRole.BASE,
            max_model_len=compressed_max_model_len,
            model=self.model,
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
