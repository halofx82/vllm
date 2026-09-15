# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent, role-local canonical execution for AsymSpec draft views.

This is an ordinary autoregressive driver, not a speculative proposer.  It
owns no scheduler state and advances only one already-bound FULL or BASE view.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig

from .cache_binding import AsymSpecDraftCacheBindingRuntime
from .draft_forward import (
    AsymSpecDraftForwardResult,
    execute_asymspec_draft_forward,
    initialize_fresh_asymspec_view_state,
)
from .execution_metadata import build_asymspec_view_execution_metadata
from .logical_cache import AsymSpecLogicalCacheGroup
from .request_state import AsymSpecRequestState
from .views import AsymSpecDraftViews, AsymSpecViewRole


@dataclass
class AsymSpecCanonicalExecutionCounters:
    """Accounting for one role-local persistent canonical stream."""

    prefill_forward_calls: int = 0
    incremental_forward_calls: int = 0
    prefill_tokens_processed: int = 0
    incremental_tokens_processed: int = 0
    catch_up_forward_calls: int = 0
    catch_up_tokens_processed: int = 0
    replayed_historical_tokens: int = 0

    @property
    def total_forward_calls(self) -> int:
        return (
            self.prefill_forward_calls
            + self.incremental_forward_calls
            + self.catch_up_forward_calls
        )


class AsymSpecCanonicalDraftDriver:
    """Drive fresh prefill and committed AR tokens for exactly one draft view.

    ``request_state`` must be created with
    ``canonical_prompt_processed=False``.  The input prompt capacity is
    reserved in its local tables, while canonical progress remains zero until
    a successful prefill returns.
    """

    def __init__(
        self,
        *,
        role: AsymSpecViewRole,
        request_state: AsymSpecRequestState,
        views: AsymSpecDraftViews,
        cache_bindings: AsymSpecDraftCacheBindingRuntime,
        vllm_config: VllmConfig,
        device: torch.device | None = None,
    ) -> None:
        self.role = role
        self.request_state = request_state
        self.views = views
        self.cache_bindings = cache_bindings
        self.vllm_config = vllm_config
        if device is None:
            model = views.view(role).model
            tensor = next(iter(model.parameters()), None)
            if tensor is None:
                tensor = next(iter(model.buffers()), None)
            device = tensor.device if tensor is not None else torch.device("cpu")
        self.device = device
        self.counters = AsymSpecCanonicalExecutionCounters()
        self._prefilled = False
        # The last successful forward predicts the next canonical token.  It
        # is retained locally so a greedy proposer can use the prompt-boundary
        # or committed-token logits without replaying a token.
        self._last_result: AsymSpecDraftForwardResult | None = None

    @property
    def canonical_len(self) -> int:
        return (
            self.request_state.full.canonical_len
            if self.role is AsymSpecViewRole.FULL
            else self.request_state.base.canonical_len
        )

    @property
    def prompt_len(self) -> int:
        return (
            self.request_state.full.prompt_len
            if self.role is AsymSpecViewRole.FULL
            else self.request_state.base.prompt_len
        )

    @property
    def last_result(self) -> AsymSpecDraftForwardResult:
        """The logits predicting the next token at the canonical boundary."""
        if self._last_result is None:
            raise RuntimeError("AsymSpec canonical view has no current logits.")
        return self._last_result

    def _set_last_result(self, result: AsymSpecDraftForwardResult) -> None:
        """Record a result only when it represents canonical state."""
        self._last_result = result

    def _ensure_attention_capacity(self, token_count: int) -> None:
        group = (
            AsymSpecLogicalCacheGroup.FULL_ATTENTION
            if self.role is AsymSpecViewRole.FULL
            else AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
        )
        self.request_state.block_tables.ensure_attention_tokens(group, token_count)

    def _advance(self, num_tokens: int) -> None:
        if self.role is AsymSpecViewRole.FULL:
            self.request_state.advance_full(num_tokens)
        else:
            self.request_state.advance_base_committed(num_tokens)

    def _forward(
        self,
        input_ids: torch.Tensor,
        *,
        query_start: int,
        canonical_end: int | None = None,
        allow_uncommitted_start: bool = False,
    ) -> AsymSpecDraftForwardResult:
        metadata = build_asymspec_view_execution_metadata(
            request_state=self.request_state,
            views=self.views,
            cache_bindings=self.cache_bindings,
            vllm_config=self.vllm_config,
            role=self.role,
            query_start=query_start,
            canonical_end=(query_start if canonical_end is None else canonical_end),
            query_len=input_ids.numel(),
            allow_uncommitted_end=True,
            allow_uncommitted_start=allow_uncommitted_start,
            device=self.device,
        )
        return execute_asymspec_draft_forward(
            role=self.role,
            input_ids=input_ids,
            metadata=metadata,
            views=self.views,
            cache_bindings=self.cache_bindings,
        )

    def prefill(self, prompt_token_ids: torch.Tensor) -> AsymSpecDraftForwardResult:
        """Run one fresh prompt prefill and commit it only after success."""
        if self._prefilled:
            raise RuntimeError("AsymSpec canonical view is already prefilled.")
        if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() <= 0:
            raise ValueError("AsymSpec canonical prefill requires packed tokens.")
        if prompt_token_ids.device != self.device:
            raise ValueError("AsymSpec canonical prompt is on the wrong device.")
        if self.canonical_len != 0:
            raise RuntimeError(
                "AsymSpec canonical prefill requires an unprocessed request state."
            )
        if prompt_token_ids.numel() != self.prompt_len:
            raise ValueError("AsymSpec canonical prompt length does not match state.")

        self._ensure_attention_capacity(prompt_token_ids.numel())
        initialize_fresh_asymspec_view_state(
            role=self.role, cache_bindings=self.cache_bindings
        )
        result = self._forward(prompt_token_ids, query_start=0)
        self._advance(prompt_token_ids.numel())
        self._set_last_result(result)
        self._prefilled = True
        self.counters.prefill_forward_calls += 1
        self.counters.prefill_tokens_processed += prompt_token_ids.numel()
        return result

    def commit_token(self, token_id: int | torch.Tensor) -> AsymSpecDraftForwardResult:
        """Consume one already-authoritative token without replaying history."""
        if not self._prefilled:
            raise RuntimeError("AsymSpec canonical view must be prefilled first.")
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() != 1:
                raise ValueError("AsymSpec committed step needs exactly one token.")
            token_ids = token_id.reshape(1).to(device=self.device, dtype=torch.int32)
        else:
            token_ids = torch.tensor([token_id], device=self.device, dtype=torch.int32)
        start = self.canonical_len
        self._ensure_attention_capacity(start + 1)
        result = self._forward(token_ids, query_start=start)
        self._advance(1)
        self._set_last_result(result)
        self.counters.incremental_forward_calls += 1
        self.counters.incremental_tokens_processed += 1
        return result

    def begin_deferred_base(self) -> None:
        """Start observing TARGET commits while this already-prefilled BASE lags.

        The frozen deferred path begins with BASE fully prefetched through the
        compressed prompt.  From then on, TARGET advances the observed
        compressed boundary while BASE retains its last committed boundary.
        """
        if self.role is not AsymSpecViewRole.BASE:
            raise RuntimeError("Only the BASE canonical driver can defer BASE.")
        if not self._prefilled:
            raise RuntimeError("BASE must be prefetched before it can defer.")
        self.request_state.begin_deferred_base_observation()

    def observe_committed_token(self, token_id: int | torch.Tensor) -> None:
        """Record one authoritative compressed token without executing BASE."""
        if self.role is not AsymSpecViewRole.BASE:
            raise RuntimeError(
                "Only the BASE canonical driver observes deferred tokens."
            )
        if not self._prefilled:
            raise RuntimeError("BASE must be prefetched before deferred observation.")
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() != 1:
                raise ValueError(
                    "AsymSpec deferred observation needs exactly one token."
                )
            token_id = int(token_id.item())
        self.request_state.advance_target([int(token_id)])

    def catch_up_base(self) -> AsymSpecDraftForwardResult | None:
        """Run the frozen packed pending range and commit it transactionally.

        No state metadata changes before the packed BASE forward returns.  On
        a forward failure the canonical boundary, observed boundary, and
        pending queue therefore remain intact for the caller to inspect or
        retry.
        """
        if self.role is not AsymSpecViewRole.BASE:
            raise RuntimeError("Only the BASE canonical driver can catch up BASE.")
        if not self._prefilled:
            raise RuntimeError("BASE must be prefetched before catch-up.")
        pending = tuple(self.request_state.base.pending_token_ids)
        if not pending:
            return None
        start = self.request_state.base.canonical_len
        self._ensure_attention_capacity(start + len(pending))
        token_ids = torch.tensor(pending, device=self.device, dtype=torch.int32)
        result = self._forward(token_ids, query_start=start)
        committed = self.request_state.catch_up_base()
        if committed != pending:
            raise AssertionError("Deferred BASE queue changed during catch-up.")
        self.counters.catch_up_forward_calls += 1
        self.counters.catch_up_tokens_processed += len(pending)
        self._set_last_result(result)
        return result
