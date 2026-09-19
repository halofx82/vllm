# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsymSpec-local request coordinates and unbound block-table metadata.

This module deliberately has no dependency on the V1 scheduler or ``Request``.
It models the three canonical streams established by the validated runtime:
the compressed verifier stream, the independently deferred compressed BASE
stream, and the augmented long-context FULL stream.  The tables are local
metadata over AsymSpec's eight already-instantiated ``BlockPool`` objects;
they are not attached to ``InputBatch`` or used for a model forward yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.block_table import (
    BlockTable,
    MultiGroupBlockTable,
    SlotMappingMode,
)

from .logical_cache import (
    AsymSpecLogicalBlockPoolRuntime,
    AsymSpecLogicalCacheGroup,
)

_ATTENTION_GROUPS = frozenset(
    (
        AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
        AsymSpecLogicalCacheGroup.FULL_ATTENTION,
    )
)
_RECURRENT_GROUPS = tuple(
    group for group in AsymSpecLogicalCacheGroup if group not in _ATTENTION_GROUPS
)


@dataclass
class AsymSpecCompressedCoordinates:
    """Canonical progress in the compressed TARGET coordinate system."""

    prompt_len: int
    canonical_len: int


@dataclass
class AsymSpecBaseCoordinates:
    """Canonical/observed progress for the deferred compressed BASE view."""

    prompt_len: int
    canonical_len: int
    observed_len: int
    pending_token_ids: list[int] = field(default_factory=list)


@dataclass
class AsymSpecFullCoordinates:
    """Canonical progress in the augmented long-context FULL coordinate space."""

    prompt_len: int
    canonical_len: int
    augmentation_offset: int

    def position_for_compressed(self, compressed_position: int) -> int:
        """Map a compressed-stream position into the augmented FULL stream."""
        if compressed_position < 0:
            raise ValueError("Compressed position must be non-negative.")
        return self.augmentation_offset + compressed_position


@dataclass(frozen=True)
class AsymSpecRecurrentBlockSlots:
    """Compact native Mamba/GDN state slots for one logical recurrent pool."""

    null_block_id: int
    committed_block_id: int
    speculative_block_ids: tuple[int, int]


@dataclass
class AsymSpecRequestBlockTables:
    """One request-local row for each semantic AsymSpec logical group."""

    multi_group_table: MultiGroupBlockTable
    table_by_group: dict[AsymSpecLogicalCacheGroup, BlockTable]
    pool_by_group: dict[AsymSpecLogicalCacheGroup, BlockPool]
    allocated_blocks_by_group: dict[
        AsymSpecLogicalCacheGroup, list[KVCacheBlock]
    ]
    recurrent_slots: dict[AsymSpecLogicalCacheGroup, AsymSpecRecurrentBlockSlots]
    _group_order: tuple[AsymSpecLogicalCacheGroup, ...]
    _released: bool = False

    @property
    def compressed_attention_table(self) -> BlockTable:
        return self.table_by_group[AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION]

    @property
    def target_attention_table(self) -> BlockTable:
        """TARGET and BASE intentionally consume the same compressed table."""
        return self.compressed_attention_table

    @property
    def base_attention_table(self) -> BlockTable:
        """TARGET and BASE intentionally consume the same compressed table."""
        return self.compressed_attention_table

    @property
    def full_attention_table(self) -> BlockTable:
        return self.table_by_group[AsymSpecLogicalCacheGroup.FULL_ATTENTION]

    def attention_block_ids(
        self, semantic_group: AsymSpecLogicalCacheGroup
    ) -> tuple[int, ...]:
        if semantic_group not in _ATTENTION_GROUPS:
            raise ValueError(f"{semantic_group.value} is not an attention group.")
        return tuple(
            block.block_id for block in self.allocated_blocks_by_group[semantic_group]
        )

    def ensure_attention_tokens(
        self, semantic_group: AsymSpecLogicalCacheGroup, token_count: int
    ) -> None:
        """Extend one attention coordinate table through ``token_count``.

        This only reserves logical IDs from the pre-existing pool and updates
        this request's CPU/GPU table buffers.  It does not touch model cache
        tensors or any generic request/scheduler state.
        """
        if self._released:
            raise RuntimeError("Cannot extend released AsymSpec request tables.")
        if semantic_group not in _ATTENTION_GROUPS:
            raise ValueError(f"{semantic_group.value} is not an attention group.")
        if token_count < 0:
            raise ValueError("AsymSpec token count must be non-negative.")
        table = self.table_by_group[semantic_group]
        required_blocks = cdiv(token_count, table.kv_cache_block_size)
        blocks = self.allocated_blocks_by_group[semantic_group]
        additional = required_blocks - len(blocks)
        # Request creation reserves the entire prompt capacity before a
        # chunked prefill begins.  Individual prefill chunks therefore ask
        # for a shorter prefix of an already append-only table.  That is a
        # capacity check, not an attempt to shrink the table.
        if additional > 0:
            blocks.extend(self.pool_by_group[semantic_group].get_new_blocks(additional))
            table.add_row([block.block_id for block in blocks], row_idx=0)

    def release(self) -> None:
        """Return this request's non-null logical blocks to their pools."""
        if self._released:
            return
        for semantic_group, blocks in self.allocated_blocks_by_group.items():
            self.pool_by_group[semantic_group].free_blocks(blocks)
        self.multi_group_table.clear_row(0)
        self._released = True


@dataclass
class AsymSpecRequestState:
    """AsymSpec-owned canonical state for one future execution request.

    ``TARGET`` and ``BASE`` share compressed token coordinates, but BASE can
    deliberately remain behind TARGET until a later catch-up forward.  FULL
    keeps the augmented long-context coordinate system and never derives its
    position from a generic request's ``num_computed_tokens``.
    """

    request_id: str
    compressed: AsymSpecCompressedCoordinates
    base: AsymSpecBaseCoordinates
    full: AsymSpecFullCoordinates
    block_tables: AsymSpecRequestBlockTables

    def begin_deferred_base_observation(self) -> None:
        """Align the observed TARGET boundary with an already-prefilled BASE.

        The canonical driver intentionally permits an isolated BASE prefill
        before TARGET execution exists.  This one-time transition establishes
        the frozen deferred-BASE invariant: prompt tokens are already
        canonical in both compressed views and the pending queue starts empty.
        """
        if self.compressed.canonical_len != 0:
            raise RuntimeError("Deferred BASE observation was already initialized.")
        if self.base.pending_token_ids:
            raise AssertionError("Deferred BASE cannot start with pending tokens.")
        if self.base.observed_len != self.base.canonical_len:
            raise AssertionError(
                "BASE observed boundary must equal its canonical boundary."
            )
        self.compressed.canonical_len = self.base.canonical_len

    def advance_full(self, num_tokens: int) -> range:
        if num_tokens < 0:
            raise ValueError("FULL advancement must be non-negative.")
        start = self.full.canonical_len
        self.full.canonical_len += num_tokens
        self.block_tables.ensure_attention_tokens(
            AsymSpecLogicalCacheGroup.FULL_ATTENTION, self.full.canonical_len
        )
        return range(start, self.full.canonical_len)

    def advance_target(self, committed_token_ids: Sequence[int]) -> range:
        """Advance TARGET and record, but do not execute, deferred BASE work."""
        token_ids = [int(token_id) for token_id in committed_token_ids]
        start = self.compressed.canonical_len
        self.compressed.canonical_len += len(token_ids)
        self.base.observed_len = self.compressed.canonical_len
        self.base.pending_token_ids.extend(token_ids)
        self.block_tables.ensure_attention_tokens(
            AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
            self.compressed.canonical_len,
        )
        return range(start, self.compressed.canonical_len)

    def catch_up_base(self) -> tuple[int, ...]:
        """Commit deferred BASE metadata through the observed TARGET prefix."""
        if self.base.canonical_len > self.compressed.canonical_len:
            raise AssertionError("BASE cannot lead the compressed TARGET stream.")
        pending = tuple(self.base.pending_token_ids)
        expected = self.compressed.canonical_len - self.base.canonical_len
        if len(pending) != expected:
            raise AssertionError(
                "Deferred BASE pending IDs must exactly cover its canonical lag."
            )
        self.base.canonical_len = self.compressed.canonical_len
        self.base.pending_token_ids.clear()
        return pending

    def advance_base_committed(self, num_tokens: int) -> range:
        """Advance BASE directly for ordinary non-deferred AR execution.

        This is deliberately separate from :meth:`catch_up_base`: the
        canonical driver exercises BASE independently before TARGET/deferred
        execution is introduced.  TARGET's compressed canonical coordinate is
        therefore left untouched, while ``observed_len`` records that BASE has
        a locally executable prefix of this length.
        """
        if num_tokens < 0:
            raise ValueError("BASE advancement must be non-negative.")
        start = self.base.canonical_len
        self.base.canonical_len += num_tokens
        self.base.observed_len = max(self.base.observed_len, self.base.canonical_len)
        self.block_tables.ensure_attention_tokens(
            AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
            self.base.canonical_len,
        )
        return range(start, self.base.canonical_len)

    def release(self) -> None:
        self.block_tables.release()


def _table_slot_mapping_mode(group) -> SlotMappingMode:
    return (
        SlotMappingMode.NONE
        if isinstance(group.kv_cache_spec, MambaSpec)
        else SlotMappingMode.TOKEN_TO_KV_SLOT
    )


def _allocate_recurrent_slots(
    pool: BlockPool,
) -> tuple[list[KVCacheBlock], AsymSpecRecurrentBlockSlots]:
    blocks = pool.get_new_blocks(3)
    if len(blocks) != 3:
        raise AssertionError("Compact AsymSpec recurrent pools require three slots.")
    return blocks, AsymSpecRecurrentBlockSlots(
        null_block_id=pool.null_block.block_id,
        committed_block_id=blocks[0].block_id,
        speculative_block_ids=(blocks[1].block_id, blocks[2].block_id),
    )


def create_asymspec_request_state(
    *,
    request_id: str,
    compressed_prompt_len: int,
    full_prompt_len: int,
    logical_pools: AsymSpecLogicalBlockPoolRuntime,
    augmentation_offset: int | None = None,
    canonical_prompt_processed: bool = True,
    device: torch.device | None = None,
) -> AsymSpecRequestState:
    """Create pure AsymSpec request metadata and allocate its logical rows.

    ``augmentation_offset`` is the long-prefix displacement of compressed
    positions in the FULL prompt.  The validated augmented prompt convention
    is ``full_prompt_len == compressed_prompt_len + augmentation_offset``.
    """
    if not request_id:
        raise ValueError("AsymSpec request_id must be non-empty.")
    if compressed_prompt_len < 0 or full_prompt_len < 0:
        raise ValueError("AsymSpec prompt lengths must be non-negative.")
    if augmentation_offset is None:
        augmentation_offset = full_prompt_len - compressed_prompt_len
    if augmentation_offset < 0 or full_prompt_len != (
        compressed_prompt_len + augmentation_offset
    ):
        raise ValueError(
            "FULL prompt length must equal compressed prompt length plus its "
            "non-negative augmentation offset."
        )
    if device is None:
        device = torch.device("cpu")

    group_order = tuple(group.semantic_group for group in logical_pools.plan.groups)
    groups = {group.semantic_group: group for group in logical_pools.plan.groups}
    multi_group_table = MultiGroupBlockTable(
        max_num_reqs=1,
        max_num_batched_tokens=max(full_prompt_len, compressed_prompt_len, 1),
        pin_memory=False,
        device=device,
        block_sizes=[groups[key].block_size for key in group_order],
        kernel_block_sizes=[groups[key].block_size for key in group_order],
        max_num_blocks=[groups[key].num_blocks for key in group_order],
        slot_mapping_modes=[
            _table_slot_mapping_mode(groups[key]) for key in group_order
        ],
    )
    tables = dict(zip(group_order, multi_group_table.block_tables, strict=True))
    allocated: dict[AsymSpecLogicalCacheGroup, list[KVCacheBlock]] = {
        group: [] for group in group_order
    }
    recurrent_slots: dict[AsymSpecLogicalCacheGroup, AsymSpecRecurrentBlockSlots] = {}
    try:
        for semantic_group in _RECURRENT_GROUPS:
            blocks, slots = _allocate_recurrent_slots(
                logical_pools.pools[semantic_group]
            )
            allocated[semantic_group] = blocks
            recurrent_slots[semantic_group] = slots

        runtime = AsymSpecRequestBlockTables(
            multi_group_table=multi_group_table,
            table_by_group=tables,
            pool_by_group=dict(logical_pools.pools),
            allocated_blocks_by_group=allocated,
            recurrent_slots=recurrent_slots,
            _group_order=group_order,
        )
        runtime.ensure_attention_tokens(
            AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION, compressed_prompt_len
        )
        runtime.ensure_attention_tokens(
            AsymSpecLogicalCacheGroup.FULL_ATTENTION, full_prompt_len
        )
        recurrent_ids = {
            group: [
                recurrent_slots[group].committed_block_id,
                *recurrent_slots[group].speculative_block_ids,
            ]
            for group in _RECURRENT_GROUPS
        }
        multi_group_table.add_row(
            tuple(
                [
                    block.block_id for block in allocated[group]
                ]
                if group in _ATTENTION_GROUPS
                else recurrent_ids[group]
                for group in group_order
            ),
            row_idx=0,
        )
    except Exception:
        for semantic_group, blocks in allocated.items():
            if blocks:
                logical_pools.pools[semantic_group].free_blocks(blocks)
        raise

    initial_compressed_len = compressed_prompt_len if canonical_prompt_processed else 0
    initial_full_len = full_prompt_len if canonical_prompt_processed else 0
    return AsymSpecRequestState(
        request_id=request_id,
        compressed=AsymSpecCompressedCoordinates(
            prompt_len=compressed_prompt_len, canonical_len=initial_compressed_len
        ),
        base=AsymSpecBaseCoordinates(
            prompt_len=compressed_prompt_len,
            canonical_len=initial_compressed_len,
            observed_len=initial_compressed_len,
        ),
        full=AsymSpecFullCoordinates(
            prompt_len=full_prompt_len,
            canonical_len=initial_full_len,
            augmentation_offset=augmentation_offset,
        ),
        block_tables=runtime,
    )
