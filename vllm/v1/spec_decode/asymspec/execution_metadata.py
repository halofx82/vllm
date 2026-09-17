# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-local native attention metadata for future AsymSpec draft forwards.

The V1 target runner continues to own ordinary request metadata.  This module
only builds the metadata for the permanently distinct FULL and BASE draft
trees, using the role-owned request tables established by ``request_state``.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig, replace
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.block_table import BlockTable

from .cache_binding import AsymSpecDraftCacheBindingRuntime
from .logical_cache import AsymSpecLogicalCacheGroup
from .request_state import AsymSpecRecurrentBlockSlots, AsymSpecRequestState
from .views import AsymSpecDraftViews, AsymSpecViewRole

_ROLE_GROUPS = {
    AsymSpecViewRole.FULL: (
        AsymSpecLogicalCacheGroup.FULL_ATTENTION,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_A,
        AsymSpecLogicalCacheGroup.FULL_MAMBA_B,
    ),
    AsymSpecViewRole.BASE: (
        AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_A,
        AsymSpecLogicalCacheGroup.BASE_MAMBA_B,
    ),
}


@dataclass(frozen=True)
class AsymSpecMetadataGroup:
    """Native metadata for one role-owned logical cache group."""

    semantic_group: AsymSpecLogicalCacheGroup
    physical_layer_names: tuple[str, ...]
    forward_context_layer_names: tuple[str, ...]
    block_table: torch.Tensor
    metadata: AttentionMetadata
    recurrent_slots: AsymSpecRecurrentBlockSlots | None


@dataclass(frozen=True)
class AsymSpecViewExecutionMetadata:
    """All native context metadata required for one future draft-tree call."""

    role: AsymSpecViewRole
    query_start: int
    query_end: int
    canonical_end: int
    positions: torch.Tensor
    attention_block_ids: torch.Tensor
    attention_slot_mapping: torch.Tensor
    attention_block_table: torch.Tensor
    common_attention_metadata: CommonAttentionMetadata
    groups: dict[AsymSpecLogicalCacheGroup, AsymSpecMetadataGroup]
    layer_metadata: dict[str, AttentionMetadata]
    layer_slot_mapping: dict[str, torch.Tensor]

    @property
    def query_len(self) -> int:
        return self.query_end - self.query_start


def _coordinates_for_role(
    state: AsymSpecRequestState,
    role: AsymSpecViewRole,
) -> int:
    return (
        state.full.canonical_len
        if role is AsymSpecViewRole.FULL
        else state.base.canonical_len
    )


def _available_end_for_role(
    state: AsymSpecRequestState,
    role: AsymSpecViewRole,
) -> int:
    return (
        state.full.canonical_len
        if role is AsymSpecViewRole.FULL
        else state.base.observed_len
    )


def _resolve_forward_context_names(
    *,
    vllm_config: VllmConfig,
    modules: dict[str, AttentionLayerBase],
) -> dict[str, str]:
    """Resolve real registered names by module identity when available."""
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_context = getattr(compilation_config, "static_forward_context", {})
    by_module_id = {
        id(module): layer_name
        for layer_name, module in static_context.items()
        if isinstance(module, AttentionLayerBase)
    }
    if not by_module_id:
        # Synthetic/unit configurations have no static forward context yet.
        return {physical_name: physical_name for physical_name in modules}
    resolved: dict[str, str] = {}
    for physical_name, module in modules.items():
        try:
            resolved[physical_name] = by_module_id[id(module)]
        except KeyError as error:
            raise ValueError(
                "AsymSpec draft layer is missing from the static forward context: "
                f"{physical_name!r}."
            ) from error
    return resolved


def _table_tensor(table: BlockTable) -> torch.Tensor:
    table.commit_block_table(1)
    return table.block_table.gpu[:1]


def _attention_slots(
    *, table: BlockTable, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    block_table = _table_tensor(table)
    block_indices = positions.to(torch.int64) // table.block_size
    block_ids = block_table[0, block_indices]
    slots = block_ids.to(torch.int64) * table.block_size + positions % table.block_size
    return block_ids, slots


def _common_attention_metadata(
    *,
    table: BlockTable,
    positions: torch.Tensor,
    query_start: int,
    query_end: int,
    slots: torch.Tensor,
) -> CommonAttentionMetadata:
    query_len = query_end - query_start
    query_start_loc_cpu = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([query_end], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(positions.device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(positions.device),
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.tensor([query_start], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=query_end,
        block_table_tensor=_table_tensor(table),
        slot_mapping=slots,
        positions=positions,
        causal=True,
        is_prefilling=torch.tensor(
            [query_len > 1], dtype=torch.bool, device=positions.device
        ),
    )


def _new_metadata_builder(
    *,
    module: AttentionLayerBase,
    spec: object,
    layer_names: tuple[str, ...],
    vllm_config: VllmConfig,
    device: torch.device,
):
    builder_cls = module.get_attn_backend().get_builder_cls()
    return builder_cls(spec, list(layer_names), vllm_config, device)


def _build_group_metadata(
    *,
    group: AsymSpecLogicalCacheGroup,
    table: BlockTable,
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start: int,
    query_end: int,
    canonical_end: int,
    initial_prefill: bool,
    physical_layer_names: tuple[str, ...],
    forward_context_layer_names: tuple[str, ...],
    modules: dict[str, AttentionLayerBase],
    spec: object,
    recurrent_slots: AsymSpecRecurrentBlockSlots | None,
    vllm_config: VllmConfig,
) -> AsymSpecMetadataGroup:
    common = _common_attention_metadata(
        table=table,
        positions=positions,
        query_start=query_start,
        query_end=query_end,
        slots=slots,
    )
    # Initial canonical prefill has no pre-existing recurrent state to anchor.
    # Frozen SCALE1 intentionally leaves this unset for the first FULL chunk;
    # subsequent chunks use their committed start coordinate.
    if isinstance(spec, MambaSpec) and not initial_prefill:
        anchor = torch.tensor(
            [canonical_end], dtype=torch.int32, device=positions.device
        )
        common.seq_lens = anchor
        common._seq_lens_cpu = anchor.cpu()
    module = modules[physical_layer_names[0]]
    builder = _new_metadata_builder(
        module=module,
        spec=spec,
        layer_names=forward_context_layer_names,
        vllm_config=vllm_config,
        device=positions.device,
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        fast_build=True,
    )
    return AsymSpecMetadataGroup(
        semantic_group=group,
        physical_layer_names=physical_layer_names,
        forward_context_layer_names=forward_context_layer_names,
        block_table=common.block_table_tensor,
        metadata=metadata,
        recurrent_slots=recurrent_slots,
    )


def build_asymspec_view_execution_metadata(
    *,
    request_state: AsymSpecRequestState,
    views: AsymSpecDraftViews,
    cache_bindings: AsymSpecDraftCacheBindingRuntime,
    vllm_config: VllmConfig,
    role: AsymSpecViewRole,
    query_len: int,
    query_start: int | None = None,
    canonical_end: int | None = None,
    allow_uncommitted_end: bool = False,
    allow_uncommitted_start: bool = False,
    initial_prefill: bool = False,
    device: torch.device | None = None,
) -> AsymSpecViewExecutionMetadata:
    """Build native role-specific metadata without executing a draft model.

    Args:
        request_state: AsymSpec-local canonical coordinates and block tables.
        views: Shared-weight, permanently distinct FULL/BASE module trees.
        cache_bindings: Permanent role-local draft cache bindings.
        vllm_config: Native configuration used by each metadata builder.
        role: FULL or BASE view to describe.
        query_len: Number of tokens in this future view-local forward.
        query_start: Optional role-local token coordinate; defaults to the
            role's current canonical boundary.
        canonical_end: GDN committed-state anchor; defaults to ``query_start``.
        allow_uncommitted_end: Permit one canonical-driver forward that ends
            beyond the committed boundary, provided its table capacity has
            already been reserved.  The default preserves the existing
            committed/catch-up validation semantics.
        allow_uncommitted_start: Permit a disposable candidate query to start
            after the canonical boundary.  This is FULL-only transaction
            metadata; it never advances request state.
        device: Metadata tensor device; defaults to the view-model device.
    """
    if query_len <= 0:
        raise ValueError("AsymSpec metadata requires a positive query length.")
    binding_marker = getattr(views, "draft_cache_bindings", None)
    if cache_bindings is not getattr(binding_marker, "runtime", None):
        raise ValueError("AsymSpec metadata must use the views' permanent bindings.")
    view = views.view(role)
    if query_start is None:
        query_start = _coordinates_for_role(request_state, role)
    if canonical_end is None:
        canonical_end = query_start
    if query_start < 0 or canonical_end < 0:
        raise ValueError("AsymSpec metadata coordinates must be non-negative.")
    query_end = query_start + query_len
    if not allow_uncommitted_end and query_end > _available_end_for_role(
        request_state, role
    ):
        raise ValueError(
            f"{role.value} metadata reaches {query_end}, beyond its reserved "
            "canonical/observed request state."
        )
    if canonical_end > query_start:
        raise ValueError("AsymSpec GDN canonical anchor cannot lead query start.")
    if device is None:
        device = next(view.model.parameters()).device

    modules = {
        physical_name: module
        for physical_name, module in view.model.named_modules()
        if isinstance(module, AttentionLayerBase)
    }
    forward_context_names = _resolve_forward_context_names(
        vllm_config=vllm_config, modules=modules
    )
    # FULL/BASE use compact Mamba state; TARGET alone owns the align mode.
    # The draft model trees were constructed with this config, and their
    # native metadata builders must see the same cache-mode contract.
    draft_vllm_config = getattr(views, "draft_vllm_config", None)
    if draft_vllm_config is None:
        draft_vllm_config = vllm_config
    # Metadata builders consult ``cache_config.mamba_cache_mode`` in addition
    # to their supplied MambaSpec.  Isolate it from TARGET's align mode even
    # when a caller has subsequently normalized the parent config in-place.
    draft_cache_config = copy(draft_vllm_config.cache_config)
    draft_cache_config.mamba_cache_mode = "none"
    draft_vllm_config = replace(
        draft_vllm_config, cache_config=draft_cache_config
    )
    positions = torch.arange(query_start, query_end, dtype=torch.int64, device=device)
    tables = request_state.block_tables
    attention_group = (
        AsymSpecLogicalCacheGroup.FULL_ATTENTION
        if role is AsymSpecViewRole.FULL
        else AsymSpecLogicalCacheGroup.COMPRESSED_ATTENTION
    )
    attention_table = tables.table_by_group[attention_group]
    if allow_uncommitted_end:
        coordinate = _coordinates_for_role(request_state, role)
        if (
            query_start != coordinate
            and not (allow_uncommitted_start and query_start >= coordinate)
        ):
            raise ValueError(
                "Uncommitted AsymSpec metadata must start at the role's "
                "canonical boundary unless it is a disposable candidate."
            )
        allocated_capacity = (
            len(tables.allocated_blocks_by_group[attention_group])
            * attention_table.block_size
        )
        if query_end > allocated_capacity:
            raise ValueError(
                "Uncommitted AsymSpec metadata exceeds reserved attention "
                "table capacity."
            )
    attention_block_ids, attention_slots = _attention_slots(
        table=attention_table, positions=positions
    )
    common = _common_attention_metadata(
        table=attention_table,
        positions=positions,
        query_start=query_start,
        query_end=query_end,
        slots=attention_slots,
    )

    groups: dict[AsymSpecLogicalCacheGroup, AsymSpecMetadataGroup] = {}
    layer_metadata: dict[str, AttentionMetadata] = {}
    layer_slot_mapping: dict[str, torch.Tensor] = {}
    for semantic_group in _ROLE_GROUPS[role]:
        group_plan = next(
            group
            for group in cache_bindings.logical_pools.plan.groups
            if group.semantic_group is semantic_group
        )
        global_layer_names = tuple(
            global_name
            for global_name in group_plan.member_layer_names
            if global_name in cache_bindings.bindings
            and cache_bindings.bindings[global_name].role is role
        )
        if not global_layer_names:
            raise ValueError(
                f"AsymSpec {role.value} group {semantic_group.value} has no layers."
            )
        bindings = tuple(cache_bindings.bindings[name] for name in global_layer_names)
        physical_names = tuple(binding.physical_layer_name for binding in bindings)
        if len(set(physical_names)) != len(physical_names):
            raise ValueError("AsymSpec metadata resolved one draft module twice.")
        if any(physical_name not in modules for physical_name in physical_names):
            raise ValueError(
                "AsymSpec metadata resolved a layer in the wrong view tree."
            )
        specs = tuple(
            cache_bindings.logical_pools.plan.physical_tensors_by_layer[
                global_name
            ].kv_cache_spec
            for global_name in global_layer_names
        )
        if any(spec != specs[0] for spec in specs[1:]):
            raise ValueError(
                "AsymSpec logical metadata group contains incompatible native "
                f"cache specs: {semantic_group.value}."
            )
        metadata_group = _build_group_metadata(
            group=semantic_group,
            table=tables.table_by_group[semantic_group],
            slots=attention_slots,
            positions=positions,
            query_start=query_start,
            query_end=query_end,
            canonical_end=canonical_end,
            initial_prefill=initial_prefill,
            physical_layer_names=physical_names,
            forward_context_layer_names=tuple(
                forward_context_names[physical_name] for physical_name in physical_names
            ),
            modules=modules,
            spec=specs[0],
            recurrent_slots=tables.recurrent_slots.get(semantic_group),
            vllm_config=draft_vllm_config,
        )
        groups[semantic_group] = metadata_group
        for layer_name in metadata_group.forward_context_layer_names:
            layer_metadata[layer_name] = metadata_group.metadata
            if metadata_group.recurrent_slots is None:
                layer_slot_mapping[layer_name] = attention_slots

    return AsymSpecViewExecutionMetadata(
        role=role,
        query_start=query_start,
        query_end=query_end,
        canonical_end=canonical_end,
        positions=positions,
        attention_block_ids=attention_block_ids,
        attention_slot_mapping=attention_slots,
        attention_block_table=common.block_table_tensor,
        common_attention_metadata=common,
        groups=groups,
        layer_metadata=layer_metadata,
        layer_slot_mapping=layer_slot_mapping,
    )


def recurrent_page_ids(
    slots: AsymSpecRecurrentBlockSlots,
) -> tuple[int, int, int, int]:
    """Return null, committed, and the two K=2 speculative page identities."""
    return (
        slots.null_block_id,
        slots.committed_block_id,
        *slots.speculative_block_ids,
    )
