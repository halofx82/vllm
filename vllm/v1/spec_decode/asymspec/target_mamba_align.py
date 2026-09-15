# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TARGET-only aligned Mamba lifecycle for native AsymSpec verification.

This is a narrow adaptation of the frozen SCALE1 ``specsteer_mamba_utils``:
it reuses vLLM's native copy buffers and fused align kernels, but constrains
them to the physical TARGET recurrent groups.  FULL and BASE draft groups
are deliberately excluded; their compact state is owned by AsymSpec views.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import mamba_utils
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


def select_target_mamba_group_ids(
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, object],
    is_draft_module: Callable[[object], bool],
) -> list[int]:
    """Return only recurrent groups owned by the physical target model.

    The cache-group index is an engine-local coordinate used solely to reach
    native block tables. Ownership comes from module identity, never from a
    historical group number or layer-name prefix.
    """
    selected: list[int] = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, MambaSpec):
            continue
        modules = [forward_context[name] for name in group.layer_names]
        if any(is_draft_module(module) for module in modules):
            continue
        selected.append(group_id)
    if not selected:
        raise RuntimeError("AsymSpec TARGET align lifecycle found no target Mamba groups.")
    return selected


def create_for_target_groups(
    *,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: tuple[MambaStateCopyFunc, ...],
    make_buffer: Callable[..., CpuGpuBuffer],
    device: torch.device,
    mamba_group_ids: Sequence[int],
) -> mamba_utils.MambaBuffers:
    """Create native align buffers restricted to TARGET Mamba groups."""
    gids = list(mamba_group_ids)
    if not gids or len(gids) != len(set(gids)):
        raise ValueError("AsymSpec TARGET Mamba groups must be non-empty and unique.")
    groups = kv_cache_config.kv_cache_groups
    if any(gid < 0 or gid >= len(groups) for gid in gids):
        raise ValueError("AsymSpec TARGET Mamba group index is out of range.")
    specs = [groups[gid].kv_cache_spec for gid in gids]
    if not all(isinstance(spec, MambaSpec) for spec in specs):
        raise ValueError("AsymSpec TARGET align selection contains a non-Mamba group.")
    spec = specs[0]
    assert isinstance(spec, MambaSpec)
    if not all(candidate == spec for candidate in specs):
        raise ValueError("AsymSpec TARGET Mamba groups require one uniform spec.")

    entries_per_req = sum(len(groups[gid].layer_names) for gid in gids) * len(copy_funcs)
    preprocess = mamba_utils.MambaCopyBuffers(
        src_ptrs=make_buffer(max_num_reqs * entries_per_req, dtype=torch.uint64),
        dst_ptrs=make_buffer(max_num_reqs * entries_per_req, dtype=torch.uint64),
        sizes=make_buffer(max_num_reqs * entries_per_req, dtype=torch.int32),
        mamba_group_ids=gids,
        mamba_spec=spec,
    )
    # Native construction enumerates groups while allocating buffers.  Supply
    # the selected subset, then restore engine-global group IDs for binding.
    filtered_config = SimpleNamespace(kv_cache_groups=[groups[gid] for gid in gids])
    postprocess = mamba_utils.MambaSpecDecodeGPUContext.create(
        max_num_reqs=max_num_reqs,
        kv_cache_config=filtered_config,  # type: ignore[arg-type]
        num_state_types=len(copy_funcs),
        device=device,
        make_buffer=make_buffer,
    )
    postprocess.mamba_group_ids = gids
    return mamba_utils.MambaBuffers(
        preprocess=preprocess, postprocess_align=postprocess
    )


def preprocess_target_verifier(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, object],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: mamba_utils.MambaCopyBuffers,
    align_ctx: mamba_utils.MambaSpecDecodeGPUContext,
) -> None:
    """Run native align preprocessing without enabling prefix caching.

    ``preprocess_mamba`` couples its copy invariant to APC.  As frozen SCALE1
    did, use a one-call immutable config view only for that assertion; live
    scheduler/cache policy remains prefix-cache disabled.
    """
    config_view = (
        cache_config
        if cache_config.enable_prefix_caching
        else replace(cache_config, enable_prefix_caching=True)
    )
    mamba_utils.preprocess_mamba(
        scheduler_output,
        kv_cache_config,
        config_view,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
        align_ctx=align_ctx,
    )
