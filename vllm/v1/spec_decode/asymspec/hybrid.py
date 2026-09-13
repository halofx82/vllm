# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free hybrid state descriptions for AsymSpec draft views."""

from dataclasses import dataclass
from re import search

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


@dataclass(frozen=True)
class AsymSpecAttentionStateSpec:
    """Token-KV state ownership metadata for one ordinary attention layer."""

    layer_name: str
    layer_index: int


@dataclass(frozen=True)
class AsymSpecRecurrentStateSpec:
    """Native Mamba/GDN state metadata for one recurrent layer.

    ``shapes`` and ``dtypes`` are obtained from the loaded Mamba layer itself,
    which is the same source used by vLLM when it later creates ``MambaSpec``.
    No state tensor or cache page is created here.
    """

    layer_name: str
    layer_index: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    mamba_type: MambaAttentionBackendEnum


@dataclass(frozen=True)
class AsymSpecHybridStateSpec:
    """Compact/native hybrid state contract for one logical AsymSpec view."""

    attention: tuple[AsymSpecAttentionStateSpec, ...]
    recurrent: tuple[AsymSpecRecurrentStateSpec, ...]
    compact_recurrent_state_slots: int


def _layer_index(layer_name: str) -> int:
    match = search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError(f"Cannot determine decoder layer index for {layer_name!r}.")
    return int(match.group(1))


def describe_qwen3_5_hybrid_state(
    model: nn.Module,
    num_speculative_tokens: int,
) -> AsymSpecHybridStateSpec:
    """Describe Qwen3.5 attention and GDN state without allocating it.

    The compact/native GDN layout retains the committed state plus one state
    per speculative token. This is the validated ``MambaSpec`` ``none``
    layout and is intentionally the only layout represented here.
    """
    if num_speculative_tokens != 2:
        raise ValueError("AsymSpec hybrid state currently requires K=2.")

    attention: list[AsymSpecAttentionStateSpec] = []
    recurrent: list[AsymSpecRecurrentStateSpec] = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, MambaBase):
            recurrent.append(
                AsymSpecRecurrentStateSpec(
                    layer_name=layer_name,
                    layer_index=_layer_index(layer_name),
                    shapes=tuple(layer.get_state_shape()),
                    dtypes=layer.get_state_dtype(),
                    mamba_type=layer.mamba_type,
                )
            )
        elif isinstance(layer, AttentionLayerBase):
            attention.append(
                AsymSpecAttentionStateSpec(
                    layer_name=layer_name,
                    layer_index=_layer_index(layer_name),
                )
            )

    attention.sort(key=lambda spec: (spec.layer_index, spec.layer_name))
    recurrent.sort(key=lambda spec: (spec.layer_index, spec.layer_name))
    if not attention or not recurrent:
        raise ValueError(
            "AsymSpec currently requires a hybrid Qwen3.5 draft with both "
            "ordinary attention and GDN/recurrent layers."
        )

    return AsymSpecHybridStateSpec(
        attention=tuple(attention),
        recurrent=tuple(recurrent),
        compact_recurrent_state_slots=1 + num_speculative_tokens,
    )
