"""Dependency-light cache protocol for the pinned Qwen text computation.

State layout and update semantics follow Transformers 4177486 cache_utils.py
(Apache-2.0). This adapter supports no-grad, unpadded text inference only, not
beam search, rollback, offloading, serialization or training through a cache.
"""

import weakref
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F

from minifrontier.models.cache_utils import copy_state


@dataclass
class MiniQwen4CacheLayer:
    conv_states: dict[int, Tensor] = field(default_factory=dict)
    recurrent_states: dict[int, Tensor] = field(default_factory=dict)
    keys: Tensor | None = None
    values: Tensor | None = None
    indexer_keys: Tensor | None = None
    record_past: bool = False


class MiniQwen4Cache:
    """One cache per model instance and batch; reset before changing either."""

    def __init__(self) -> None:
        self.layers: list[MiniQwen4CacheLayer] = []
        self.length = 0
        self.owner: weakref.ReferenceType | None = None
        self.signature: tuple | None = None
        self.failed = False
        self.position_ids: Tensor | None = None

    def reset(self) -> None:
        self.layers.clear()
        self.length = 0
        self.owner = None
        self.signature = None
        self.failed = False
        self.position_ids = None

    def prepare(self, owner: torch.nn.Module, layers: int, batch: int, device, dtype) -> None:
        signature = (batch, device, dtype, tuple(p._version for p in owner.parameters()))
        if self.failed:
            raise ValueError("cache update failed previously; reset before reuse")
        if self.owner is not None and (self.owner() is not owner or self.signature != signature):
            raise ValueError("cache belongs to another model, batch, device or dtype; reset it")
        self.owner, self.signature = weakref.ref(owner), signature
        if not self.layers:
            self.layers = [MiniQwen4CacheLayer() for _ in range(layers)]
        # A failed/partial forward must never leave a reusable inconsistent cache.
        self.failed = True

    def commit(self, length: int) -> None:
        self.length += length
        self.failed = False

    def snapshot(self):
        if self.failed:
            raise ValueError("cannot snapshot an invalid cache")
        return dict(
            owner=self.owner,
            signature=self.signature,
            length=self.length,
            layers=copy_state(self.layers),
            position_ids=copy_state(self.position_ids),
        )

    def restore(self, snapshot):
        if self.owner is not None and (
            self.owner != snapshot["owner"] or self.signature != snapshot["signature"]
        ):
            raise ValueError("snapshot belongs to another cache owner or batch")
        self.owner, self.signature, self.length = (
            snapshot["owner"],
            snapshot["signature"],
            snapshot["length"],
        )
        self.layers = copy_state(snapshot["layers"])
        self.position_ids = copy_state(snapshot["position_ids"])
        self.failed = False

    def has_previous_state(self, layer_idx: int, state_idx: int = 0) -> bool:
        return state_idx in self.layers[layer_idx].conv_states

    def update_conv_state(
        self, states: Tensor, layer_idx: int, state_idx: int = 0, *, conv_kernel_size: int
    ) -> Tensor:
        layer = self.layers[layer_idx]
        previous = layer.conv_states.get(state_idx)
        if previous is None:
            full = F.pad(states, (max(0, conv_kernel_size - states.shape[-1]), 0))
            layer.conv_states[state_idx] = full[..., -conv_kernel_size:].clone()
        else:
            if previous.shape[-1] != conv_kernel_size:
                raise ValueError("convolution cache width changed")
            full = torch.cat((previous, states), dim=-1)
            previous.copy_(full[..., -conv_kernel_size:])
        return full

    def update_recurrent_state(self, states: Tensor, layer_idx: int) -> Tensor:
        layer = self.layers[layer_idx]
        if 0 not in layer.recurrent_states:
            layer.recurrent_states[0] = states.clone()
        else:
            layer.recurrent_states[0].copy_(states)
        return layer.recurrent_states[0]

    def update_indexer(self, states: Tensor, layer_idx: int) -> Tensor:
        layer = self.layers[layer_idx]
        previous = layer.indexer_keys
        layer.indexer_keys = (
            states.clone() if previous is None else torch.cat((previous, states), 1)
        )
        return layer.indexer_keys

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        layer = self.layers[layer_idx]
        layer.keys = keys.clone() if layer.keys is None else torch.cat((layer.keys, keys), -2)
        layer.values = (
            values.clone() if layer.values is None else torch.cat((layer.values, values), -2)
        )
        return layer.keys, layer.values
