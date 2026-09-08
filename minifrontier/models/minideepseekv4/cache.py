"""Bounded SWA tails, completed compressed blocks and explicit rollback state."""

import weakref

import torch

from minifrontier.models.cache_utils import copy_state


class MiniDeepSeekV4Cache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.owner = None
        self.signature = None
        self.length = 0
        self.layers = []
        self.failed = False

    def prepare(self, model, ids):
        if self.failed:
            raise ValueError("cache invalidated by failed forward; reset it")
        signature = (
            ids.shape[0],
            ids.device,
            model.embed.weight.dtype,
            model.training_phase,
            tuple(p._version for p in model.parameters()),
            torch.get_autocast_dtype(ids.device.type)
            if torch.is_autocast_enabled(ids.device.type)
            else None,
        )
        if self.owner is None:
            self.owner = weakref.ref(model)
            self.signature = signature
            self.layers = [{} for _ in model.layers]
        elif self.owner() is not model or signature != self.signature:
            raise ValueError("cache model weights, phase, batch, dtype or device changed")

    def snapshot(self):
        if self.failed:
            raise ValueError("cannot snapshot invalid cache")
        return dict(
            owner=self.owner,
            signature=self.signature,
            length=self.length,
            layers=copy_state(self.layers),
        )

    def restore(self, snapshot):
        if self.owner is not None and (
            self.owner != snapshot["owner"] or self.signature != snapshot["signature"]
        ):
            raise ValueError("snapshot belongs to another model or batch")
        self.owner, self.signature, self.length = (
            snapshot["owner"],
            snapshot["signature"],
            snapshot["length"],
        )
        self.layers = copy_state(snapshot["layers"])
        self.failed = False

    def commit(self, length):
        self.length += length
        if any(layer["length"] != self.length for layer in self.layers):
            self.failed = True
            raise ValueError("attention cache layer lengths disagree")
