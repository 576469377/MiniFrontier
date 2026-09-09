"""All-path transactional inference cache; snapshot/replay is required for KDA rollback."""

import weakref

import torch

from minifrontier.models.cache_utils import copy_state


class MiniFrontier1Cache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.length = 0
        self.layers = []
        self.lookup = None
        self.position_base = None
        self.owner = None
        self.signature = None
        self.failed = False

    def prepare(self, owner, ids):
        signature = (
            ids.shape[0],
            ids.device,
            next(owner.parameters()).dtype,
            owner.training_phase,
            tuple(p._version for p in owner.parameters()),
            tuple(b._version for b in owner.buffers()),
        )
        if self.failed or (
            self.owner is not None and (self.owner() is not owner or self.signature != signature)
        ):
            raise ValueError("cache failed or model/phase/weights/batch changed; reset it")
        self.owner, self.signature = weakref.ref(owner), signature
        if not self.layers:
            self.layers = [None for _ in owner.layers]
        self.failed = True

    def commit(self, length, positions):
        self.length += length
        self.position_base = positions.amax(dim=(0, 2)) + 1
        self.failed = False

    def snapshot(self):
        if self.failed:
            raise ValueError("cannot snapshot an invalid cache")
        return {
            key: copy_state(getattr(self, key))
            for key in ("length", "layers", "lookup", "position_base", "owner", "signature")
        }

    def restore(self, snapshot):
        if self.owner is not None and (
            snapshot["owner"] != self.owner or snapshot["signature"] != self.signature
        ):
            raise ValueError("snapshot belongs to another model/batch")
        for key, value in snapshot.items():
            setattr(self, key, copy_state(value))
        self.failed = False

    def reorder(self, indices):
        if (
            self.failed
            or self.signature is None
            or indices.ndim != 1
            or indices.dtype != torch.long
        ):
            raise ValueError("invalid cache reorder")
        order = indices.tolist()
        if not order or min(order) < 0 or max(order) >= self.signature[0]:
            raise ValueError("cache reorder index out of range")
        self.layers = [[copy_state(layer[i]) for i in order] for layer in self.layers]
        if self.lookup is not None:
            self.lookup = [copy_state(self.lookup[i]) for i in order]
        assert self.position_base is not None
        self.position_base = self.position_base[indices.to(self.position_base.device)].clone()
        self.signature = (len(order), *self.signature[1:])
