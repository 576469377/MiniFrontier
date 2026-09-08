"""Per-layer MLA KV, KDA recurrent and short-convolution state with rollback."""

import weakref

import torch

from minifrontier.models.cache_utils import copy_state


class MiniKimiK3Cache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.length = 0
        self.owner = None
        self.signature = None
        self.recurrent_states = []
        self.conv_states = []
        self.key_values = []
        self.failed = False

    def prepare(self, model, ids):
        if self.failed:
            raise ValueError("cache invalidated by an earlier failed forward; reset it")
        signature = (
            ids.shape[0],
            ids.device,
            model.embed_tokens.weight.dtype,
            tuple(p._version for p in model.parameters()),
            torch.get_autocast_dtype(ids.device.type)
            if torch.is_autocast_enabled(ids.device.type)
            else None,
        )
        if self.owner is None:
            self.owner = weakref.ref(model)
            self.signature = signature
            self.recurrent_states = [None] * model.config.num_hidden_layers
            self.conv_states = [None] * model.config.num_hidden_layers
            self.key_values = [None] * model.config.num_hidden_layers
        elif self.owner() is not model or signature != self.signature:
            raise ValueError("cache owner, batch, dtype or device changed")

    def update(self, keys, values, index):
        previous = self.key_values[index]
        if previous is not None:
            keys = torch.cat((previous[0], keys), dim=2)
            values = torch.cat((previous[1], values), dim=2)
        self.key_values[index] = (keys, values)
        return keys, values

    def commit(self, length):
        self.length += length

    def snapshot(self):
        if self.failed:
            raise ValueError("cannot snapshot an invalid cache")
        return dict(
            owner=self.owner,
            signature=self.signature,
            length=self.length,
            recurrent=copy_state(self.recurrent_states),
            conv=copy_state(self.conv_states),
            kv=copy_state(self.key_values),
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
        self.recurrent_states = copy_state(snapshot["recurrent"])
        self.conv_states = copy_state(snapshot["conv"])
        self.key_values = copy_state(snapshot["kv"])
        self.failed = False
