"""Exact prefix recomputation for reference generation; no KV-speed claim."""

import weakref

import torch


class MiniDeepSeekV41Cache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ids = None
        self.media = None
        self.owner = None
        self.signature = None
        self.length = 0
        self.failed = False

    def forward(self, model, ids, *, attention_mask=None, labels=None, media=None, **kwargs):
        if self.failed:
            raise ValueError("reference cache failed; reset before reuse")
        try:
            if model.training or torch.is_grad_enabled() or labels is not None:
                raise ValueError("reference cache requires eval/no_grad without labels")
            if attention_mask is not None and not attention_mask.bool().all():
                raise ValueError("reference cache accepts unpadded batches only")
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
                self.owner, self.signature = weakref.ref(model), signature
            elif self.owner() is not model or self.signature != signature:
                raise ValueError("reference cache belongs to different weights, model or batch")
            if self.length and media:
                raise ValueError("images must be supplied during the initial prefill")
            full = ids if self.ids is None else torch.cat((self.ids, ids), -1)
            output = model(full, media=self.media if self.length else media, **kwargs)
            for name in ("logits", "hidden_states", "multistream_hidden"):
                value = getattr(output, name)
                if value is not None:
                    setattr(output, name, value[:, -ids.shape[1] :])
            self.ids = full.detach().clone()
            if not self.length:
                self.media = media
            self.length = full.shape[1]
            return output
        except BaseException:
            self.failed = True
            raise
