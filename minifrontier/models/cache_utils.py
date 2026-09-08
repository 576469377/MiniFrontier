"""Cache state cloning and failure invalidation, independent of attention equations."""

from dataclasses import fields, is_dataclass
from functools import wraps

import torch


def copy_state(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: copy_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(copy_state(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: copy_state(getattr(value, f.name)) for f in fields(value)})
    return value


def cache_transaction(forward):
    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        cache = kwargs.get("cache")
        try:
            return forward(self, *args, **kwargs)
        except BaseException:
            if cache is not None:
                cache.failed = True
            raise

    return wrapped
