"""Coordinated free-space checks for bounded local training artifacts.

Reservations are serialized across project writers using flock. Existing files
remain present until an atomic replacement succeeds; estimates include that
temporary overlap. No unrelated or historical artifact is deleted.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
from pathlib import Path

GIB = 1024**3


class StorageLimitError(OSError):
    pass


def require_space(path, incoming_bytes, *, reserve_bytes=None):
    if incoming_bytes < 0:
        raise ValueError("incoming bytes must be nonnegative")
    if reserve_bytes is None:
        reserve_bytes = int(float(os.environ.get("MINIFRONTIER_MIN_FREE_GIB", "50")) * GIB)
    if reserve_bytes < 0:
        raise ValueError("disk reserve must be nonnegative")
    parent = Path(path).resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < incoming_bytes + reserve_bytes:
        raise StorageLimitError(
            f"insufficient disk space: {free / GIB:.2f} GiB free, "
            f"{incoming_bytes / GIB:.2f} GiB incoming plus {reserve_bytes / GIB:.2f} GiB reserve; "
            "existing checkpoints retained"
        )
    return free


@contextlib.contextmanager
def reserve_write(path, incoming_bytes, *, reserve_bytes=None):
    """All cooperative writers on this filesystem use the same lock directory."""
    lock_root = Path(os.environ.get("MINIFRONTIER_STORAGE_LOCK_DIR", "/tmp"))
    lock_root.mkdir(parents=True, exist_ok=True)
    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = lock_root / f"minifrontier-storage-{parent.stat().st_dev}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            require_space(parent, incoming_bytes, reserve_bytes=reserve_bytes)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def tensor_payload_bytes(value):
    """Conservative torch serialization estimate, including aliased storages."""
    import torch

    seen = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(v) for v in item)
        return 0

    return int(visit(value) * 1.12) + 8 * 1024**2
