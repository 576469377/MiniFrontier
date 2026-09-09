"""Shared inference API; browser and argument parsing are imported on demand."""

from importlib import import_module

from .runtime import generate_ids as generate_ids
from .runtime import load_checkpoint as load_checkpoint
from .runtime import respond as respond


def __getattr__(name):
    if name in {"PAGE", "checkpoints", "demo_device", "experiment_directories", "serve"}:
        return getattr(import_module(".demo", __name__), name)
    if name == "main":
        return import_module(".cli", __name__).main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
