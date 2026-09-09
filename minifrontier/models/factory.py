"""One model factory shared by inspection, training and inference."""

from __future__ import annotations

import json
from pathlib import Path


def model_classes(name):
    if name == "minifrontier1":
        from .minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM

        return MiniFrontier1Config, MiniFrontier1ForCausalLM
    if name == "miniqwen4":
        from .miniqwen4 import MiniQwen4Config, MiniQwen4ForCausalLM

        return MiniQwen4Config, MiniQwen4ForCausalLM
    if name == "minikimik3":
        from .minikimik3 import MiniKimiK3Config, MiniKimiK3ForCausalLM

        return MiniKimiK3Config, MiniKimiK3ForCausalLM
    if name == "minideepseekv4":
        from .minideepseekv4 import MiniDeepSeekV4Config, MiniDeepSeekV4ForCausalLM

        return MiniDeepSeekV4Config, MiniDeepSeekV4ForCausalLM
    raise ValueError(f"unknown model: {name}")


def build_model(name, values=None, *, phase="dense_pretrain"):
    config_cls, cls = model_classes(name)
    if values is None:
        if name == "minifrontier1":
            return cls(config_cls(), training_phase=phase)
        from minifrontier.catalog import default_manifest_path

        values = json.loads((default_manifest_path().parent / f"{name}.json").read_text())
    elif isinstance(values, (str, Path)):
        values = json.loads(Path(values).read_text())
    values = dict(values)
    for key in ("ple_layer_ids", "compress_ratios"):
        if key in values:
            values[key] = tuple(values[key])
    config = config_cls(**values)
    if name == "minikimik3":
        if phase != "dense_pretrain":
            raise ValueError("Kimi does not use a sparse-indexer stage")
        return cls(config)
    return cls(config, training_phase=phase)


def configure_posttraining(model):
    """Retain learned sparse routing while freezing the discrete index selector."""
    model.indexer_loss_enabled = False
    for name, parameter in model.named_parameters():
        if ".indexer." in name:
            parameter.requires_grad_(False)
