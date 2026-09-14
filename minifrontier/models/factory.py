"""One model factory shared by inspection, training and inference."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


def model_classes(name):
    if name == "minifrontier11":
        from .minifrontier11 import MiniFrontier11Config, MiniFrontier11ForCausalLM

        return MiniFrontier11Config, MiniFrontier11ForCausalLM
    if name == "minideepseekv41":
        from .minideepseekv41 import MiniDeepSeekV41Config, MiniDeepSeekV41ForCausalLM

        return MiniDeepSeekV41Config, MiniDeepSeekV41ForCausalLM
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


def model_processing(name):
    """Select preprocessing from the same package as the model implementation."""
    _, model_cls = model_classes(name)
    package = model_cls.__module__.rsplit(".", 1)[0]
    return import_module(f"{package}.processing")


def mf_config(values=None, *, model_version=None):
    """Decode the existing MF config schema into its independent versioned class."""
    values = {} if values is None else dict(values)
    version = values.get("model_version", model_version or "1.0-reference-v1")
    if model_version is not None and version != model_version:
        raise ValueError("explicit model version differs from config or parent checkpoint")
    names = {"1.0-reference-v1": "minifrontier1", "1.1-reference-v1": "minifrontier11"}
    if version not in names:
        raise ValueError(f"unknown MF model version: {version}")
    config_cls, _ = model_classes(names[version])
    return config_cls(**values)


def build_model(name, values=None, *, phase=None):
    if phase is None:
        phase = "sparse_pretrain" if name == "minideepseekv41" else "dense_pretrain"
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
