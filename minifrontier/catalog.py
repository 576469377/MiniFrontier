"""Current source-reproduction entry, deliberately independent of the old registry."""

from __future__ import annotations

import json
import sysconfig
from pathlib import Path
from typing import Any


def default_manifest_path() -> Path:
    """Find the checkout catalog or the wheel's installed shared data."""
    checkout = Path(__file__).resolve().parents[1] / "configs/models.json"
    if checkout.is_file():
        return checkout
    installed = Path(sysconfig.get_path("data")) / "share/minifrontier/configs/models.json"
    if not installed.is_file():
        raise FileNotFoundError(
            "model catalog is missing; reinstall MiniFrontier or pass --manifest"
        )
    return installed


def inspect_current_models(
    manifest_path: str | Path, *, count_backbones: bool = False
) -> dict[str, Any]:
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != "new_source_reproductions_only"
    ):
        raise ValueError("expected a new-source-reproduction manifest, not an old training recipe")
    if set(manifest.get("models", {})) != {"miniqwen4", "minikimik3", "minideepseekv4"}:
        raise ValueError("current scope is Qwen4, Kimi-K3 and DeepSeek-V4 only")
    from minifrontier.models.factory import build_model, model_classes

    result = {}
    for name, entry in manifest["models"].items():
        if entry["implementation"] != name:
            raise ValueError(f"unsupported source implementation: {entry['implementation']}")
        config_name = entry["config"]
        if not isinstance(config_name, str) or Path(config_name).name != config_name:
            raise ValueError("configuration must reside beside its manifest")
        config_path = (path.parent / config_name).resolve(strict=True)
        if config_path.parent != path.parent:
            raise ValueError("configuration must reside beside its manifest")
        config_class, model_class = model_classes(name)
        revision = getattr(model_class, "source_revision", None)
        if name == "miniqwen4":
            from minifrontier.models.miniqwen4 import MiniQwen4TextModel

            revision = MiniQwen4TextModel.source_revision
        if entry["source_revision"] != revision:
            raise ValueError("source revision does not match implementation")
        if entry["training_ready"] is not True:
            raise ValueError(
                "catalog training status differs from the accepted text implementation"
            )
        values = json.loads(config_path.read_text())
        for key in ("ple_layer_ids", "compress_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        config_class(**values).upstream_config()
        record = dict(
            entry,
            capacity=values,
            complete_model_parameters=None,
            text_backbone_parameters=None,
            dense_lm_parameters_without_mtp=None,
        )
        if count_backbones:
            model = build_model(name, values)
            count = sum(p.numel() for p in model.parameters() if p.is_floating_point())
            head = model.lm_head if hasattr(model, "lm_head") else model.head
            record["dense_lm_parameters_without_mtp"] = count
            record["text_backbone_parameters"] = count - sum(p.numel() for p in head.parameters())
            record["nonfloating_routing_entries"] = sum(
                p.numel() for p in model.parameters() if not p.is_floating_point()
            )
            del model
        result[name] = record
    return result
