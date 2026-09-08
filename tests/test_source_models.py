import json
from pathlib import Path

import pytest

from minifrontier.catalog import inspect_current_models
from minifrontier.cli import main

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/models.json"


def test_current_targets_never_report_old_model_counts():
    result = inspect_current_models(MANIFEST)
    assert set(result) == {"miniqwen4", "minikimik3", "minideepseekv4"}
    assert {key: item["display_name"] for key, item in result.items()} == {
        "miniqwen4": "MiniQwen4",
        "minikimik3": "MiniKimi-K3",
        "minideepseekv4": "MiniDeepSeek-V4",
    }
    for item in result.values():
        assert item["complete_model_parameters"] is None
        assert item["training_ready"] is True
        assert item["gpus_per_model"] == 2
    for name in ("minikimik3", "minideepseekv4"):
        assert result[name]["config"] == name + ".json"
        assert result[name]["implementation"] == name
        assert result[name]["text_backbone_parameters"] is None


def test_source_cli_uses_new_entry(capsys):
    main(["models", "--manifest", str(MANIFEST)])
    result = json.loads(capsys.readouterr().out)
    assert result["miniqwen4"]["implementation"] == "miniqwen4"
    assert "deepseek_v3" not in result


@pytest.mark.parametrize(
    "change", ["legacy_implementation", "legacy_config", "ready", "wrong_revision"]
)
def test_current_entry_rejects_legacy_substitution(tmp_path, change):
    manifest = json.loads(MANIFEST.read_text())
    # Copy the small source capacity config so the manifest stays self-contained.
    for source in MANIFEST.parent.glob("*.json"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    if change == "legacy_implementation":
        manifest["models"]["miniqwen4"]["implementation"] = "qwen_flash_next"
    elif change == "legacy_config":
        manifest["models"]["minikimik3"]["config"] = "../models/mini_kimi_k3_4x3090.json"
    elif change == "ready":
        manifest["models"]["minideepseekv4"]["training_ready"] = False
    else:
        manifest["models"]["miniqwen4"]["source_revision"] = "unverified"
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_current_models(path)


def test_unrelated_recipe_is_not_a_model_manifest(tmp_path):
    unrelated = tmp_path / "recipe.json"
    unrelated.write_text('{"model": "unimplemented"}')
    with pytest.raises(ValueError, match="new-source"):
        inspect_current_models(unrelated)
