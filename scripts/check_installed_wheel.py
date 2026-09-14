"""Run this script outside a checkout after installing the built wheel."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import torch

from minifrontier import provenance
from minifrontier.catalog import default_manifest_path
from minifrontier.inference.demo_web import page
from minifrontier.models.minifrontier1.configuration import MF1_VERSION, MF11_VERSION
from minifrontier.training.strategy_gate import check


def run_cli(*arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "minifrontier", *arguments],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert provenance.checkout_root() is None, "this check requires the installed wheel"
    assert provenance.source_identity()["commit"] is None
    dist = distribution("minifrontier")
    assert dist.metadata["License-Expression"] == "Apache-2.0 AND MIT AND LicenseRef-Kimi-K3"
    for name in ("Apache-2.0.txt", "MIT-DeepSeek.txt", "LicenseRef-Kimi-K3.txt"):
        assert any(str(path).endswith("LICENSES/" + name) for path in dist.files or [])
    models = json.loads(run_cli("models", capture_output=True).stdout)
    assert set(models) == {
        "minikimik3",
        "miniqwen4",
        "minideepseekv4",
        "minideepseekv41",
        "minifrontier1",
        "minifrontier11",
    }
    config_root = default_manifest_path().parent
    for model in models.values():
        config = json.loads((config_root / model["config"]).read_text())
        assert config == model["capacity"]
    assert models["minideepseekv41"]["config"] == "minideepseekv41.json"
    assert models["minifrontier11"]["config"] == "minifrontier11.json"
    assert models["minifrontier11"]["capacity"]["model_version"] == MF11_VERSION
    assert json.loads((config_root / "strategies/minideepseekv41-plan.json").read_text())

    assets = files("minifrontier.inference").joinpath("web")
    stylesheet = assets.joinpath("demo.css").read_text(encoding="utf-8")
    assert stylesheet.strip()
    for name in ("checkpoint", "mf1"):
        template = assets.joinpath(f"{name}.html").read_text(encoding="utf-8")
        script = assets.joinpath(f"{name}.js").read_text(encoding="utf-8")
        assert "<!doctype html>" in template.lower() and script.strip()
        rendered = page(name)
        assert stylesheet in rendered and script in rendered
        assert "<!--STYLE-->" not in rendered and "<!--SCRIPT-->" not in rendered

    run_cli("quickstart", "--model", "all", "--device", "cpu", "--output", str(args.output))
    reports = list(args.output.glob("*/report.json"))
    assert len(reports) == 3
    for path in reports:
        report = json.loads(path.read_text())
        assert report["resume_executed"] and report["final_sft"]["state"] == "complete"
    mf1_output = args.output / "minifrontier1"
    run_cli(
        "mf1",
        "quickstart",
        "--model-version",
        "1.0",
        "--device",
        "cpu",
        "--updates",
        "8",
        "--output",
        str(mf1_output),
    )
    report = json.loads((mf1_output / "report.json").read_text())
    assert report["config"]["model_version"] == MF1_VERSION
    assert len(report["results"]) == 5
    assert report["results"][0]["state"] == "paused"
    assert report["results"][0]["step"] == 4
    assert report["results"][1]["step"] == 8
    assert all(result["state"] == "budget_complete_unqualified" for result in report["results"][1:])
    for phase, expected_step, stage in (
        ("pilot", 8, "pretrain"),
        ("indexer", 2, "dense_distill"),
        ("p2", 2, "sparse_cpt"),
        ("sft", 2, "sft"),
    ):
        # Only the tiny fixtures created above are loaded, never research checkpoints.
        saved = torch.load(
            mf1_output / phase / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        assert saved["model_name"] == "minifrontier1"
        assert saved["config"]["model_version"] == MF1_VERSION
        assert saved["mf1_phase"] == phase and saved["step"] == expected_step
        assert saved["stage"] == stage
        del saved
    run_cli(
        "mf1",
        "generate",
        "--checkpoint",
        str(mf1_output / "sft/checkpoint.pt"),
        "--prompt",
        "Color?",
        "--image",
        str(mf1_output / "data/media/train-32-0.png"),
        "--max-new-tokens",
        "4",
        "--temperature",
        "0",
        "--device",
        "cpu",
    )
    mf11_output = args.output / "minifrontier11"
    try:
        run_cli(
            "mf1",
            "quickstart",
            "--model-version",
            "1.1",
            "--device",
            "cpu",
            "--output",
            str(mf11_output),
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        assert "MF1.1 training requires a Git checkout containing its versioned plan" in exc.stderr
    else:
        raise AssertionError("wheel accepted MF1.1 training without its Git versioned plan")
    assert not any(mf11_output.rglob("*.pt")), "rejected MF1.1 training produced a checkpoint"
    assert not (mf11_output / "report.json").exists()
    run_cli(
        "generate",
        "--checkpoint",
        str(args.output / "minideepseekv4/sft/model.pt"),
        "--prompt",
        "What is 10 + 2?",
        "--max-new-tokens",
        "12",
        "--temperature",
        "0",
        "--device",
        "cpu",
    )
    try:
        check(
            "absent-plan", "phase", "absent-evidence", data="absent-data", config=None, output=None
        )
    except ValueError as exc:
        assert "requires a Git checkout" in str(exc)
    else:
        raise AssertionError("wheel accepted formal strategy training without Git resources")
    print(
        "Installed wheel: six-model catalog, bundled configs/demo assets, four CPU offline runs, "
        "MF1.0 resume/SFT/media generation, MF1.1 Git requirement, licenses and formal gate passed"
    )


if __name__ == "__main__":
    main()
