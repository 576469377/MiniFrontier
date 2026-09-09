"""Run this script outside a checkout after installing the built wheel."""

import argparse
import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

from minifrontier import provenance
from minifrontier.training.strategy_gate import check


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
    subprocess.run([sys.executable, "-m", "minifrontier", "models"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "minifrontier",
            "quickstart",
            "--model",
            "all",
            "--device",
            "cpu",
            "--output",
            str(args.output),
        ],
        check=True,
    )
    reports = list(args.output.glob("*/report.json"))
    assert len(reports) == 3
    for path in reports:
        report = json.loads(path.read_text())
        assert report["resume_executed"] and report["final_sft"]["state"] == "complete"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "minifrontier",
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
        ],
        check=True,
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
        "Installed wheel: models, three offline runs, CLI generation, licenses and formal gate passed"
    )


if __name__ == "__main__":
    main()
