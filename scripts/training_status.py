"""Show durable pipeline status and the newest metrics without importing PyTorch."""

import argparse
import json
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--run", default="educational-v1")
    args = parser.parse_args()
    for path in sorted(Path(args.root).glob(f"*/{args.run}/recipe.json")):
        recipe = json.loads(path.read_text())
        state = json.loads((path.parent / "pipeline.json").read_text())
        stage = state.get("stage", recipe["stages"][-1][0])
        metrics_path = path.parent / stage / "metrics.jsonl"
        metrics = []
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # A live logger may be between writes.
        train: dict[str, Any] = next(
            (row for row in reversed(metrics) if row["event"] == "train"), {}
        )
        quality_path = path.parent / "quality.json"
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        print(
            json.dumps(
                dict(
                    model=recipe["model"],
                    gpus=recipe["gpus"],
                    state=state["state"],
                    capability_status=quality.get("capability_status", "unassessed"),
                    stage=stage,
                    step=train.get("step", 0),
                    loss=train.get("loss"),
                    tokens_per_second=train.get("tokens_per_second"),
                    pid=state.get("pid"),
                    log=str(path.parent / stage / "train.log"),
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
