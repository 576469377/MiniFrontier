"""Continue from completed V-C on the same visible device, verifying each independent handoff."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from minifrontier.data.text import sha256
from scripts.v41_vision_curriculum import read, status


def continue_plan(path, contrastive_output):
    plan = read(path)
    out = contrastive_output
    if (
        plan.get("schema") != "v41-continuation-v1"
        or out.resolve() != Path(plan["contrastive_output"]).resolve()
    ):
        raise ValueError("continuation plan belongs to a different V-C run")
    if [s["stage"] for s in plan["stages"]] != [
        "attach-temporary",
        "visual-ar",
        "extract-vit",
        "attach-main",
        "joint",
    ]:
        raise ValueError("invalid V4.1 curriculum stage order")
    result = read(out / "status.json")
    if result["state"] != "complete" or sha256(out / "tower.pt") != result["tower_sha256"]:
        raise ValueError("missing completed V-C tower")
    logs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    val = [r for r in logs if "validation" in r]
    if len(val) < 2 or val[0]["step"] != 0 or val[-1]["step"] != result["step"]:
        raise ValueError("missing V-C retrieval comparison")
    for stage in plan["stages"]:
        receipt = Path(stage["receipt"])
        if any(sha256(p) != h for p, h in stage["bindings"].items()):
            raise ValueError("stage dataset, mixture or configuration changed")
        artifact = Path(stage["artifact"])
        args = stage["args"]
        training = stage["stage"] in {"visual-ar", "joint"}
        module = "minifrontier.training.train" if training else "scripts.v41_vision_curriculum"
        if receipt.exists():
            record = read(receipt)
            if record["plan_sha256"] != sha256(path) or record["sha256"] != sha256(artifact):
                raise ValueError("curriculum receipt disagrees with artifact or plan")
            continue
        if training:
            run = artifact.parent
            complete = (
                (run / "status.json").exists()
                and read(run / "status.json")["state"] == "complete"
                and artifact.is_file()
            )
            if not complete:
                if (run / "checkpoint.pt").exists():
                    args = list(args)
                    offset = args.index("--init")
                    args[offset : offset + 2] = ["--resume", str(run / "checkpoint.pt")]
                subprocess.run([sys.executable, "-m", module, *args], check=True)
            report = read(run / "status.json")
            if report["state"] != "complete" or report["token_ledger"]["image_occurrences"] <= 0:
                raise ValueError("autoregressive stage has incomplete visual supervision")
        else:
            # Recompute an interrupted attachment with its bound parents into an atomic file.
            temp = artifact.with_name(artifact.name + ".pending")
            if temp.exists():
                temp.unlink()
            subprocess.run([sys.executable, "-m", module, *args, "--output", str(temp)], check=True)
            temp.replace(artifact)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        status(
            receipt,
            dict(
                stage=stage["stage"], plan_sha256=sha256(path), sha256=sha256(artifact), argv=args
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--contrastive-output", type=Path, required=True)
    a = p.parse_args()
    continue_plan(a.plan, a.contrastive_output)
