"""Show durable pipeline status and the newest metrics without importing PyTorch."""

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def latest_train(path):
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        for line in reversed(stream.read().splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "train":
                return value
    return {}


def strategy_status(root):
    import shutil

    root = Path(root).resolve()
    print(
        json.dumps(
            dict(disk_free_gib=round(shutil.disk_usage(root).free / 1024**3, 2), reserve_gib=50)
        )
    )
    for family in ("minikimik3", "miniqwen4", "minideepseekv4"):
        diagnostic = root / "strategy-diagnostics-v2" / family
        pilot = root / "strategy-recipe-pilots-v2" / family
        attempts = [
            p
            for p in (root / "strategy-recipe-pilots-v2").glob(f"{family}*")
            if (p / "pilot.json").is_file()
        ]
        if attempts:
            pilot = max(attempts, key=lambda p: (p / "pilot.json").stat().st_mtime_ns)
        supervisor = read_json(pilot / "pilot.json")
        diagnostic = Path(supervisor.get("diagnostic", diagnostic))
        trials = supervisor.get("trials", [])
        current = Path(trials[-1]["output"]) if trials else diagnostic
        state = read_json(current / "status.json")
        run = read_json(current / "run.json")
        train = latest_train(current / "metrics.jsonl")
        ledger = (
            state.get("token_ledger", {})
            if state.get("state") == "complete"
            else train.get("token_ledger", state.get("token_ledger", {}))
        )
        budget = run.get("ce_token_budget", state.get("token_budget"))
        profile = read_json(current / "performance.json")
        speed = profile.get("ce_tokens_per_second")
        row = dict(
            model=family,
            path=str(current),
            supervisor=supervisor.get("stage"),
            state=state.get(
                "state",
                "running"
                if train
                and (
                    supervisor.get("stage", "").startswith("running_")
                    or supervisor.get("stage") == "waiting_diagnostic"
                )
                else "initializing",
            ),
            step=ledger.get("optimizer_updates", 0),
            ce_tokens=ledger.get("ce_tokens", 0),
            ce_token_budget=budget,
            ce_progress=round(ledger.get("ce_tokens", 0) / budget, 4) if budget else None,
            image_occurrences=ledger.get("image_occurrences", 0),
            sequence_length=run.get("sequence_length"),
            measured_ce_tokens_per_second=speed,
            estimate_remaining_update_hours=max(0, budget - ledger.get("ce_tokens", 0))
            / speed
            / 3600
            if speed and budget
            else None,
            estimate_scope="this trial's optimizer updates only; excludes evaluation/checkpoint and later trials",
            source_commit=run.get("source", {}).get("commit"),
            capability_status="unassessed",
        )
        if supervisor.get("error"):
            row["error"] = supervisor["error"]
        continuation = read_json(current / "continuation.json")
        if continuation:
            row["prior_diagnostic_ce"] = continuation.get("prior_diagnostic_ce", 0)
            row["cumulative_diagnostic_ce"] = row["prior_diagnostic_ce"] + row["ce_tokens"]
        print(json.dumps(row, ensure_ascii=False))

    for queue_path in sorted(root.glob("strategy-single-gpu*/queue.json")):
        queue = read_json(queue_path)
        plan = read_json(queue_path.parent / "queue-plan.json")
        specs = {job["id"]: job for job in plan.get("jobs", [])}
        for job in queue.get("jobs", []):
            current = Path(job["output"])
            state = read_json(current / "status.json")
            train = latest_train(current / "metrics.jsonl")
            ledger = (
                state.get("token_ledger", {})
                if state.get("state") == "complete"
                else train.get("token_ledger", {})
            )
            profile = read_json(current / "performance.json")
            spec = specs.get(job["id"], {})
            reference = (
                read_json(Path(spec["reference_performance"]))
                if spec.get("reference_performance")
                else {}
            )
            single, dual = (
                profile.get("ce_tokens_per_second"),
                reference.get("ce_tokens_per_second"),
            )
            row = dict(
                kind="single_gpu_trial",
                trial=job["id"],
                gpu_id=job["gpu_id"],
                state=job["state"],
                path=str(current),
                step=ledger.get("optimizer_updates", 0),
                ce_tokens=ledger.get("ce_tokens", 0),
                ce_token_budget=20_000_000,
                measured_ce_tokens_per_second=single,
                capability_status="unassessed",
            )
            if single and dual and spec.get("variant") == "reference":
                row["single_vs_dual_run_throughput"] = single / dual
                row["estimated_two_single_runs_vs_one_dual"] = 2 * single / dual
                row["comparison_scope"] = (
                    "same global input batch; separate measurement times, shared-host load may differ"
                )
            if job.get("error"):
                row["error"] = job["error"]
            print(json.dumps(row, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument(
        "--run", default="strategy-v2", help="strategy-v2 or a named historical educational run"
    )
    args = parser.parse_args()
    if args.run == "strategy-v2":
        strategy_status(args.root)
        return
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
