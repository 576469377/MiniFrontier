"""Export small, path-normalized experiment records without weights or samples."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from minifrontier.data import sha256

RUN_KEYS = (
    "model_name",
    "kind",
    "config",
    "phase",
    "stage",
    "optimizer",
    "world_size",
    "batch_size",
    "grad_accum",
    "input_batch_tokens",
    "sequence_length",
    "seed",
    "data_sha256",
    "tokenizer_sha256",
    "steps",
    "lr",
    "muon_lr",
    "warmup_steps",
    "weight_decay",
    "clip_grad",
    "router_bias_rate",
    "ce_token_budget",
    "input_token_budget",
    "response_token_budget",
    "warmup_tokens",
    "schedule",
    "adam_eps",
    "vision_lr",
    "projector_lr",
    "normalization",
    "token_mixture",
    "media_mixture",
    "source",
    "performance_profile",
    "router_balance",
    "init_transition",
    "parameters",
    "started_at",
)
METRIC_KEYS = (
    "event",
    "step",
    "loss",
    "lm_loss",
    "grad_norm",
    "lr",
    "step_seconds",
    "tokens_per_second",
    "peak_allocated_mib",
    "token_ledger",
    "supervised_tokens",
    "examples",
    "sampling",
    "exact_parameters",
)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def portable(value, workspace):
    if isinstance(value, str):
        return value.replace(str(workspace), "${WORKSPACE}")
    if isinstance(value, list):
        return [portable(item, workspace) for item in value]
    if isinstance(value, dict):
        return {portable(key, workspace): portable(item, workspace) for key, item in value.items()}
    return value


def export(workspace, output):
    workspace, output = workspace.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("use a new timestamped snapshot directory")
    output.mkdir(parents=True)
    parents = workspace / "outputs/strategy-recipe-pilots-v2"
    commands = {}
    launches = read(workspace / "outputs/strategy-diagnostics-v2/launches.json")
    for launch in launches:
        command = launch["command"]
        commands[command[command.index("--output") + 1]] = command
    for path in parents.glob("*/pilot.json"):
        for trial in read(path).get("trials", []):
            commands[trial["output"]] = trial["command"]
    for path in (workspace / "outputs/strategy-diagnostics-v2").glob("*/continuation.json"):
        command = read(path)["command"]
        commands[command[command.index("--output") + 1]] = command
    paths = sorted(parents.glob("*/*/run.json"))
    paths += sorted((workspace / "outputs/strategy-diagnostics-v2").glob("*/run.json"))
    queue_plans, queue_jobs = {}, {}
    for queue_path in sorted((workspace / "outputs").glob("strategy-*gpu*/queue-plan.json")):
        queue_plan = read(queue_path)
        queue_plans[queue_path.parent.name] = {
            k: queue_plan[k] for k in ("scope", "jobs") if k in queue_plan
        }
        for job in read(queue_path.parent / "queue.json").get("jobs", []):
            queue_jobs[job["output"]] = {k: job[k] for k in ("id", "state") if k in job}
        for job in queue_plan.get("jobs", []):
            commands[job["output"]] = job["command"]
            path = Path(job["output"]) / "run.json"
            if path.is_file():
                paths.append(path)
    paths = sorted(set(paths))
    entries = []
    for path in paths:
        root = path.parent
        run, status = read(path), read(root / "status.json")
        metrics = []
        for line in (root / "metrics.jsonl").read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") in {"train", "validation", "rank_verification", "paused"}:
                metrics.append({k: entry[k] for k in METRIC_KEYS if k in entry})
        train = next((e for e in reversed(metrics) if e["event"] == "train"), {})
        ledger = (
            status.get("token_ledger", {})
            if status.get("state") == "complete"
            else train.get("token_ledger", {})
        )
        name = "--".join(root.relative_to(workspace / "outputs").parts)
        profile = read(root / "performance.json")
        sharing = read(root / "co_residency.json")
        record = dict(
            id=name,
            state=status.get("state", "running"),
            objective="learnability diagnostic"
            if "diagnostics" in name
            else "controlled recipe pilot",
            run={k: run[k] for k in RUN_KEYS if k in run},
            command=commands.get(str(root)),
            token_ledger=ledger,
            co_residency=sharing,
            hardware_scope="shared_gpu_during_run" if sharing else "original_device_allocation",
            metrics=metrics,
            performance={
                k: profile[k]
                for k in (
                    "measured_updates",
                    "ce_tokens_per_second",
                    "images_per_second",
                    "frames_per_second",
                    "updates",
                    "scope",
                    "includes",
                )
                if k in profile
            },
            original_run_sha256=sha256(path),
            capability_status="unassessed",
            main_budget_eligible=False,
            limitation="training completion/NLL is not a language or visual capability pass; diagnostic recall is training-set memorization",
        )
        (output / f"{name}.json").write_text(
            json.dumps(portable(record, workspace), indent=2) + "\n"
        )
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "event",
                    "step",
                    "lm_loss",
                    "loss",
                    "lr",
                    "grad_norm",
                    "step_seconds",
                    "tokens_per_second",
                    "ce_tokens",
                    "input_tokens",
                ],
            )
            writer.writeheader()
            for row in metrics:
                flat = {**row, **row.get("token_ledger", {})}
                writer.writerow({key: flat.get(key) for key in writer.fieldnames})
        validation = next((e for e in reversed(metrics) if e["event"] == "validation"), {})
        entries.append(
            dict(
                id=name,
                state=record["state"],
                ce_tokens=ledger.get("ce_tokens", 0),
                validation_lm_loss=validation.get("lm_loss"),
                ce_per_second=profile.get("ce_tokens_per_second"),
            )
        )
    # A retired queue can still say "waiting" after its workers were adopted elsewhere.
    # Actual run artifacts take precedence over those historical controller states.
    for location, job in queue_jobs.items():
        actual = read(Path(location) / "status.json")
        if actual.get("state"):
            job["state"] = actual["state"]
    snapshot = dict(
        captured_at=datetime.now(UTC).isoformat(),
        runs=entries,
        pending_jobs=[
            job
            for location, job in queue_jobs.items()
            if not (Path(location) / "run.json").is_file()
        ],
        queue_jobs=list(queue_jobs.values()),
        portable_queue_plans=portable(queue_plans, workspace),
        scope="numeric records only; no training samples, media, checkpoints, hostname, device UUID or process IDs",
        recipe_selection="not frozen; complete optimizer, LR, MTP, tokenizer and seed comparisons first",
    )
    (output / "index.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    # These files contain provenance, counts and hashes, never corpus rows.
    for relative in (
        "data/strategy-pilot-public-v2/source-audit.json",
        "data/strategy-visual-public-pilot-v3/source-audit.json",
        "data/strategy-recipe-public-v2/recipe-audit.json",
        "data/strategy-recipe-joint-v2/recipe-audit.json",
    ):
        path = workspace / relative
        if path.is_file():
            name = path.parent.name + "--" + path.name
            (output / name).write_text(json.dumps(portable(read(path), workspace), indent=2) + "\n")
    rows = [
        "# Research preview experiment snapshot",
        "",
        f"Captured: {snapshot['captured_at']}",
        "",
        "Numeric records only. Running trials are partial; diagnostics measure learnability, not chat quality.",
        "",
        "| Experiment | State | CE tokens | Latest validation LM NLL | Measured CE/s |",
        "|---|---|---:|---:|---:|",
    ]
    for item in entries:
        rows.append(
            f"| [{item['id']}]({item['id']}.json) ([CSV]({item['id']}.csv)) | {item['state']} | {item['ce_tokens']} | {item['validation_lm_loss']} | {item['ce_per_second']} |"
        )
    rows += [
        "",
        "Commands retain `${WORKSPACE}` as an explicit binding. Use the recorded source commit and data/tokenizer hashes.",
        "A matching dataset must be reconstructed from its pinned preparation recipe; this snapshot does not redistribute the dataset.",
        "See the project experiment guide for failure reports, limitations and reconstruction instructions.",
    ]
    (output / "README.md").write_text("\n".join(rows) + "\n")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.workspace, args.output)
    print(json.dumps(dict(runs=len(report["runs"]), output=str(args.output))))


if __name__ == "__main__":
    main()
