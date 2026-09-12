"""Show durable pipeline status and the newest metrics without importing PyTorch."""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def reverse_metrics(path):
    """Read complete JSON records backwards, including across buffer boundaries."""
    try:
        with path.open("rb") as stream:
            position = stream.seek(0, 2)
            pending = b""
            while position:
                size = min(position, 131072)
                position -= size
                stream.seek(position)
                lines = (stream.read(size) + pending).split(b"\n")
                pending = lines.pop(0)
                for line in reversed(lines):
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(value, dict):
                        yield value
            try:
                value = json.loads(pending)
                if isinstance(value, dict):
                    yield value
            except (ValueError, UnicodeDecodeError):
                pass
    except OSError:
        return


def training_processes(proc_root=Path("/proc")):
    """Return local output directories and PIDs; None means visibility is unavailable."""
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    found: dict[str, list[int]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
            cli = "minifrontier" in args or any(
                Path(arg).name == "minifrontier" for arg in args[:2]
            )
            if not ("minifrontier.training.train" in args or (cli and "train" in args)):
                continue
            output = args[args.index("--output") + 1]
            path = Path(output)
            if not path.is_absolute():
                path = entry.joinpath("cwd").resolve(strict=True) / path
            found.setdefault(str(path.resolve()), []).append(int(entry.name))
        except PermissionError:
            return None
        except (OSError, ValueError, IndexError):
            continue  # A process may exit while its command line is read.
    return found


def formal_row(path, processes):
    run, status = read_json(path / "run.json"), read_json(path / "status.json")
    recent: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    progress: dict[str, Any] = {}
    event = None
    for row in reverse_metrics(path / "metrics.jsonl"):
        if not progress and ("token_ledger" in row or "train_lm_loss" in row):
            progress = row
        if event is None and row.get("event") in {
            "start",
            "train",
            "paused",
            "complete",
            "completed",
        }:
            event = row["event"]
        if row.get("event") == "train" or "train_lm_loss" in row:
            if event is None:
                event = "train"
            if len(recent) < 20:
                recent.append(row)
        if not validation and row.get("event") == "validation":
            validation = row
        if len(recent) == 20 and validation:
            break
    train = progress or (recent[0] if recent else {})
    train_ledger = train.get("token_ledger", train)
    ledger = train_ledger
    saved_ledger = status.get("token_ledger", status.get("ledger", {}))
    step = train.get("step", ledger.get("optimizer_updates", 0))
    if status.get("step", 0) >= step:
        ledger = saved_ledger or ledger
        step = status.get("step", step)
    unit = run.get("unit", "input_tokens" if run.get("input_token_budget") else "ce_tokens")
    budget = (
        run.get("token_budget")
        or run.get("ce_token_budget")
        or run.get("input_token_budget")
        or status.get("token_budget")
    )
    consumed = ledger.get("phase_tokens", ledger.get(unit, 0))
    pids = processes.get(str(path.resolve()), []) if processes is not None else None
    state = "running" if pids else "unknown" if processes is None else "stopped"
    if pids == []:
        durable = (
            status.get("state") if status.get("step", 0) >= step and event != "start" else None
        )
        terminal = event if event in {"paused", "complete", "completed"} else durable
        if terminal in {"paused", "complete", "completed"}:
            state = "completed" if terminal in {"complete", "completed"} else "paused"
    speed_key = "input_per_second" if unit == "input_tokens" else "ce_per_second"
    observations = [
        r.get(
            speed_key,
            r.get("input_batch_actual", 0) / r["step_seconds"]
            if unit == "input_tokens" and r.get("step_seconds", 0) > 0
            else r.get("tokens_per_second")
            if unit == "ce_tokens"
            else None,
        )
        for r in recent
    ]
    speeds = [
        float(s) for s in observations if isinstance(s, (int, float)) and math.isfinite(s) and s > 0
    ]
    speed = sum(speeds) / len(speeds) if speeds else None
    return dict(
        model=path.parent.name,
        phase=run.get("pretraining_program", {}).get("phase", run.get("mf1_phase", path.name)),
        path=str(path),
        state=state,
        pids=pids,
        process_observation="local_proc" if processes is not None else "unavailable",
        recorded_state=status.get("state"),
        step=step,
        ce_tokens=ledger.get("ce_tokens", 0),
        main_ce_tokens=ledger.get(
            "main_ce_tokens",
            train.get("main_ce_tokens", train_ledger.get("ce_tokens", 0))
            + ledger.get("ce_tokens", 0)
            - train_ledger.get("ce_tokens", 0),
        )
        if train
        else ledger.get("main_ce_tokens", ledger.get("ce_tokens", 0)),
        phase_token_budget=budget,
        phase_tokens=consumed,
        budget_unit=unit,
        phase_progress=round(consumed / budget, 4) if budget else None,
        recent_tokens_per_second=round(speed, 2) if speed else None,
        throughput_samples=len(speeds),
        estimate_remaining_update_hours=round(max(0, budget - consumed) / speed / 3600, 2)
        if speed and budget
        else None,
        estimate_scope="recent logged updates only; excludes evaluation, saves, waiting and later phases",
        validation=dict(
            step=validation.get("step"),
            main_ce_tokens=validation.get("main_ce_tokens"),
            nll=validation.get("lm_loss", validation.get("nll")),
            ce_tokens=validation.get("supervised_tokens", validation.get("ce_tokens")),
            scope=validation.get("evaluation_scope"),
            selection_sha256=validation.get("selection_sha256"),
        )
        if validation
        else None,
    )


def formal_status(root, proc_root=Path("/proc")):
    root = Path(root).resolve()
    processes = training_processes(proc_root)
    print(
        json.dumps(
            dict(disk_free_gib=round(shutil.disk_usage(root).free / 1024**3, 2), reserve_gib=80)
        )
    )
    for family in ("minikimik3", "miniqwen4", "minideepseekv4", "minifrontier1"):
        candidates = [
            p for p in (root / "strategy-base-pretraining-v1" / family).glob("*") if p.is_dir()
        ]
        if not candidates:
            print(json.dumps(dict(model=family, state="not_started")))
            continue

        def priority(path):
            files = [path / name for name in ("metrics.jsonl", "status.json", "run.json")]
            return bool((processes or {}).get(str(path))), max(
                (p.stat().st_mtime_ns for p in files if p.exists()), default=0
            )

        print(json.dumps(formal_row(max(candidates, key=priority), processes), ensure_ascii=False))


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
            if "train_lm_loss" in value and "ce_tokens" in value:
                return dict(value, token_ledger=value)
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
        sharing = read_json(current / "co_residency.json")
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
            hardware_scope="shared_gpu_during_run" if sharing else "original_device_allocation",
        )
        if supervisor.get("error"):
            row["error"] = supervisor["error"]
        continuation = read_json(current / "continuation.json")
        if continuation:
            row["prior_diagnostic_ce"] = continuation.get("prior_diagnostic_ce", 0)
            row["cumulative_diagnostic_ce"] = row["prior_diagnostic_ce"] + row["ce_tokens"]
        print(json.dumps(row, ensure_ascii=False))

    seen_outputs = set()
    queues = [
        *root.glob("remote-*/outputs/strategy-*gpu*/queue.json"),
        *root.glob("strategy-exclusive-gpu*/queue.json"),
        *root.glob("strategy-performance-gpu*/queue.json"),
        *root.glob("strategy-real-batch-gpu*/queue.json"),
        *root.glob("strategy-batch16-gpu*/queue.json"),
        *root.glob("strategy-batch-*-gpu*/queue.json"),
        *root.glob("strategy-single-gpu*/queue.json"),
        *root.glob("strategy-shared-gpu*/queue.json"),
    ]
    for queue_path in queues:
        queue = read_json(queue_path)
        plan = read_json(queue_path.parent / "queue-plan.json")
        specs = {job["id"]: job for job in plan.get("jobs", [])}
        for job in queue.get("jobs", []):
            current = Path(job["output"])
            remote = queue_path.parent.parent.parent.name.startswith("remote-")
            logical_output = job["output"]
            if remote:
                relative = current.relative_to(plan["workspace"])
                current = queue_path.parent.parent.parent / relative
                logical_output = str(root.parent / relative)
            if logical_output in seen_outputs:
                continue
            seen_outputs.add(logical_output)
            state = read_json(current / "status.json")
            train = latest_train(current / "metrics.jsonl")
            ledger = (
                state.get("token_ledger", {})
                if state.get("state") == "complete"
                else train.get("token_ledger", {})
            )
            profile = read_json(current / "performance.json")
            sharing = read_json(current / "co_residency.json")
            spec = specs.get(job["id"], {})
            command = spec.get("command", [])
            budget = spec.get("token_budget", 20_000_000)
            if "--token-budget" in command:
                budget = int(command[command.index("--token-budget") + 1])
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
                kind="performance_trial"
                if spec.get("result_file") == "report.json"
                else "single_gpu_trial",
                trial=job["id"],
                gpu_id=job.get("gpu_id"),
                host=plan.get("host_label", "local"),
                state="complete" if state.get("state") == "complete" else job["state"],
                path=str(current),
                step=ledger.get("optimizer_updates", 0),
                ce_tokens=ledger.get("ce_tokens", 0),
                ce_token_budget=budget,
                measured_ce_tokens_per_second=single,
                capability_status="unassessed",
                role=job.get("role", spec.get("role", "primary")),
                hardware_scope="shared_gpu_during_run" if sharing else "original_device_allocation",
                mtp_weight=spec.get("mtp_weight"),
            )
            if single and dual and spec.get("variant") == "reference" and not sharing:
                row["single_vs_dual_run_throughput"] = single / dual
                row["estimated_two_single_runs_vs_one_dual"] = 2 * single / dual
                row["comparison_scope"] = (
                    "same target input batch; microbatch, device count, measurement times and host load may differ"
                )
            if job.get("error"):
                row["error"] = job["error"]
            print(json.dumps(row, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument(
        "--run",
        default="formal",
        help="formal (default), strategy-v2, or a historical educational run",
    )
    args = parser.parse_args()
    if args.run == "formal":
        formal_status(args.root)
        return
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
