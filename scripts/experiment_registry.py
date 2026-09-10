"""Build a read-only experiment index from durable local and mirrored records.

The index never starts a trainer, changes a recipe, or promotes a checkpoint.
Unknown or stale state remains explicit instead of implying training completion.
"""

import argparse
import fcntl
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from scripts.training_status import latest_train, read_json


def experiment_id(host, relative):
    return "exp-" + hashlib.sha256(f"{host}:{relative}".encode()).hexdigest()[:12]


def allowed_root(path):
    return (
        path.is_dir()
        and not path.is_symlink()
        and not (path / ".git").exists()
        and not any(t in path.name for t in ("source", "cache", "controller", "tensorboard"))
        and path.name.startswith(
            (
                "strategy-",
                "mf1-",
                "performance-",
                "miniqwen",
                "minikimi",
                "minideepseek",
                "preview-quickstart",
                "preview-installed-wheel",
                "public-readme-check",
            )
        )
    )


def collect(workspace):
    review = read_json(workspace / "configs/experiments.json")
    outputs = workspace / "outputs"
    roots = [("local", outputs)] + [
        (p.name, p / "outputs")
        for p in sorted(outputs.glob("remote-*"))
        if (p / "outputs").is_dir()
    ]
    entries = []
    issues = []
    for host, root in roots:
        parents = [p for p in root.iterdir() if allowed_root(p)]
        targets: dict[Path, dict[str, Any]] = {}
        for parent in parents:
            for plan_path in parent.rglob("queue-plan.json"):
                queue_root = plan_path.parent
                queue_name = str(queue_root.relative_to(root))
                plan = read_json(plan_path)
                state = read_json(queue_root / "queue.json")
                states = {j["id"]: j for j in state.get("jobs", [])}
                for job in plan.get("jobs", []):
                    try:
                        relative_path = Path(job["output"]).relative_to(
                            Path(plan["workspace"]) / "outputs"
                        )
                    except (KeyError, ValueError):
                        issues.append(
                            dict(
                                kind="unresolved_queue_output",
                                queue=queue_name,
                                trial=job.get("id"),
                            )
                        )
                        continue
                    target = root / relative_path
                    record = dict(
                        job=job,
                        status=states.get(job["id"], {}),
                        queue=queue_name,
                        queue_updated_at=state.get("updated_at", 0),
                        dispatch_paused=(queue_root / "dispatch-pause.json").is_file(),
                    )
                    old = targets.get(target)
                    if old:
                        old_active = old["status"].get("state") == "running"
                        new_active = record["status"].get("state") == "running"
                        if old_active and new_active:
                            issues.append(
                                dict(
                                    kind="duplicate_active_output",
                                    host=host,
                                    output=str(relative_path),
                                    queues=[old["queue"], queue_name],
                                )
                            )
                        if record["queue_updated_at"] <= old["queue_updated_at"]:
                            continue
                    targets[target] = record
            for run in parent.rglob("run.json"):
                targets.setdefault(run.parent, {})
            for experiment_path in parent.rglob("experiment.json"):
                if read_json(experiment_path).get("cohort"):
                    targets.setdefault(experiment_path.parent, {})
            for report_path in parent.rglob("report.json"):
                report = read_json(report_path)
                if "cases" in report and "controls" in report:
                    targets.setdefault(report_path.parent, {})
        for target, binding in sorted(targets.items()):
            relative = str(target.relative_to(root))
            job, queued = binding.get("job", {}), binding.get("status", {})
            run = read_json(target / "run.json")
            report = read_json(target / "report.json")
            state = read_json(target / "status.json")
            supervisor = read_json(target / "supervisor.json")
            interruption = read_json(target / "interruption.json")
            experiment = read_json(target / "experiment.json")
            own_experiment = bool(experiment)
            parent_id = None
            parent_case_state = None
            if not experiment:
                for ancestor in target.parents:
                    if ancestor == root:
                        break
                    experiment = read_json(ancestor / "experiment.json")
                    if experiment:
                        parent_id = experiment_id(host, str(ancestor.relative_to(root)))
                        parent_status = read_json(ancestor / "status.json")
                        parent_case_state = next(
                            (
                                c.get("state")
                                for c in parent_status.get("real", [])
                                if target.name == f"mb{c.get('batch')}"
                            ),
                            None,
                        )
                        break
            train = latest_train(target / "metrics.jsonl")
            data_audit = {}
            if run.get("kind") == "data_construction" and host == "local":
                data_root = (workspace / run.get("data_output", "")).resolve()
                if data_root.is_relative_to(workspace / "data"):
                    data_audit = read_json(data_root / "source-audit.json")
            elif run.get("kind") == "data_construction":
                data_audit = read_json(target / "data-audit.json")
            ledger = train.get("token_ledger", {}) or state.get(
                "token_ledger", state.get("ledger", {})
            )
            kind = (
                "synthetic" if report.get("controls") else "sweep" if own_experiment else "training"
            )
            if run.get("kind") == "data_construction":
                kind = "data_construction"
            status = (
                interruption.get("state")
                or supervisor.get("state")
                or state.get("state")
                or report.get("state")
                or queued.get("state")
                or parent_case_state
                or "unverified"
            )
            if data_audit:
                status = data_audit.get("status", "unverified")
                if status == "building" and host != "local":
                    observation = read_json(target / "process-observation.json")
                    fresh = 0 <= time.time() - observation.get("observed_unix", 0) <= 120
                    status = (
                        "observed_running"
                        if fresh
                        and observation.get("pid") == run.get("pid")
                        and observation.get("argv_matches") is True
                        else "unverified_remote_process"
                    )
                elif status == "building":
                    try:
                        command = Path(f"/proc/{int(run['pid'])}/cmdline").read_bytes()
                        expected = [str(arg).encode() for arg in run.get("command", [])]
                        status = (
                            "running"
                            if expected and command.split(b"\0")[:-1] == expected
                            else "unverified_process_identity"
                        )
                    except (OSError, KeyError, ValueError):
                        status = "interrupted_without_final_status"
            if binding.get("dispatch_paused") and status.startswith(("waiting", "pending")):
                status = "paused_for_review"
            if state.get("state") == "complete" and not interruption:
                ledger = state.get("token_ledger", state.get("ledger", ledger))
            budget = run.get("ce_token_budget", run.get("token_budget", job.get("token_budget")))
            if own_experiment and state.get("real"):
                ledger = dict(ce_tokens=0, optimizer_updates=0)
                for case in state["real"]:
                    child = target / "real" / f"mb{case['batch']}"
                    observed = latest_train(child / "metrics.jsonl").get("token_ledger", {})
                    finished = read_json(child / "status.json")
                    if finished.get("state") == "complete":
                        observed = finished.get("token_ledger", observed)
                    for key in ledger:
                        ledger[key] += observed.get(key, 0)
                command = experiment.get("real_command", [])
                if "--ce-tokens" in command and state.get("real_candidates"):
                    budget = int(command[command.index("--ce-tokens") + 1]) * len(
                        state["real_candidates"]
                    )
            files = [
                p
                for p in [
                    target / "metrics.jsonl",
                    target / "status.json",
                    target / "report.json",
                    target / "interruption.json",
                    target / "supervisor.json",
                    target.with_suffix(".exclusive.log"),
                ]
                if p.exists()
            ]
            newest = max((p.stat().st_mtime for p in files), default=0)
            if data_audit:
                newest = max(newest, data_audit.get("updated_unix", 0))
            if own_experiment:
                newest = max(
                    [newest, *(p.stat().st_mtime for p in target.glob("real/mb*/metrics.jsonl"))]
                )
            if (
                status == "running"
                and time.time() - max(newest, binding.get("queue_updated_at", 0)) > 900
            ):
                status = "unverified_stale"
            profile = read_json(target / "performance.json")
            source = (
                run.get("source")
                or report.get("source")
                or experiment.get("source")
                or job.get("source", {})
            )
            model = (
                run.get("model_name")
                or experiment.get("model")
                or job.get("model")
                or report.get("controls", {}).get("model")
            )
            if not model:
                model = next(
                    (
                        name
                        for name in ("miniqwen4", "minikimik3", "minideepseekv4")
                        if name in relative
                    ),
                    "minifrontier1" if "mf1" in relative else "unknown",
                )
            if kind == "data_construction":
                model = "shared-corpus"
            entry = dict(
                id=experiment_id(host, relative),
                host=host,
                output=relative,
                model=model,
                cohort=experiment.get("cohort", relative.split("/")[0]),
                parent_experiment_id=parent_id,
                comparison_group=experiment.get("comparison_group"),
                kind=kind,
                state=status,
                trainer_state=state.get("state"),
                supervisor_state=supervisor.get("state"),
                queue=binding.get("queue"),
                trial=job.get("id"),
                gpu_id=queued.get("gpu_id"),
                role=job.get("role", experiment.get("role")),
                source=source,
                seed=run.get("seed"),
                batch=run.get("batch_size"),
                input_batch_target=run.get("input_batch_tokens"),
                input_batch_policy=run.get("input_batch_policy", "legacy_whole_microbatch")
                if model != "minifrontier1"
                else "mf1_whole_example",
                input_batch_schedule=run.get("input_batch_schedule"),
                lr_schedule=run.get("schedule"),
                warmup_tokens=run.get("warmup_tokens"),
                sequence_length=run.get(
                    "sequence_length", report.get("controls", {}).get("length")
                ),
                data_sha256=run.get("data_sha256", run.get("dataset_manifest_sha256")),
                tokenizer_sha256=run.get("tokenizer_sha256"),
                ce_tokens=ledger.get("ce_tokens", 0),
                optimizer_updates=ledger.get("optimizer_updates", 0),
                ce_token_budget=budget,
                measured_ce_per_second=profile.get("ce_tokens_per_second"),
                performance_measured_updates=profile.get("measured_updates"),
                capability_status=state.get("capability_status", "unassessed"),
                main_budget_eligible=False
                if kind in {"synthetic", "sweep", "data_construction"}
                or run.get("kind") == "acceptance"
                else None,
                retention=experiment.get("retention", "original_run_policy"),
                interruption_reason=interruption.get("reason"),
                evidence_files=[p.name for p in files],
                sweep_cases={
                    key: [
                        {
                            k: row[k]
                            for k in ("batch", "state", "ce_per_second", "peak_reserved_gib")
                            if k in row
                        }
                        for row in state.get(key, [])
                    ]
                    for key in ("synthetic", "real")
                },
            )
            if kind == "data_construction":
                entry["data_progress"] = dict(
                    data_kind=data_audit.get("kind", "text_candidate_inventory"),
                    formal_admission=data_audit.get("formal_admission", False),
                    accepted_records=sum(
                        s.get("accepted_records", 0) for s in data_audit.get("sources", {}).values()
                    ),
                    candidate_reference_tokens=sum(
                        s.get("accepted_reference_tokens", 0)
                        for s in data_audit.get("sources", {}).values()
                    ),
                    split_reference_tokens=data_audit.get("split_reference_tokens"),
                    database_bytes=data_audit.get("database_bytes"),
                    audit_updated_unix=data_audit.get("updated_unix"),
                    unique_images=data_audit.get("unique_images", 0),
                    media_bytes=data_audit.get("media_bytes", 0),
                    error=data_audit.get("error"),
                )
            families = [
                family
                for family in review.get("families", [])
                if any(
                    fnmatchcase(relative.split("/")[0], pattern) for pattern in family["outputs"]
                )
            ]
            entry["review_family"] = families[0]["id"] if len(families) == 1 else None
            entry["review_decision"] = (
                families[0]["decision"] if len(families) == 1 else "needs_review"
            )
            entry["limitations"] = families[0]["limitations"] if len(families) == 1 else []
            if review and len(families) != 1:
                issues.append(
                    dict(kind="experiment_review_coverage", id=entry["id"], matches=len(families))
                )
            measurements = [
                row["input_tokens"] for row in profile.get("updates", []) if "input_tokens" in row
            ]
            entry["measured_input_per_update"] = (
                dict(
                    min=min(measurements),
                    mean=sum(measurements) / len(measurements),
                    max=max(measurements),
                )
                if measurements
                else None
            )
            entry["audit_findings"] = []
            if model == "minideepseekv4" and run.get("optimizer") == "deepseek_muon":
                entry["effective_muon_lr"] = run.get("lr")
                if run.get("muon_lr") != run.get("lr"):
                    entry["audit_findings"].append("deepseek_muon_lr_argument_unused")
            if kind == "training":
                for field in ("source", "data_sha256", "tokenizer_sha256"):
                    if not entry[field]:
                        entry["audit_findings"].append("missing_" + field)
            if (
                measurements
                and run.get("input_batch_tokens")
                and max(measurements)
                > run["input_batch_tokens"]
                + (run.get("sequence_length") or 0) * run.get("world_size", 1)
            ):
                entry["audit_findings"].append("whole_microbatch_global_batch_overshoot")
            entries.append(entry)
    # Remote migration can leave a local waiting reservation with no run. Keep
    # the executed remote identity and record the abandoned reservation as an alias.
    remote_outputs = {e["output"]: e for e in entries if e["host"] != "local"}
    kept = []
    for entry in entries:
        remote = remote_outputs.get(entry["output"])
        if (
            entry["host"] == "local"
            and remote
            and not entry["evidence_files"]
            and entry["state"].startswith(("waiting", "pending", "delegated_remote"))
        ):
            remote.setdefault("superseded_reservations", []).append(entry["id"])
        else:
            kept.append(entry)
    devices: dict[tuple[str, int], str] = {}
    for entry in kept:
        if entry["state"] == "running" and entry["gpu_id"] is not None:
            device = (entry["host"], entry["gpu_id"])
            if device in devices:
                issues.append(
                    dict(
                        kind="multiple_active_tasks_on_gpu",
                        host=device[0],
                        gpu_id=device[1],
                        experiments=[devices[device], entry["id"]],
                    )
                )
            devices[device] = entry["id"]
    return dict(
        schema_version=1,
        observed_at=time.time(),
        scope="read-only index; mirrored timestamps are observations, not proof of a live remote process",
        experiments=kept,
        issues=issues,
        counts=dict(Counter(e["state"] for e in kept)),
    )


def render(snapshot):
    lines = [
        "# 实验台账",
        "",
        f"更新时间: {datetime.fromtimestamp(snapshot['observed_at'], UTC).isoformat()}。自动读取运行产物。",
        "",
        "`complete` 表示该次执行完成。能力验收单独记录。`stopped_by_user` 不计为完成。",
        "",
        "| ID | 主机 | 模型 | 类型 | 状态 | batch | CE / 预算 | 输出目录 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for e in sorted(
        snapshot["experiments"], key=lambda e: (e["state"] != "running", e["host"], e["output"])
    ):
        progress = (
            (
                f"候选图像 {e['data_progress']['unique_images']}; QA {e['data_progress']['accepted_records']}"
                if e["data_progress"]["data_kind"] == "visual_candidate_inventory"
                else f"候选参考 token {e['data_progress']['candidate_reference_tokens']}"
            )
            if e["kind"] == "data_construction"
            else f"{e['ce_tokens']} / {e['ce_token_budget'] or '—'}"
        )
        lines.append(
            f"| {e['id']} | {e['host']} | {e['model']} | {e['kind']} | {e['state']} | {e['batch'] or '—'} | {progress} | `{e['output']}` |"
        )
    if snapshot["issues"]:
        lines.extend(
            [
                "",
                "需要核对:",
                "",
                *[f"- `{json.dumps(issue, ensure_ascii=False)}`" for issue in snapshot["issues"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _refresh(workspace, output):
    snapshot = collect(workspace)
    output.mkdir(parents=True, exist_ok=True)
    old = read_json(output / "current.json")
    previous = {e["id"]: e["state"] for e in old.get("experiments", [])}
    with (output / "events.jsonl").open("a") as stream:
        for e in snapshot["experiments"]:
            if previous.get(e["id"]) != e["state"]:
                stream.write(
                    json.dumps(
                        dict(
                            observed_at=snapshot["observed_at"],
                            id=e["id"],
                            previous=previous.get(e["id"]),
                            state=e["state"],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    for name, content in [
        ("current.json", json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"),
        ("current.md", render(snapshot)),
    ]:
        temporary = (output / name).with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(output / name)
    return snapshot


def refresh(workspace, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "registry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _refresh(workspace, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--watch", type=int, default=0, help="refresh interval in seconds; 0 means once"
    )
    args = parser.parse_args()
    if args.watch and args.watch < 10:
        parser.error("watch interval must be at least 10 seconds")
    workspace = args.workspace.resolve()
    while True:
        snapshot = refresh(workspace, args.output or workspace / "outputs/experiment-registry")
        print(
            json.dumps(
                dict(
                    experiments=len(snapshot["experiments"]),
                    states=snapshot["counts"],
                    issues=len(snapshot["issues"]),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
