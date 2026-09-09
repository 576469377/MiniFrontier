"""Adopt six LR trials and add six MTP ablations, at most two runs per GPU.

Only GPUs 0-5 are authorized. Existing training processes are adopted unchanged;
the previous exclusive queue controller must be stopped before this one starts.
The frozen training source and original queue plan remain immutable.
"""

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TextIO

from minifrontier.data import sha256
from minifrontier.hardware import query_gpus
from minifrontier.storage import GIB, StorageLimitError, require_space
from scripts import run_single_gpu_queue as queue_helper
from scripts import training_status as status_helper
from scripts.run_single_gpu_queue import (
    predecessor_ready,
    read_json,
    replace_option,
    source_check,
    training_environment,
    write_json,
)
from scripts.training_status import latest_train

MEMORY_GIB = {"minikimik3": 5, "miniqwen4": 11, "minideepseekv4": 7}
PLACEMENTS = (
    ("miniqwen4", 0, (0.0, 0.2)),
    ("minikimik3", 2, (0.0, 0.2)),
    ("minideepseekv4", 4, (0.0, 0.1)),
)
TERMINAL = {"complete", "failed", "stopped_memory"}


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(identity):
    return bool(identity) and process_identity(identity["pid"]) == identity


def validate_plan(plan):
    jobs = plan["jobs"]
    if len(jobs) != 12 or len({j["output"] for j in jobs}) != 12:
        raise ValueError("expected twelve distinct trial outputs")
    if {j["gpu_id"] for j in jobs} != set(range(6)):
        raise ValueError("only GPUs 0-5 are authorized")
    for gpu_id in range(6):
        pair = [j for j in jobs if j["gpu_id"] == gpu_id]
        if len(pair) != 2 or {j["role"] for j in pair} != {"primary", "companion"}:
            raise ValueError("each GPU needs exactly one primary and one companion")
        if sum(j["reserved_gib"] for j in pair) + 2 > 24:
            raise ValueError("planned device memory exceeds capacity including reserve")


def prepare(workspace, source, output, worker):
    workspace, source, output, worker = (p.resolve() for p in (workspace, source, output, worker))
    if output.exists():
        raise FileExistsError("preserve the existing shared queue")
    # Includes pending original trials, all six additions and the largest atomic
    # checkpoint overlap; cooperative checkpoint writers serialize writes.
    require_space(workspace, 92 * GIB)
    old_path = workspace / "outputs/strategy-single-gpu-v2/queue-plan.json"
    old_plan = read_json(old_path)
    old_state = read_json(old_path.parent / "queue.json")
    source_check(source, old_plan["jobs"][0]["source"])
    primary = copy.deepcopy(old_plan["jobs"])
    for job in primary:
        job.update(
            role="primary",
            memory_gib=MEMORY_GIB[job["model"]],
            reserved_gib=MEMORY_GIB[job["model"]] + 1,
        )
    configs, companions = {}, []
    for model, first_gpu, weights in PLACEMENTS:
        reference = next(j for j in primary if j["model"] == model and j["variant"] == "reference")
        for offset, weight in enumerate(weights):
            job = copy.deepcopy(reference)
            variant = f"mtp-{weight:g}"
            destination = output / model / variant
            command = job["command"]
            original = Path(command[command.index("--config") + 1])
            config = read_json(original)
            baseline_weight = config["mtp_loss_coef"]
            config["mtp_loss_coef"] = weight
            config_path = output / "configs" / f"{model}-{variant}.json"
            configs[config_path] = config
            replace_option(command, "--config", config_path)
            replace_option(command, "--output", destination)
            # Only the MTP weight changes the learning recipe; keep the MTP
            # module present at weight zero to preserve initialization layout.
            del job["inputs"][str(original)]
            job.update(
                id=f"{model}/{variant}",
                variant=variant,
                role="companion",
                gpu_id=first_gpu + offset,
                output=str(destination),
                predecessor=None,
                baseline_mtp_weight=baseline_weight,
                mtp_weight=weight,
                objective="MTP loss-weight ablation at fixed Muon reference learning rate",
            )
            companions.append(job)
    plan = dict(
        schema_version=1,
        workspace=str(workspace),
        source_root=str(source),
        output=str(output),
        controller_sha256=sha256(__file__),
        worker=str(worker),
        worker_sha256=sha256(worker),
        old_plan=str(old_path),
        old_plan_sha256=sha256(old_path),
        old_controller=process_identity(old_state["pid"]),
        jobs=[*companions, *primary],
        helper_sha256={
            "queue": sha256(queue_helper.__file__),
            "status": sha256(status_helper.__file__),
        },
        main_budget_eligible=False,
        reserve_gib=2,
        disk_reserve_gib=50,
        future_artifacts_and_atomic_overlap_gib=92,
        scope="six additional MTP ablations; adopt original six LR runs; two processes per GPU maximum",
    )
    validate_plan(plan)
    output.mkdir(parents=True)
    (output / "configs").mkdir()
    for path, config in configs.items():
        write_json(path, config)
    for job in companions:
        config_path = job["command"][job["command"].index("--config") + 1]
        job["inputs"][config_path] = sha256(config_path)
    write_json(output / "queue-plan.json", plan)
    write_json(output / "previous-queue-at-prepare.json", old_state)
    print(json.dumps(dict(plan=str(output / "queue-plan.json"), additional_trials=6)))


def app_counts():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    result: dict[str, int] = {}
    for line in text.splitlines():
        uuid = line.strip()
        result[uuid] = result.get(uuid, 0) + 1
    return result


def can_start(job, gpu, contexts):
    return (
        gpu is not None
        and contexts is not None
        and contexts.get(gpu.uuid, 0) < 2
        and gpu.free_gib >= job["reserved_gib"] + 2
    )


def mark_sharing(path, gpu_id, peers):
    root = Path(path)
    if not (root / "run.json").is_file():
        return
    destination = root / "co_residency.json"
    info = read_json(destination)
    if not info:
        train = latest_train(root / "metrics.jsonl")
        info = dict(
            schema_version=1,
            shared_from=time.time(),
            shared_from_logged_step=train.get("step", 0),
            main_budget_eligible=False,
            scope="contains shared-GPU execution; do not label the entire performance profile dedicated",
            devices={},
        )
    intervals = info["devices"].setdefault(str(gpu_id), [])
    if len(peers) >= 2:
        if not intervals or "ended_at" in intervals[-1]:
            intervals.append(dict(started_at=time.time(), peers=sorted(peers)))
        elif intervals[-1]["peers"] != sorted(peers):
            intervals[-1]["ended_at"] = time.time()
            intervals.append(dict(started_at=time.time(), peers=sorted(peers)))
        else:
            return
    elif intervals and "ended_at" not in intervals[-1]:
        intervals[-1]["ended_at"] = time.time()
    else:
        return
    write_json(destination, info)


def execute(plan_path):
    plan = read_json(plan_path)
    validate_plan(plan)
    if (
        sha256(__file__) != plan["controller_sha256"]
        or sha256(plan["worker"]) != plan["worker_sha256"]
    ):
        raise ValueError("controller or worker changed after preparation")
    if {"queue": sha256(queue_helper.__file__), "status": sha256(status_helper.__file__)} != plan[
        "helper_sha256"
    ]:
        raise ValueError("controller helper source changed after preparation")
    if sha256(plan["old_plan"]) != plan["old_plan_sha256"]:
        raise ValueError("original queue plan changed")
    if alive(plan["old_controller"]):
        raise RuntimeError("stop only the previous queue controller before takeover")
    workspace, source, output = (Path(plan[k]) for k in ("workspace", "source_root", "output"))
    status_path = output / "queue.json"
    if status_path.exists():
        raise FileExistsError("shared controller already started; explicit recovery is required")
    source_check(source, plan["jobs"][0]["source"])
    for job in plan["jobs"]:
        if any(sha256(path) != digest for path, digest in job["inputs"].items()):
            raise ValueError(f"changed input for {job['id']}")
    lock_root = workspace / "outputs/gpu-locks"
    lock_root.mkdir(exist_ok=True)
    leases = []
    for gpu_id in range(6):
        lease = (lock_root / f"gpu-{gpu_id}.lock").open("a")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
    old_status_path = Path(plan["old_plan"]).parent / "queue.json"
    old_state = read_json(old_status_path)
    previous = {j["id"]: j for j in old_state["jobs"]}
    statuses = []
    for job in plan["jobs"]:
        status = dict(
            id=job["id"],
            role=job["role"],
            gpu_id=job["gpu_id"],
            output=job["output"],
            state="waiting_memory",
        )
        if job["role"] == "primary":
            status.update(previous[job["id"]])
            if status["state"] == "running":
                identity = process_identity(status["pid"])
                cmd = (
                    Path(f"/proc/{status['pid']}/cmdline").read_bytes().split(b"\0")
                    if identity
                    else []
                )
                if not identity or str(job["output"]).encode() not in cmd:
                    raise RuntimeError("cannot safely adopt the recorded primary process")
                status.update(identity=identity, adopted=True)
        statuses.append(status)
    state = dict(
        pid=os.getpid(),
        started_at=time.time(),
        plan_sha256=sha256(plan_path),
        main_budget_eligible=False,
        jobs=statuses,
    )
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs: dict[str, TextIO] = {}
    write_json(status_path, state)
    while True:
        devices = {g.index: g for g in query_gpus()}
        contexts = app_counts()
        for job, status in zip(plan["jobs"], statuses, strict=True):
            if status["state"] in TERMINAL:
                continue
            name, gpu = job["id"], devices.get(job["gpu_id"])
            if status["state"] == "running":
                child = processes.get(name)
                code = child.poll() if child is not None else None
                live = code is None if child is not None else alive(status.get("identity"))
                if not live:
                    result = read_json(Path(job["output"]) / "status.json")
                    status.update(
                        state="complete"
                        if result.get("state") == "complete" and (code is None or code == 0)
                        else "failed",
                        exit_code=code,
                        finished_at=time.time(),
                    )
                    if name in logs:
                        logs.pop(name).close()
                    continue
                # Only terminate an added companion, never the adopted primary
                # or unrelated device owners, if the physical reserve is eroded.
                if (
                    job["role"] == "companion"
                    and gpu
                    and gpu.free_gib < 3
                    and alive(status.get("identity"))
                ):
                    os.kill(status["pid"], signal.SIGTERM)
                    status.update(
                        state="stopped_memory",
                        reason="physical GPU free memory below 3 GiB",
                        finished_at=time.time(),
                    )
                continue
            if job["predecessor"] and not predecessor_ready(job["predecessor"]):
                status["state"] = "waiting_predecessor"
                continue
            if not can_start(job, gpu, contexts):
                status["state"] = "waiting_memory"
                continue
            assert gpu is not None and contexts is not None
            try:
                require_space(output, 12 * GIB)
            except StorageLimitError as error:
                status.update(state="waiting_disk", reason=str(error))
                continue
            destination = Path(job["output"])
            if destination.exists():
                status.update(state="failed", error="output already exists; refusing overwrite")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [
                job["command"][0],
                plan["worker"],
                "--memory-gib",
                str(job["memory_gib"]),
                "--",
                *job["command"][4:],
            ]
            log_path = destination.parent / (destination.name + ".log")
            if log_path.exists():
                status.update(state="failed", error="log already exists; refusing overwrite")
                continue
            log = log_path.open("x")
            try:
                child = subprocess.Popen(
                    command,
                    cwd=source,
                    env=training_environment(workspace, source, gpu.uuid),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as error:
                log.close()
                status.update(state="failed", error=str(error))
                continue
            processes[name], logs[name] = child, log
            status.update(
                state="running",
                pid=child.pid,
                identity=process_identity(child.pid),
                gpu_uuid=gpu.uuid,
                started_at=time.time(),
                launch_command=command,
            )
            # Account for a not-yet-initialized CUDA context in this same pass.
            contexts[gpu.uuid] = contexts.get(gpu.uuid, 0) + 1
        for gpu_id in range(6):
            observed = [s for s in statuses if s["gpu_id"] == gpu_id]
            resident = [s for s in statuses if s["gpu_id"] == gpu_id and s["state"] == "running"]
            primary = next(
                j for j in plan["jobs"] if j["gpu_id"] == gpu_id and j["role"] == "primary"
            )
            predecessor = read_json(primary["predecessor"])
            if predecessor.get("trials"):
                trial = predecessor["trials"][-1]
                external = dict(id="previous-dual/" + primary["model"], output=trial["output"])
                observed.append(external)
                if not predecessor_ready(primary["predecessor"]):
                    resident.append(external)
            peers = [s["id"] for s in resident]
            for item in observed:
                mark_sharing(item["output"], gpu_id, peers if item["id"] in peers else [])
        state["updated_at"] = time.time()
        state["devices"] = [
            dict(
                gpu_id=g.index,
                used_mib=g.used_mib,
                free_mib=g.free_mib,
                utilization_percent=g.utilization_percent,
            )
            for g in devices.values()
            if g.index < 6
        ]
        write_json(status_path, state)
        mirrored = {s["id"]: s for s in statuses if s["role"] == "primary"}
        old_state.update(
            managed_by=str(status_path),
            pid=os.getpid(),
            updated_at=time.time(),
            jobs=[mirrored[j["id"]] for j in old_state["jobs"]],
        )
        write_json(old_status_path, old_state)
        if all(s["state"] in TERMINAL for s in statuses):
            return int(any(s["state"] != "complete" for s in statuses))
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("prepare")
    for name in ("workspace", "source-root", "output", "worker"):
        build.add_argument("--" + name, type=Path, required=True)
    run = modes.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.workspace, args.source_root, args.output, args.worker)
    else:
        raise SystemExit(execute(args.plan))


if __name__ == "__main__":
    main()
