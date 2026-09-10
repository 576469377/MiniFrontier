"""Run a bounded experiment queue with one project task per physical GPU.

Plans contain immutable commands, source/input identities, and optional learning
gates. Existing processes can be observed without interrupting their training.
Each host runs its own controller and keeps its own device leases.
"""

import argparse
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TextIO

from minifrontier.data import sha256
from minifrontier.hardware import query_gpus
from minifrontier.storage import GIB, StorageLimitError, require_space
from scripts.run_shared_gpu_queue import alive, process_identity
from scripts.run_single_gpu_queue import (
    predecessor_ready,
    read_json,
    source_check,
    training_environment,
    write_json,
)

TERMINAL = {"complete", "failed", "needs_diagnosis", "blocked_by_control_run"}


def compute_apps():
    """NVML PIDs are host identifiers: never use them to signal local processes."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        result: dict[str, set[int]] = {}
        for line in raw.splitlines():
            if line.strip():
                uuid, pid = line.split(",")
                result.setdefault(uuid.strip(), set()).add(int(pid))
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def eligible(gpu, apps, reserved, policy, job):
    if gpu is None or apps is None or gpu.index in reserved:
        return False
    exceptions = policy.get("ignored_contexts", {}).get(str(gpu.index), {})
    ignored = set(exceptions.get("host_pids", [])) if exceptions.get("uuid") == gpu.uuid else set()
    if apps.get(gpu.uuid, set()) - ignored:
        return False
    # An explicitly documented inert context may still consume VRAM. Subtract
    # nothing from memory accounting; use the actual free memory reported now.
    if not ignored and gpu.used_mib >= 1024:
        return False
    return gpu.utilization_percent <= 5 and gpu.free_gib >= job["memory_gib"] + 1 + job.get(
        "reserve_gib", 2
    )


def gate(job):
    if job.get("predecessor_receipt"):
        receipt = read_json(job["predecessor_receipt"])
        if receipt.get("state") != "complete" or not receipt.get("verified_exports"):
            return "waiting_predecessor", {}
    control = job.get("control_result")
    if control:
        base = Path(control)
        result = read_json(base / "supervisor.json")
        if result.get("state") != "complete":
            state = result.get("state", "")
            return (
                "blocked_by_control_run"
                if state == "failed" or state.startswith("stopped")
                else "waiting_control_result",
                {},
            )
        train = read_json(base / "generation-train.json")
        val = read_json(base / "generation-val.json")
        summary = train.get("summaries", {}).get("language/matched", {})
        visual = [
            val.get("summaries", {}).get(f"{domain}/matched", {}) for domain in ("vision", "video")
        ]
        values = dict(
            train_correct=summary.get("correct", 0),
            train_total=summary.get("total", 0),
            train_eos=train.get("eos_count", 0),
            visual_correct=sum(x.get("correct", 0) for x in visual),
            visual_total=sum(x.get("total", 0) for x in visual),
        )
        passed = (
            values["train_total"] == 32
            and values["train_correct"] >= 29
            and values["train_eos"] >= 30
            and values["visual_total"] == 32
            and values["visual_correct"] >= 28
            and (base / "checkpoint.pt").is_file()
        )
        return ("ready" if passed else "needs_diagnosis"), values
    if job.get("predecessor") and not predecessor_ready(job["predecessor"]):
        return "waiting_predecessor", {}
    return "ready", {}


def validate(plan):
    ids = plan["allowed_gpu_ids"]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("provide distinct nonnegative physical GPU indices")
    jobs = plan["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs) or len({j["output"] for j in jobs}) != len(jobs):
        raise ValueError("job IDs and outputs must be unique")
    for job in jobs:
        if not 0 < job["memory_gib"] < 80 or not job.get("reserve_gib", 2) >= 2:
            raise ValueError("invalid memory budget")
        command = job["command"]
        if any(
            "torch.distributed.run" in arg or "nproc_per_node" in arg or "torchrun" in arg
            for arg in command
        ):
            raise ValueError("distributed launches are not supported by the exclusive queue")
        if job.get("allowed_gpu_ids") and not set(job["allowed_gpu_ids"]) <= set(ids):
            raise ValueError("job requests a device outside the host pool")
    if plan.get("main_budget_eligible") is not False:
        raise ValueError("this queue is for bounded acceptance experiments")


def completed(job):
    result = read_json(Path(job["output"]) / job.get("result_file", "status.json"))
    if result.get("state") != "complete":
        return False
    if job.get("result_file") == "report.json":
        return bool(result.get("cases")) and all(
            c.get("ce_per_second", 0) > 0 for c in result["cases"]
        )
    return True


def execute(plan_path):
    plan = read_json(plan_path)
    validate(plan)
    if plan["controller_sha256"] != sha256(__file__):
        raise ValueError("controller changed after the plan was frozen")
    workspace = Path(plan["workspace"])
    output = plan_path.parent
    controller_lock = (output / "controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / "queue.json"
    digest = sha256(plan_path)
    state = read_json(status_path)
    if state and state.get("plan_sha256") != digest:
        raise ValueError("existing queue belongs to a different plan")
    if not state:
        state = dict(
            plan_sha256=digest,
            started_at=time.time(),
            main_budget_eligible=False,
            jobs=[dict(id=j["id"], output=j["output"], state="pending") for j in plan["jobs"]],
        )
    state.update(pid=os.getpid(), identity=process_identity(os.getpid()), state="running")
    lock_root = workspace / "outputs/gpu-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    (workspace / "outputs/single-gpu-tmp").mkdir(exist_ok=True)
    running: dict[str, tuple[subprocess.Popen[bytes], TextIO, TextIO]] = {}
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        devices = {g.index: g for g in query_gpus()}
        apps = compute_apps()
        reserved = set()
        observed = []
        for prior in plan.get("adopted", []):
            active = alive(prior["identity"])
            if active:
                reserved.update(prior["gpu_ids"])
            observed.append(dict(prior, state="running" if active else "finished"))
        for job, status in zip(plan["jobs"], state["jobs"], strict=True):
            if status["state"] != "running":
                continue
            owned = running.get(job["id"])
            active = owned[0].poll() is None if owned else alive(status.get("identity"))
            if active:
                reserved.add(status["gpu_id"])
                continue
            code = owned[0].returncode if owned else None
            status.update(
                state="complete" if code in (None, 0) and completed(job) else "failed",
                exit_code=code,
                finished_at=time.time(),
            )
            if owned:
                owned[1].close()
                owned[2].close()
                del running[job["id"]]
        for job, status in zip(plan["jobs"], state["jobs"], strict=True):
            if status["state"] in TERMINAL | {"running"}:
                continue
            readiness, values = gate(job)
            status.update(state=readiness, gate_observed=values)
            if readiness != "ready":
                continue
            status["state"] = "waiting_gpu"
            for index in job.get("allowed_gpu_ids", plan["allowed_gpu_ids"]):
                gpu = devices.get(index)
                if not eligible(gpu, apps, reserved, plan, job):
                    continue
                lease = (lock_root / f"gpu-{index}.lock").open("a")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    # Re-query under the lease: another cooperative launcher
                    # may have acquired a CUDA context since the first sample.
                    fresh = next((g for g in query_gpus() if g.index == index), None)
                    if fresh is None or not eligible(fresh, compute_apps(), reserved, plan, job):
                        lease.close()
                        continue
                    require_space(output, 16 * GIB)
                    source = Path(job["source_root"])
                    source_check(source, job["source"])
                    if any(sha256(path) != value for path, value in job["inputs"].items()):
                        raise ValueError("frozen experiment input changed")
                    destination = Path(job["output"])
                    if destination.exists():
                        raise FileExistsError("output exists; explicit recovery is required")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    command = [str(index) if arg == "{gpu}" else arg for arg in job["command"]]
                    if "--init" in command:
                        status["init_sha256"] = sha256(command[command.index("--init") + 1])
                    log = destination.with_suffix(".exclusive.log").open("x")
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=source,
                            env=training_environment(workspace, source, fresh.uuid),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            # Preserve exclusivity if this controller exits;
                            # the top-level worker keeps the lease descriptor.
                            pass_fds=(lease.fileno(),),
                        )
                    except BaseException:
                        log.close()
                        raise
                    running[job["id"]] = process, lease, log
                    reserved.add(index)
                    status.update(
                        state="running",
                        pid=process.pid,
                        identity=process_identity(process.pid),
                        gpu_id=index,
                        gpu_uuid=fresh.uuid,
                        command=command,
                        started_at=time.time(),
                    )
                except StorageLimitError as error:
                    lease.close()
                    status.update(state="waiting_disk", reason=str(error))
                except Exception as error:
                    lease.close()
                    status.update(state="failed", error=f"{type(error).__name__}: {error}")
                break
        state.update(updated_at=time.time(), adopted=observed, reserved_gpu_ids=sorted(reserved))
        write_json(status_path, state)
        if all(j["state"] in TERMINAL for j in state["jobs"]) and not any(
            alive(j["identity"]) for j in plan.get("adopted", [])
        ):
            state["state"] = "complete"
            break
        time.sleep(plan.get("poll_seconds", 15))
    if stopping:
        state["state"] = "controller_stopped_workers_preserved"
    write_json(status_path, state)
    controller_lock.close()
    return int(any(j["state"] == "failed" for j in state["jobs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.plan.resolve()))


if __name__ == "__main__":
    main()
