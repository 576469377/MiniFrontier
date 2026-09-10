"""Six single-GPU learning-rate trials, after the existing paired-GPU pilots.

The reference and lower-LR trials use the same frozen trainer, data, seed and
global input target. Neither this queue nor its checkpoints admit formal PT.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from minifrontier.data import sha256
from minifrontier.hardware import query_gpus
from minifrontier.storage import GIB, StorageLimitError, require_space

FAMILIES = {
    "minikimik3": (0, "minikimik3", "--muon-lr", "0.005"),
    "miniqwen4": (2, "miniqwen4-after-q0-extension", "--muon-lr", "0.003"),
    "minideepseekv4": (4, "minideepseekv4", "--lr", "0.0001"),
}
COMPLETE = "two_trials_complete_further_comparisons_required"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def replace_option(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def single_command(original, output, variant, option, value):
    marker = next(
        i for i in range(len(original) - 1) if original[i : i + 2] == ["-m", "minifrontier"]
    )
    command = [original[0], *original[marker:]]
    if any(flag in command for flag in ("--resume", "--init")):
        raise ValueError("single-GPU comparisons must start from the same random initialization")
    replace_option(command, "--output", output)
    # Historical two-rank pilots used one sample per rank. Preserve their
    # effective microbatch; newer single-GPU pilots retain their screened size.
    if "torch.distributed.run" in original:
        replace_option(command, "--batch-size", 2)
    if variant == "lower-lr":
        replace_option(command, option, value)
    for flag, expected in (
        ("--input-batch-tokens", "16384"),
        ("--ce-tokens", "20000000"),
        ("--sequence-length", "512"),
        ("--optimizer", "auto"),
    ):
        if command[command.index(flag) + 1] != expected:
            raise ValueError(f"reference pilot has an unexpected {flag}")
    return command


def training_environment(workspace, source, uuid):
    env = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        MINIFRONTIER_MIN_FREE_GIB="50",
        HF_HOME=str(workspace / "data/hf-cache"),
        TRITON_CACHE_DIR=str(workspace / "outputs/kernel-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(workspace / "outputs/inductor-cache"),
        TMPDIR=str(workspace / "outputs/single-gpu-tmp"),
    )
    return env


def source_check(source, expected):
    code = "import json; from minifrontier.provenance import source_identity; print(json.dumps(source_identity()))"
    identity = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=source,
            env=dict(os.environ, PYTHONPATH=str(source)),
            text=True,
            timeout=60,
        )
    )
    if identity != expected or identity.get("dirty"):
        raise ValueError("frozen training source differs from the paired-GPU reference")


def prepare(workspace, source, output):
    workspace, source, output = workspace.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("queue output already exists; preserve the previous queue")
    require_space(workspace, 72 * GIB)
    jobs = []
    for model, (first_gpu, previous_name, option, value) in FAMILIES.items():
        previous = workspace / "outputs/strategy-recipe-pilots-v2" / previous_name
        pilot = read_json(previous / "pilot.json")
        source_check(source, pilot["source"])
        reference = next(t for t in pilot["trials"] if t["optimizer"] == "auto")
        for offset, variant in enumerate(("reference", "lower-lr")):
            destination = output / model / variant
            command = single_command(reference["command"], destination, variant, option, value)
            inputs = {}
            for flag in ("--config", "--token-mixture", "--media-mixture"):
                if flag in command:
                    path = command[command.index(flag) + 1]
                    inputs[path] = sha256(path)
            data = Path(command[command.index("--data") + 1])
            for filename in ("manifest.json", "tokenizer.json"):
                inputs[str(data / filename)] = sha256(data / filename)
            jobs.append(
                dict(
                    id=f"{model}/{variant}",
                    model=model,
                    variant=variant,
                    gpu_id=first_gpu + offset,
                    output=str(destination),
                    predecessor=str(previous / "pilot.json"),
                    source=pilot["source"],
                    reference_performance=str(Path(reference["output"]) / "performance.json"),
                    inputs=inputs,
                    command=command,
                )
            )
    output.mkdir(parents=True)
    plan = dict(
        schema_version=1,
        workspace=str(workspace),
        source_root=str(source),
        output=str(output),
        controller_sha256=sha256(__file__),
        main_budget_eligible=False,
        scope="single-GPU Muon reference and one lower learning rate per family; optimizer choice remains under evaluation",
        allowed_gpu_ids=list(range(8)),
        jobs=jobs,
    )
    write_json(output / "queue-plan.json", plan)
    print(json.dumps(dict(plan=str(output / "queue-plan.json"), jobs=len(jobs))))


def predecessor_ready(path):
    pilot = read_json(path)
    if pilot.get("stage") != COMPLETE:
        return False
    trials = pilot.get("trials", [])
    return {t.get("optimizer") for t in trials} == {"auto", "adamw"} and all(
        read_json(Path(t["output"]) / "status.json").get("state") == "complete"
        and (Path(t["output"]) / "model.pt").is_file()
        for t in trials
    )


def gpu_ready(gpu):
    return (
        gpu is not None
        and gpu.used_mib < 1024
        and gpu.free_gib >= 22
        and gpu.utilization_percent <= 5
    )


def busy_gpu_uuids():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {line.strip() for line in output.splitlines() if line.strip()}


def execute(plan_path):
    plan = read_json(plan_path)
    if plan.get("controller_sha256") != sha256(__file__):
        raise ValueError("queue controller changed after preparation")
    workspace, source, output = (Path(plan[k]) for k in ("workspace", "source_root", "output"))
    ids = [job["gpu_id"] for job in plan["jobs"]]
    allowed = plan.get("allowed_gpu_ids", list(range(8)))
    if len(ids) != len(set(ids)) or not set(ids) <= set(allowed):
        raise ValueError("queue requires distinct GPUs from the authorized host pool")
    (workspace / "outputs/single-gpu-tmp").mkdir(exist_ok=True)
    lock_root = workspace / "outputs/gpu-locks"
    lock_root.mkdir(exist_ok=True)
    status_path = output / "queue.json"
    # Restarting a running queue needs an explicit checkpoint recovery plan.
    if status_path.exists():
        raise FileExistsError("queue has already started; refusing to duplicate running jobs")
    state = dict(
        pid=os.getpid(),
        started_at=time.time(),
        plan_sha256=sha256(plan_path),
        main_budget_eligible=False,
        jobs=[
            dict(id=j["id"], gpu_id=j["gpu_id"], output=j["output"], state="waiting_predecessor")
            for j in plan["jobs"]
        ],
    )
    running: dict[str, tuple[subprocess.Popen[bytes], TextIO, TextIO]] = {}
    while True:
        devices = {gpu.index: gpu for gpu in query_gpus()}
        busy = busy_gpu_uuids()
        for job, status in zip(plan["jobs"], state["jobs"], strict=True):
            name = job["id"]
            if name in running:
                process, lease, log = running[name]
                code = process.poll()
                if code is not None:
                    result = read_json(Path(job["output"]) / "status.json")
                    status.update(
                        state="complete"
                        if code == 0 and result.get("state") == "complete"
                        else "failed",
                        exit_code=code,
                        finished_at=time.time(),
                    )
                    log.close()
                    lease.close()
                    del running[name]
                continue
            if status["state"] in {"complete", "failed"}:
                continue
            if not predecessor_ready(job["predecessor"]):
                status["state"] = "waiting_predecessor"
                continue
            gpu = devices.get(job["gpu_id"])
            if gpu is None or not gpu_ready(gpu) or busy is None or gpu.uuid in busy:
                status["state"] = "waiting_gpu"
                continue
            lease = (lock_root / f"gpu-{job['gpu_id']}.lock").open("a")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                status["state"] = "waiting_gpu_lease"
                continue
            try:
                require_space(output, 12 * GIB)
            except StorageLimitError as error:
                lease.close()
                status.update(state="waiting_disk", reason=str(error))
                continue
            try:
                source_check(source, job["source"])
                if any(sha256(path) != digest for path, digest in job["inputs"].items()):
                    raise ValueError("reference data, tokenizer, mixture or config changed")
                destination = Path(job["output"])
                if destination.exists():
                    raise FileExistsError("trial output exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                log = destination.with_suffix(".log").open("x")
                try:
                    process = subprocess.Popen(
                        job["command"],
                        cwd=source,
                        env=training_environment(workspace, source, gpu.uuid),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                running[name] = process, lease, log
                status.update(
                    state="running", pid=process.pid, gpu_uuid=gpu.uuid, started_at=time.time()
                )
            except Exception as error:
                lease.close()
                status.update(state="failed", error=f"{type(error).__name__}: {error}")
        state["updated_at"] = time.time()
        write_json(status_path, state)
        if all(j["state"] in {"complete", "failed"} for j in state["jobs"]):
            return 1 if any(j["state"] == "failed" for j in state["jobs"]) else 0
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("prepare")
    build.add_argument("--workspace", type=Path, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.workspace, args.source_root, args.output)
    else:
        raise SystemExit(execute(args.plan))


if __name__ == "__main__":
    main()
