"""Run one bounded MF1 mechanism trial with a CUDA memory ceiling and an owned-child guard."""

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3


def gpu_status(index):
    raw = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        .strip()
        .split(",")
    )
    return dict(uuid=raw[0].strip(), free_gib=int(raw[1]) / 1024, utilization=int(raw[2]))


def admission(free_gib, disk_free_gib, memory_gib):
    # Include CUDA context overhead and keep a reserve above the older queue's 3 GiB guard.
    return free_gib >= memory_gib + 1 + 5 and disk_free_gib >= 50 + 16


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(args):
    import torch

    from minifrontier.training.minifrontier1 import train

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_gib * GIB / torch.cuda.get_device_properties(0).total_memory, 0
    )
    result = train(
        data=args.data,
        output=args.output,
        config=str(Path(args.source_checkout) / "configs/minifrontier1/model_228m_native.json"),
        device="cuda:0",
        phase="pilot",
        run_kind="acceptance",
        seed=42,
        steps=10000,
        token_budget=args.token_budget,
        input_batch_tokens=args.input_batch_tokens,
        optimizer_kind="adamw",
        lr=args.lr,
        vision_lr=1e-4,
        save_every=25,
        eval_every=25,
        weights={
            "zh_general": 0.45,
            "en_general": 0.35,
            "math": 0.18,
            "vision": 0.015,
            "video": 0.005,
        },
        stop_after_updates=2 if not args.resume else None,
        resume=str(Path(args.output) / "checkpoint.pt") if args.resume else None,
    )
    print(json.dumps(result), flush=True)


def supervise(args):
    source, data, output = (
        Path(p).resolve() for p in (args.source_checkout, args.data, args.output)
    )
    if not (source / ".git").exists():
        raise ValueError("use a frozen Git checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise ValueError("the frozen source checkout must be clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if not (data / "manifest.json").is_file():
        raise ValueError("prepare and freeze the mechanism data before launching")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "supervisor.json").exists():
        raise FileExistsError("supervisor already ran; inspect its state before explicit recovery")
    status = dict(
        state="waiting_resources",
        pid=os.getpid(),
        source_commit=commit,
        gpu=args.gpu,
        memory_ceiling_gib=args.memory_gib,
        physical_reserve_gib=5,
        disk_reserve_gib=50,
        token_budget=args.token_budget,
        lr=args.lr,
        scope="shared-GPU 228M mechanism experiment; not formal pretraining",
        started_at=time.time(),
    )
    write(output / "supervisor.json", status)
    gpu = gpu_status(args.gpu)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu["uuid"],
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        MINIFRONTIER_MIN_FREE_GIB="50",
        PYTHONUNBUFFERED="1",
    )
    command = [
        sys.executable,
        str(source / "scripts/run_mf1_trial.py"),
        "--worker",
        "--source-checkout",
        str(source),
        "--data",
        str(data),
        "--output",
        str(output),
        "--gpu",
        str(args.gpu),
        "--lr",
        str(args.lr),
        "--memory-gib",
        str(args.memory_gib),
        "--token-budget",
        str(args.token_budget),
        "--input-batch-tokens",
        str(args.input_batch_tokens),
    ]
    write(
        output / "launch.json",
        dict(
            command=command,
            source_commit=commit,
            source_checkout=str(source),
            gpu_uuid=gpu["uuid"],
            same_seed_and_data_as_other_lr=True,
            notes=[
                "2-update checkpoint/reload before continuous training",
                "only this supervisor's child is stopped by the resource guard",
            ],
        ),
    )
    child = None
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for resume in (False, True):
            while not admission(
                gpu["free_gib"], shutil.disk_usage(output).free / GIB, args.memory_gib
            ):
                if interrupted:
                    status["state"] = "stopped"
                    return
                time.sleep(5)
                gpu = gpu_status(args.gpu)
            with (output / "worker.log").open("a") as log:
                child = subprocess.Popen(
                    command + (["--resume"] if resume else []),
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                status.update(
                    state="running" if resume else "checking_resume",
                    child_pid=child.pid,
                    gpu_uuid=gpu["uuid"],
                    phase_started_at=time.time(),
                )
                write(output / "supervisor.json", status)
                while child.poll() is None:
                    try:
                        gpu = gpu_status(args.gpu)
                        reason = (
                            "stopped_memory"
                            if gpu["free_gib"] < 5
                            else "stopped_disk"
                            if shutil.disk_usage(output).free / GIB < 60
                            else "stopped"
                            if interrupted
                            else None
                        )
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        reason = "stopped_monitor_error"
                        status["error"] = str(error)
                    if reason:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        status.update(state=reason, exit_code=child.returncode)
                        return
                    status.update(free_device_gib=gpu["free_gib"], updated_at=time.time())
                    write(output / "supervisor.json", status)
                    time.sleep(2)
            code = child.returncode
            if code != 0:
                status.update(state="failed", exit_code=code)
                return
            progress = json.loads((output / "status.json").read_text())
            if not resume:
                if progress["state"] != "paused" or progress["step"] != 2:
                    raise RuntimeError("two-update checkpoint precheck did not pause as expected")
                status["checkpoint_reload_started_at"] = time.time()
            else:
                if (
                    progress["state"] != "budget_complete_unqualified"
                    or progress["ledger"]["ce_tokens"] < args.token_budget
                ):
                    raise RuntimeError(
                        "trial exited without consuming the declared mechanism budget"
                    )
                status.update(state="complete", ledger=progress["ledger"], finished_at=time.time())
            gpu = gpu_status(args.gpu)
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write(output / "supervisor.json", status)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--token-budget", type=int, default=500000)
    parser.add_argument("--input-batch-tokens", type=int, default=1024)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not 0 < args.token_budget <= 2000000
        or not 0 < args.memory_gib < 16
        or args.lr <= 0
        or not math.isfinite(args.lr)
        or args.gpu < 0
        or args.input_batch_tokens < 1
    ):
        raise ValueError("invalid bounded mechanism experiment controls")
    worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    main()
