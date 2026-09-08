"""Launch three independent two-GPU pipelines, with durable logs and exact resumes.

Default device groups use only 0-5. Each controller runs stages sequentially and
stops on failure. Existing checkpoints resume only with the original recipe.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def budget_steps(manifest, stage, sequence_length, global_batch, *, steps=None, epochs=None):
    """Resolve an explicit sample-coverage budget; never imply a fluency guarantee."""
    if (steps is None) == (epochs is None) or global_batch < 1 or sequence_length < 2:
        raise ValueError(f"set exactly one explicit steps or epochs budget for {stage}")
    record = manifest["stages"][stage]["train"]
    examples = (
        (record["supervised_tokens"] - sequence_length) // (sequence_length - 1) + 1
        if stage == "pretrain"
        else record["examples"]
    )
    if examples < 1 or (epochs is not None and (not math.isfinite(epochs) or epochs <= 0)):
        raise ValueError("empty dataset or invalid epoch budget")
    updates = math.ceil(epochs * examples / global_batch) if epochs is not None else steps
    if updates is None or updates < 0:
        raise ValueError("steps must be nonnegative")
    return updates, dict(
        dataset_examples=examples,
        examples_seen=updates * global_batch,
        epochs=updates * global_batch / examples,
        predicted_tokens=updates * global_batch * (sequence_length - 1)
        if stage == "pretrain"
        else None,
    )


def controller(recipe_path):
    recipe_path = Path(recipe_path).resolve()
    controller_lock = (recipe_path.parent / ".controller.lock").open("a")
    try:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("a controller already owns this run") from error
    recipe = json.loads(recipe_path.read_text())
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=recipe["gpus"],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    previous = None
    for stage, steps in recipe["stages"]:
        directory = recipe_path.parent / stage
        directory.mkdir(exist_ok=True)
        status = directory / "status.json"
        checkpoint = directory / "checkpoint.pt"
        if (
            status.exists()
            and json.loads(status.read_text())["state"] == "complete"
            and (directory / "model.pt").exists()
        ):
            previous = directory / "model.pt"
            continue
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(recipe['gpus'].split(','))}",
            "-m",
            "minifrontier.training.train",
            "--model",
            recipe["model"],
            "--data",
            recipe["data"],
            "--output",
            str(directory),
            "--stage",
            stage,
            "--steps",
            str(steps),
            "--sequence-length",
            str(
                recipe.get("indexer_sequence_length", recipe["sequence_length"])
                if stage in {"dense_distill", "sparse_cpt"}
                else recipe["sequence_length"]
            ),
            "--batch-size",
            str(recipe["batch_size"]),
            "--grad-accum",
            str(recipe["grad_accum"]),
            "--save-every",
            str(recipe["save_every"]),
            "--eval-every",
            str(recipe["eval_every"]),
            "--eval-batches",
            str(recipe.get("eval_batches", 64)),
            "--log-every",
            "5",
            "--lr",
            str(recipe["lr"] if stage in {"pretrain", "sparse_cpt"} else recipe["lr"] / 3),
            "--warmup-steps",
            str(min(50, max(1, steps // 10))),
            "--run-kind",
            recipe["run_kind"],
        ]
        if recipe.get("config"):
            command += ["--config", recipe["config"]]
        if checkpoint.exists():
            command += ["--resume", str(checkpoint)]
        elif previous:
            command += ["--init", str(previous)]
        with (directory / "train.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state = dict(
                state="running",
                stage=stage,
                pid=process.pid,
                command=command,
                controller_pid=os.getpid(),
                started_at=time.time(),
            )
            (recipe_path.parent / "pipeline.json").write_text(json.dumps(state, indent=2))
            code = process.wait()
        if code:
            state.update(state="failed", returncode=code, ended_at=time.time())
            (recipe_path.parent / "pipeline.json").write_text(json.dumps(state, indent=2))
            raise SystemExit(code)
        previous = directory / "model.pt"
    (recipe_path.parent / "pipeline.json").write_text(
        json.dumps(
            dict(
                state="complete",
                checkpoint=str(previous),
                ended_at=time.time(),
                controller_pid=os.getpid(),
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller")
    p.add_argument("--data", default="data/educational-v1")
    p.add_argument("--run", default="educational-v1")
    p.add_argument("--models", nargs="+", default=["miniqwen4", "minikimik3", "minideepseekv4"])
    p.add_argument("--gpu-groups", nargs="+", default=["0,1", "2,3", "4,5"])
    pretrain_budget = p.add_mutually_exclusive_group()
    pretrain_budget.add_argument("--pretrain-steps", type=int)
    pretrain_budget.add_argument("--pretrain-epochs", type=float)
    p.add_argument("--distill-steps", type=int, default=100)
    p.add_argument("--cpt-steps", type=int, default=200)
    sft_budget = p.add_mutually_exclusive_group()
    sft_budget.add_argument("--sft-steps", type=int)
    sft_budget.add_argument("--sft-epochs", type=float)
    p.add_argument(
        "--dpo-steps",
        type=int,
        default=0,
        help="optional preference stage; first review SFT generations and held-out loss",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--indexer-sequence-length", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--run-kind", choices=["educational", "acceptance"], default="educational")
    args = p.parse_args()
    if args.controller:
        controller(args.controller)
        return
    if (args.pretrain_steps is None and args.pretrain_epochs is None) or (
        args.sft_steps is None and args.sft_epochs is None
    ):
        p.error(
            "set explicit --pretrain-steps/--pretrain-epochs and --sft-steps/--sft-epochs; small step counts are execution checks, not a trained chat model"
        )
    if min(args.batch_size, args.grad_accum) < 1 or args.eval_batches < 0:
        p.error("batch size and accumulation must be positive; eval-batches must be nonnegative")
    if len(args.models) != len(args.gpu_groups):
        raise ValueError("provide one GPU group per model")
    devices = [int(d) for group in args.gpu_groups for d in group.split(",")]
    if len(set(devices)) != len(devices):
        raise ValueError("GPU groups must not overlap")
    if not (ROOT / args.data / "manifest.json").is_file():
        raise FileNotFoundError("prepare the corpus before launching training")
    manifest = json.loads((ROOT / args.data / "manifest.json").read_text())
    # Check actual occupancy; never kill another job or wait indefinitely.
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    occupancy = {int(row.split(",")[0]): int(row.split(",")[1]) for row in query.splitlines()}
    if any(occupancy.get(d, 999999) > 1024 for d in devices):
        raise RuntimeError(f"requested GPUs are occupied: {occupancy}")
    for name, group in zip(args.models, args.gpu_groups, strict=True):
        global_batch = len(group.split(",")) * args.batch_size * args.grad_accum
        pretrain_steps, pretrain_coverage = budget_steps(
            manifest,
            "pretrain",
            args.sequence_length,
            global_batch,
            steps=args.pretrain_steps,
            epochs=args.pretrain_epochs,
        )
        sft_steps, sft_coverage = budget_steps(
            manifest,
            "sft",
            args.sequence_length,
            global_batch,
            steps=args.sft_steps,
            epochs=args.sft_epochs,
        )
        directory = ROOT / "outputs" / name / args.run
        directory.mkdir(parents=True, exist_ok=True)
        recipe_path = directory / "recipe.json"
        stages = [("pretrain", pretrain_steps)]
        if name != "minikimik3":
            stages += [("dense_distill", args.distill_steps), ("sparse_cpt", args.cpt_steps)]
        stages += [("sft", sft_steps), ("dpo", args.dpo_steps)]
        stages = [(stage, count) for stage, count in stages if count > 0]
        recipe = dict(
            model=name,
            gpus=group,
            stages=stages,
            data=str((ROOT / args.data).resolve()),
            sequence_length=args.sequence_length,
            indexer_sequence_length=args.indexer_sequence_length,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            coverage=dict(pretrain=pretrain_coverage, sft=sft_coverage),
            lr=args.lr,
            run_kind=args.run_kind,
        )
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != json.loads(
            json.dumps(recipe)
        ):
            raise ValueError(f"existing run has a different recipe: {recipe_path}")
        recipe_path.write_text(json.dumps(recipe, indent=2))
        with (directory / "controller.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--controller", str(recipe_path)],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            json.dumps(
                dict(model=name, gpus=group, controller_pid=process.pid, recipe=str(recipe_path))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
