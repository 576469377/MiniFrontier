"""Single-card/DDP draft adaptation after a selected, frozen final target.

Target-generated trajectories stay in memory. Checkpoints contain only draft,
optimizer, per-rank RNG and cursor state, and immutable target/data/source hashes.
The local AdamW recipe and replay-based sampler require their own quality/profile.
"""

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from minifrontier.data import StageDataset, sha256
from minifrontier.inference.runtime import load_checkpoint
from minifrontier.provenance import source_identity
from minifrontier.storage import require_space, reserve_write
from minifrontier.training.drafts import DraftObjective, build_draft, target_trajectory
from minifrontier.training.runtime import (
    BatchCursor,
    atomic_save,
    restore_rng,
    rng_state,
    validation_indices,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draft-positions", type=int, required=True)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--rollout-tokens", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--steps", type=int, default=1000000, help="safety limit; never implies completion"
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-positions", type=int, default=10000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--profile-warmup", type=int, default=50)
    p.add_argument("--profile-updates", type=int, default=200)
    p.add_argument("--min-device-free-gib", type=float, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-kind", choices=["acceptance", "strategy"], default="acceptance")
    p.add_argument("--strategy-plan")
    p.add_argument("--strategy-phase")
    p.add_argument("--strategy-evidence")
    p.add_argument("--config")
    return p


def atomic_json(value, path):
    raw = json.dumps(value, indent=2).encode()
    with reserve_write(path, len(raw)):
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def main(argv=None):
    args = parser().parse_args(argv)
    for key in (
        "draft_positions",
        "sequence_length",
        "rollout_tokens",
        "grad_accum",
        "steps",
        "save_every",
        "eval_every",
        "eval_examples",
        "profile_updates",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.rollout_tokens < 6 or args.profile_warmup < 0 or args.warmup_positions < 0:
        raise ValueError("rollout needs at least six positions; warmups cannot be negative")
    if args.lr <= 0 or args.clip_grad <= 0 or args.weight_decay < 0 or args.min_device_free_gib < 0:
        raise ValueError("invalid optimizer or device reserve settings")
    rank, world, local = (
        int(os.getenv(key, default))
        for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    device = torch.device(
        f"cuda:{local}" if world > 1 and args.device.startswith("cuda") else args.device
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        run(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, rank, world, device):
    output, data = Path(args.output).resolve(), Path(args.data).resolve()
    require_space(output, 0)
    occupied = torch.tensor(
        int(rank == 0 and output.exists() and any(output.iterdir()) and not args.resume),
        device=device,
    )
    if world > 1:
        dist.broadcast(occupied, src=0)
    if occupied:
        raise FileExistsError("draft output is not empty; use a new path or exact resume")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    target, _tokenizer, meta = load_checkpoint(args.target, device)
    if _tokenizer.token_to_id("<|image|>") != 7:
        raise ValueError("drafts require the frozen strategy tokenizer mapping")
    family = meta["model_name"]
    manifest = json.loads((data / "manifest.json").read_text())
    tokenizer_hash = sha256(Path(args.target).resolve().parent / "tokenizer.json")
    if (
        manifest["tokenizer"]["sha256"] != tokenizer_hash
        or sha256(data / "tokenizer.json") != tokenizer_hash
    ):
        raise ValueError("draft, target and dataset must share the exact frozen tokenizer")
    if manifest.get("family", family) != family:
        raise ValueError("native draft corpus uses a different model processor")
    if args.sequence_length > getattr(
        target.config, "max_position_embeddings", getattr(target.config, "max_seq_len", 0)
    ):
        raise ValueError("draft sequence bucket exceeds target context")
    train = StageDataset(data, "sft", "train", args.sequence_length)
    val = StageDataset(data, "sft", "val", args.sequence_length)
    if not len(train) or not len(val):
        raise ValueError("draft adaptation requires nonempty independent SFT train/val prompts")
    source = source_identity()
    target_hash = sha256(args.target)
    spec = dict(
        kind=args.run_kind,
        model_name=family,
        target_sha256=target_hash,
        target_stage=meta["stage"],
        tokenizer_sha256=tokenizer_hash,
        data_sha256=sha256(data / "manifest.json"),
        source=source,
        world_size=world,
        phase=getattr(target, "training_phase", getattr(target, "phase", "dense_pretrain")),
        config=vars(target.config),
        sequence_length=args.sequence_length,
        rollout_tokens=args.rollout_tokens,
        grad_accum=args.grad_accum,
        draft_position_budget=args.draft_positions,
        lr=args.lr,
        optimizer="local-adamw-draft-v1",
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        warmup_positions=args.warmup_positions,
        performance_profile=dict(warmup=args.profile_warmup, updates=args.profile_updates),
        min_device_free_gib=args.min_device_free_gib,
        eval_examples=args.eval_examples,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    # Nested vision dataclasses must be serialized canonically too.
    from dataclasses import asdict

    spec["config"] = asdict(target.config)
    if args.run_kind == "strategy":
        admit(args, spec, output, data)
    draft = build_draft(family, target)
    target_versions = tuple(p._version for p in target.parameters())
    objective = DraftObjective(family, draft)
    params = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    wrapped: Any = (
        DDP(
            objective,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if world > 1
        else objective
    )
    cursor = BatchCursor(len(train), args.seed, rank, world)
    positions = step = skipped = 0
    best = float("inf")
    profile: list[dict] = []
    if args.resume:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["run_spec"] != spec:
            raise ValueError("draft resume target/data/source/recipe differs")
        draft.load_state_dict(saved["draft"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        cursor.offset, step, positions = (
            saved["data_offset"],
            saved["step"],
            saved["draft_positions"],
        )
        skipped, best, profile = saved["skipped"], saved["best"], saved["profile"]
        restore_rng(saved["rng"][rank], device)
    else:
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
    if rank == 0:
        atomic_json(spec, output / "run.json")

    def record(row):
        if rank == 0:
            raw = json.dumps(row) + "\n"
            with (
                reserve_write(output / "metrics.jsonl", len(raw.encode())),
                (output / "metrics.jsonl").open("a") as stream,
            ):
                stream.write(raw)
            print(raw.rstrip(), flush=True)

    def artifact():
        return dict(
            format="minifrontier-draft-v1",
            model_name=family,
            target_sha256=target_hash,
            tokenizer_sha256=tokenizer_hash,
            draft=draft.state_dict(),
            step=step,
            draft_positions=positions,
            run_spec=spec,
            capability_status="unassessed",
        )

    def save():
        if any(
            p.requires_grad or p.grad is not None for p in target.parameters()
        ) or target_versions != tuple(p._version for p in target.parameters()):
            raise RuntimeError("frozen target changed during draft adaptation")
        if world > 1 and positions >= args.draft_positions:
            disagreements = torch.zeros((), device=device)
            for parameter in draft.parameters():
                for chunk in parameter.detach().flatten().split(1024 * 1024):
                    reference = chunk.clone()
                    dist.broadcast(reference, src=0)
                    disagreements += chunk.ne(reference).sum()
            dist.all_reduce(disagreements)
            if disagreements:
                raise RuntimeError("draft parameters differ across DDP ranks")
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state(device))
        else:
            states = [rng_state(device)]
        if rank == 0:
            atomic_save(
                dict(
                    artifact(),
                    optimizer=optimizer.state_dict(),
                    data_offset=cursor.offset,
                    rng=states,
                    skipped=skipped,
                    best=best,
                    profile=profile,
                ),
                output / "checkpoint.pt",
            )
            atomic_save(artifact(), output / "draft.pt")
            atomic_json(
                dict(
                    state="complete" if positions >= args.draft_positions else "incomplete",
                    step=step,
                    draft_positions=positions,
                    budget=args.draft_positions,
                    capability_status="unassessed",
                ),
                output / "status.json",
            )

    def validate():
        nonlocal best
        state = rng_state(device)
        torch.manual_seed(args.seed + 10000 + rank)
        draft.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        try:
            with torch.no_grad():
                for index in validation_indices(
                    len(val), limit=args.eval_examples, seed=args.seed, rank=rank, world_size=world
                ):
                    trajectory, length, media = target_trajectory(
                        target, val[index], rollout_tokens=args.rollout_tokens
                    )
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        loss, count, normalizer = objective(trajectory, length, media)
                    totals += torch.tensor(
                        [float(loss) * normalizer, normalizer, count],
                        device=device,
                        dtype=torch.float64,
                    )
            if world > 1:
                dist.all_reduce(totals)
            if totals[1] <= 0:
                raise ValueError("target produced no evaluable held-out draft positions")
            value = float(totals[0] / totals[1])
            record(
                dict(event="validation", step=step, objective=value, draft_positions=int(totals[2]))
            )
            if not math.isfinite(value):
                raise ValueError("nonfinite held-out draft objective")
            if value < best:
                best = value
                if rank == 0:
                    atomic_save(
                        dict(artifact(), validation_objective=value), output / "best-draft.pt"
                    )
        finally:
            restore_rng(state, device)
            draft.train()

    if not args.resume:
        validate()
    consecutive_empty = 0
    while positions < args.draft_positions and step < args.steps:
        started = time.monotonic()
        io_seconds = 0.0
        media_totals = torch.zeros(2, device=device, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for micro in range(args.grad_accum):
            io_started = time.monotonic()
            sample = train[cursor.next()[0]]
            io_seconds += time.monotonic() - io_started
            trajectory, length, media = target_trajectory(
                target, sample, rollout_tokens=args.rollout_tokens
            )
            media_totals += torch.tensor(
                [
                    sum(s.get("resource_kind", "image") == "image" for s in media),
                    sum(s.get("source_frame_count", 0) for s in media),
                ],
                device=device,
                dtype=torch.float64,
            )
            sync = (
                wrapped.no_sync()
                if world > 1 and micro + 1 < args.grad_accum
                else contextlib.nullcontext()
            )
            with (
                sync,
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                loss, count, normalizer = wrapped(trajectory, length, media)
                # DSpark's position-decayed weights have a different denominator
                # from its integer budget count; aggregate both independently.
                (loss * normalizer).backward()
            totals += torch.tensor(
                [float(loss.detach()) * normalizer, normalizer, count],
                device=device,
                dtype=torch.float64,
            )
        if world > 1:
            dist.all_reduce(totals)
            dist.all_reduce(media_totals)
        count = int(totals[2])
        if count == 0:
            consecutive_empty += 1
            skipped += 1
            if consecutive_empty >= 32:
                raise ValueError(
                    "target yielded 32 empty draft windows; no optimizer decay applied"
                )
            continue
        consecutive_empty = 0
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.mul_(world / totals[1])
        norm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad, error_if_nonfinite=True)
        warm = min(1.0, (positions + count) / max(1, args.warmup_positions))
        progress = min(1.0, positions / args.draft_positions)
        lr = args.lr * warm * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1
        positions += count
        peak_allocated = peak_reserved = 0.0
        free_gib: Any = float("inf")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            free, _ = torch.cuda.mem_get_info(device)
            free_gib = torch.tensor(free / 1024**3, device=device)
            if world > 1:
                dist.all_reduce(free_gib, op=dist.ReduceOp.MIN)
            if free_gib < args.min_device_free_gib:
                save()
                raise RuntimeError("draft training violated the reserved GPU memory margin")
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        elapsed = torch.tensor(time.monotonic() - started, device=device, dtype=torch.float64)
        extra_performance = torch.tensor(
            [io_seconds, peak_allocated, peak_reserved], device=device, dtype=torch.float64
        )
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(extra_performance, op=dist.ReduceOp.MAX)
        row = dict(
            event="train",
            step=step,
            draft_positions=positions,
            window_positions=count,
            loss=float(totals[0] / totals[1]),
            loss_normalizer=float(totals[1]),
            grad_norm=float(norm),
            lr=lr,
            seconds=float(elapsed),
            io_seconds=float(extra_performance[0]),
            peak_allocated_gib=float(extra_performance[1]),
            peak_reserved_gib=float(extra_performance[2]),
            min_device_free_gib=float(free_gib) if device.type == "cuda" else None,
            images=int(media_totals[0]),
            frames=int(media_totals[1]),
        )
        record(row)
        if args.profile_warmup < step <= args.profile_warmup + args.profile_updates:
            profile.append(row)
            if len(profile) == args.profile_updates and rank == 0:
                atomic_json(
                    dict(
                        recipe=spec,
                        measured_updates=len(profile),
                        updates=profile,
                        includes=[
                            "data",
                            "target_rollout",
                            "target_features",
                            "draft_forward",
                            "backward",
                            "optimizer",
                            "ddp",
                        ],
                        images_per_second=sum(r["images"] for r in profile)
                        / sum(r["seconds"] for r in profile),
                        frames_per_second=sum(r["frames"] for r in profile)
                        / sum(r["seconds"] for r in profile),
                        draft_positions_per_second=sum(r["window_positions"] for r in profile)
                        / sum(r["seconds"] for r in profile),
                    ),
                    output / "performance.json",
                )
        if step % args.eval_every == 0 or positions >= args.draft_positions:
            validate()
        if step % args.save_every == 0 or positions >= args.draft_positions:
            save()
    save()


def admit(args, spec, output, data):
    from minifrontier.training.strategy_gate import check, validate_runtime

    if not all((args.strategy_plan, args.strategy_phase, args.strategy_evidence, args.config)):
        raise ValueError("formal draft training requires plan, phase, evidence and target config")
    report = check(
        args.strategy_plan,
        args.strategy_phase,
        args.strategy_evidence,
        data=data,
        config=args.config,
        output=output,
    )
    phase, errors = report["phase"], report["errors"]
    validate_runtime(args.strategy_plan, args.strategy_phase, spec["phase"], spec["world_size"])
    if spec["model_name"] != report["model"]:
        errors.append("draft model differs from strategy")
    requested = json.loads(Path(args.config).read_text())
    if any(spec["config"].get(key) != value for key, value in requested.items()):
        errors.append("draft target configuration differs from the admitted configuration")
    if (
        phase["objective"] != "draft_positions"
        or not phase["budget_min"] <= args.draft_positions <= phase["budget_max"]
    ):
        errors.append("draft positions must match the planned 10M-30M phase")
    if (
        args.sequence_length not in phase["sequence_lengths"]
        or spec["phase"] != phase["attention_phase"]
    ):
        errors.append("draft context/attention phase differs from the plan")
    expected_stages = {
        "minikimik3": {"mopd"},
        "minideepseekv4": {"opd"},
        "miniqwen4": {"sft", "grpo", "dpo"},
    }
    if spec["target_stage"] not in expected_stages[spec["model_name"]]:
        errors.append("draft target is not the selected final posttraining stage")
    evidence = json.loads(Path(args.strategy_evidence).read_text())
    if not any(
        value.get("checkpoint_sha256") == spec["target_sha256"]
        for key, value in evidence.get("completed_phases", {}).items()
        if key in phase["depends_on"]
    ):
        errors.append("draft target is not the exact quality-selected predecessor")
    profile_path = Path(evidence.get("performance", ""))
    measured = (
        json.loads(profile_path.read_text()).get("recipe", {}) if profile_path.is_file() else {}
    )
    for key in (
        "target_sha256",
        "world_size",
        "grad_accum",
        "rollout_tokens",
        "sequence_length",
        "optimizer",
        "lr",
        "weight_decay",
    ):
        if measured.get(key) != spec[key]:
            errors.append(f"draft profile differs in {key}")
    if errors:
        raise ValueError("strategy draft phase blocked:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
