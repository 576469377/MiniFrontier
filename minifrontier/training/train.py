"""Single GPU / replicated DDP training with stage-aware resume and evaluation."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from minifrontier.data import StageDataset, sha256
from minifrontier.models.factory import build_model, configure_posttraining
from minifrontier.training.budgets import TokenLedger, scaled_ce, window_counts
from minifrontier.training.posttrain import dpo_loss
from minifrontier.training.runtime import (
    BatchCursor,
    RouterBalance,
    atomic_save,
    restore_rng,
    rng_state,
    validation_indices,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["miniqwen4", "minikimik3", "minideepseekv4"])
    p.add_argument("--config")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage",
        choices=["pretrain", "sft", "dpo", "dense_distill", "sparse_cpt", "grpo", "mopd"],
        default="pretrain",
    )
    p.add_argument("--steps", type=int, help="explicit update budget or token-budget safety limit")
    tokens = p.add_mutually_exclusive_group()
    tokens.add_argument("--ce-tokens", type=int, help="actual next-token supervision budget")
    tokens.add_argument(
        "--input-tokens", type=int, help="valid input-position budget for frozen indexer training"
    )
    p.add_argument("--warmup-tokens", type=int)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--router-bias-rate", type=float, default=0.001)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument(
        "--eval-batches",
        type=int,
        default=64,
        help="uniformly sampled validation examples per rank; 0 evaluates the whole split",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-mixture", help="JSON domain-to-token proportions; all domains required")
    p.add_argument("--max-data-epochs", type=int, help="cap per-domain reuse in a token mixture")
    p.add_argument("--profile-warmup", type=int, default=50)
    p.add_argument("--profile-updates", type=int, default=200)
    p.add_argument("--min-device-free-gib", type=float, default=0)
    p.add_argument("--init", help="weights/checkpoint from the preceding stage")
    p.add_argument("--resume", help="exact checkpoint to resume, including optimizer/data/RNG")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "qwen_muon", "kimi_muon", "deepseek_muon"],
        default="auto",
    )
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--rl-data", help="directory made by prepare-rl-data")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rollout-tokens", type=int, default=32)
    p.add_argument("--teacher-map", help="JSON mapping domain:effort to local teacher checkpoint")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--run-kind", choices=["educational", "acceptance"], default="educational")
    return p


def batch(dataset, indices, device):
    rows = [dataset[i] for i in indices]
    from minifrontier.multimodal import collate

    return collate(rows, device)


def main(argv=None):
    args = parser().parse_args(argv)
    token_budget = args.ce_tokens or args.input_tokens
    if args.steps is None and token_budget is None:
        raise ValueError("set an explicit --steps, --ce-tokens or --input-tokens budget")
    if any(value is not None and value < 1 for value in (args.ce_tokens, args.input_tokens)):
        raise ValueError("token budgets must be positive")
    if token_budget is not None and args.stage in {"dpo", "grpo", "mopd"}:
        raise ValueError("CE/input budgets do not describe preference or RL response budgets")
    if args.input_tokens is not None and args.stage != "dense_distill":
        raise ValueError("input-position budget is reserved for frozen indexer distillation")
    if args.ce_tokens is not None and args.stage == "dense_distill":
        raise ValueError("frozen indexer updates cannot be counted as main LM supervision")
    if args.steps is None:
        args.steps = token_budget
    if any(
        getattr(args, key) < 1
        for key in (
            "steps",
            "batch_size",
            "grad_accum",
            "sequence_length",
            "save_every",
            "eval_every",
            "log_every",
        )
    ):
        raise ValueError("step counts and capacities must be positive")
    if args.eval_batches < 0:
        raise ValueError("eval-batches must be nonnegative (0 means the whole validation split)")
    if args.stage in {"grpo", "mopd"}:
        if (
            not args.rl_data
            or args.group_size < 2
            or not 1 <= args.rollout_tokens < args.sequence_length
        ):
            raise ValueError("RL requires --rl-data, group size >= 2 and a valid rollout budget")
        if args.stage == "mopd" and not args.teacher_map:
            raise ValueError("MOPD requires --teacher-map")
    if args.resume and args.init:
        raise ValueError("--init and --resume are mutually exclusive")
    for key in ("lr", "muon_lr", "clip_grad"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=20)
        )
    try:
        run(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, rank, world, device):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data_dir, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and not args.resume:
        raise FileExistsError("output contains a checkpoint; use --resume or a new directory")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    data_hash = sha256(data_dir / "manifest.json")
    if sha256(data_dir / "tokenizer.json") != data_manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer file differs from data manifest")
    saved = (
        torch.load(args.resume or args.init, map_location="cpu", weights_only=True)
        if args.resume or args.init
        else None
    )
    phase = (
        args.stage
        if args.stage in {"dense_distill", "sparse_cpt"}
        else (saved.get("phase", "dense_pretrain") if saved else "dense_pretrain")
    )
    if saved and saved["model_name"] != args.model:
        raise ValueError("checkpoint belongs to a different model")
    if args.stage in {"sft", "dpo", "dense_distill", "sparse_cpt", "grpo", "mopd"} and not saved:
        raise ValueError("this stage requires --init or --resume from a trained checkpoint")
    if args.init:
        assert saved is not None
        allowed = {
            "dense_distill": {"pretrain"},
            "sparse_cpt": {"dense_distill"},
            "sft": {"pretrain", "sparse_cpt", "sft"},
            "dpo": {"sft", "dpo"},
            "grpo": {"sft", "dpo", "grpo"},
            "mopd": {"sft", "dpo", "grpo", "mopd"},
        }
        if args.stage in allowed and saved["stage"] not in allowed[args.stage]:
            raise ValueError(
                "invalid stage transition; follow pretrain -> indexer stages -> SFT -> DPO"
            )
    config = args.config or (saved["config"] if saved else None)
    model = build_model(args.model, config, phase=phase).to(device)
    if data_manifest["tokenizer"]["vocab_size"] > model.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model capacity")
    if args.sequence_length > getattr(
        model.config, "max_position_embeddings", getattr(model.config, "max_seq_len", 0)
    ):
        raise ValueError("training sequence length exceeds model context")
    if saved:
        if saved.get("tokenizer_sha256") != data_manifest["tokenizer"]["sha256"]:
            raise ValueError("checkpoint tokenizer differs from training corpus")
        if asdict(model.config) != asdict(type(model.config)(**saved["config"])):
            raise ValueError("checkpoint capacity differs from requested configuration")
        model.load_state_dict(saved["model"], strict=True)
    if args.stage in {"sft", "dpo", "grpo", "mopd"}:
        configure_posttraining(model)
    optimizer: torch.optim.Optimizer
    optimizer_kind = args.optimizer
    if optimizer_kind == "auto":
        optimizer_kind = {
            "miniqwen4": "qwen_muon",
            "minikimik3": "kimi_muon",
            "minideepseekv4": "deepseek_muon",
        }[args.model]
    if optimizer_kind == "qwen_muon":
        if args.model != "miniqwen4":
            raise ValueError("Qwen semantic optimizer only supports MiniQwen4")
        from .miniqwen4_optim import MiniQwen4Optimizer

        optimizer = MiniQwen4Optimizer(
            model, lr=args.muon_lr, adam_lr=args.lr, weight_decay=args.weight_decay
        )
    elif optimizer_kind in {"kimi_muon", "deepseek_muon"}:
        if optimizer_kind == "kimi_muon":
            if args.model != "minikimik3":
                raise ValueError("Kimi semantic optimizer only supports MiniKimi-K3")
            from .minikimik3_optim import MiniKimiK3Optimizer

            optimizer = MiniKimiK3Optimizer(
                model,
                lr=args.muon_lr,
                adam_lr=args.lr,
                weight_decay=args.weight_decay,
                eps=args.adam_eps,
            )
        else:
            if args.model != "minideepseekv4":
                raise ValueError("DeepSeek optimizer only supports MiniDeepSeek-V4")
            from .minideepseekv4_optim import MiniDeepSeekV4Optimizer

            optimizer = MiniDeepSeekV4Optimizer(
                model, lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps
            )
    else:
        decay: list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (
                    decay if parameter.ndim >= 2 and "ngram_embedding" not in name else no_decay
                ).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0},
            ],
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=args.adam_eps,
        )
    reference: Any = None
    if args.stage in {"dpo", "grpo"}:
        assert saved is not None
        reference = build_model(args.model, asdict(model.config), phase=phase).to(device).eval()
        reference.load_state_dict(
            saved["reference"] if args.resume and "reference" in saved else saved["model"]
        )
        configure_posttraining(reference)
        for parameter in reference.parameters():
            if parameter.is_floating_point():
                parameter.requires_grad_(False)
    dataset_stage = args.stage if args.stage in {"sft", "dpo"} else "pretrain"
    train: Any = StageDataset(data_dir, dataset_stage, "train", args.sequence_length)
    val: Any = StageDataset(data_dir, dataset_stage, "val", args.sequence_length)
    rollout = validation_rollout = None
    if args.stage in {"grpo", "mopd"}:
        from tokenizers import Tokenizer

        from .rollouts import RolloutObjective, TaskDataset

        tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
        train = TaskDataset(
            args.rl_data, "train", tokenizer, args.sequence_length - args.rollout_tokens
        )
        val = TaskDataset(
            args.rl_data, "val", tokenizer, args.sequence_length - args.rollout_tokens
        )
        rollout = RolloutObjective(
            model,
            reference,
            tokenizer,
            train,
            group_size=args.group_size,
            max_new_tokens=args.rollout_tokens,
            teacher_map=args.teacher_map,
            device=device,
        )
        # Share the same one-resident-teacher registry with validation.
        validation_rollout = RolloutObjective(
            model,
            reference,
            tokenizer,
            val,
            group_size=args.group_size,
            max_new_tokens=args.rollout_tokens,
        )
        validation_rollout.teachers = rollout.teachers
    run_spec = dict(
        model_name=args.model,
        config=asdict(model.config),
        phase=phase,
        stage=args.stage,
        optimizer=optimizer_kind,
        world_size=world,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        sequence_length=args.sequence_length,
        seed=args.seed,
        data_sha256=data_hash,
        tokenizer_sha256=data_manifest["tokenizer"]["sha256"],
        steps=args.steps,
        lr=args.lr,
        muon_lr=args.muon_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        router_bias_rate=args.router_bias_rate,
        ce_token_budget=args.ce_tokens,
        input_token_budget=args.input_tokens,
        warmup_tokens=args.warmup_tokens,
        schedule=args.schedule,
        adam_eps=args.adam_eps,
        normalization="global-accumulation-ce-sum-v2",
        token_mixture=json.loads(Path(args.token_mixture).read_text())
        if args.token_mixture
        else None,
        max_data_epochs=args.max_data_epochs,
        performance_profile=dict(
            warmup=args.profile_warmup,
            updates=args.profile_updates,
            min_device_free_gib=args.min_device_free_gib,
        ),
        router_balance="kimi-quantile-histogram-v1"
        if args.model == "minikimik3"
        else "source-sign-bias",
    )
    from minifrontier.provenance import source_identity

    run_spec["source"] = source_identity()
    if args.stage in {"grpo", "mopd"}:
        assert rollout is not None
        run_spec["rollout"] = dict(
            group_size=args.group_size,
            rollout_tokens=args.rollout_tokens,
            train_sha256=sha256(Path(args.rl_data) / "train.jsonl"),
            val_sha256=sha256(Path(args.rl_data) / "val.jsonl"),
            teacher_map=json.loads(Path(args.teacher_map).read_text())
            if args.teacher_map
            else None,
            teacher_sha256={
                path: sha256(path)
                for path in {rollout.teachers.path(key) for key in rollout.teachers}
            }
            if args.teacher_map
            else None,
        )
    first_step = 0
    offset = 0
    ledger = TokenLedger()
    if args.resume:
        assert saved is not None
        if saved["run_spec"] != run_spec:
            raise ValueError(
                "resume recipe differs (including schedule, batch, world size or corpus); use --init for a new run"
            )
        optimizer.load_state_dict(saved["optimizer"])
        first_step, offset = saved["step"], saved["data_offset"]
        restore_rng(saved["rng"][rank], device)
        ledger = TokenLedger(**saved["token_ledger"])
    cursor: Any
    if args.token_mixture:
        from .mixture import TokenMixtureCursor

        cursor = TokenMixtureCursor(
            train,
            run_spec["token_mixture"],
            batch_size=args.batch_size,
            rank=rank,
            world_size=world,
            seed=args.seed,
            max_epochs=args.max_data_epochs,
            denominator="input" if args.input_tokens else "ce",
        )
        if args.resume:
            assert saved is not None
            cursor.load_state_dict(saved["data_cursor"])
    else:
        cursor = BatchCursor(len(train), args.seed, rank, world, args.batch_size, offset)
    # Indexer objectives can leave compression parameters inactive on short samples.
    wrapped = (
        DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        if world > 1
        else model
    )
    balance: Any
    if args.model == "minikimik3":
        from .kimi_quantile_balance import KimiQuantileBalance

        balance = KimiQuantileBalance(model)
    else:
        balance = RouterBalance(model)
    qk_clip = None
    if args.model == "minikimik3":
        from .kimi_qk_clip import KimiQKClip

        qk_clip = KimiQKClip(model)
    if args.resume:
        assert saved is not None
        balance.load_state_dict(saved["router_balance"])
        if qk_clip is not None:
            qk_clip.load_state_dict(saved["qk_clip"])
    writer = None
    if rank == 0:
        (output / "run.json").write_text(
            json.dumps(
                dict(
                    **run_spec,
                    started_at=time.time(),
                    kind=args.run_kind,
                    data=str(data_dir),
                    parameters=sum(p.numel() for p in model.parameters() if p.is_floating_point()),
                    command_options=vars(args),
                ),
                indent=2,
            )
        )
        # Every checkpoint has a portable tokenizer beside it.
        (output / "tokenizer.json").write_bytes((data_dir / "tokenizer.json").read_bytes())
        if not args.no_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                str(output / "tensorboard"), purge_step=first_step + 1 if first_step else None
            )
    del saved

    def autocast():
        return torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        )

    def objective(module, x, y, **extras):
        if rollout is not None:
            return rollout(module, x, y)
        if args.stage == "dpo":
            x, y = x.flatten(0, 1), y.flatten(0, 1)
            with torch.no_grad():
                ref = reference(x, attention_mask=x.ne(0)).logits
            logits = module(x, attention_mask=x.ne(0)).logits
            loss, accuracy = dpo_loss(logits, ref, y)
            return loss, loss.detach(), accuracy
        # Inputs are right-padded; labels carry the loss mask. Sparse stages use unpadded pretraining.
        result = module(x, labels=y, attention_mask=x.ne(0), return_logits=False, **extras)
        return result.loss, result.lm_loss, result

    def record(value):
        if rank == 0:
            print(json.dumps(value), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(value) + "\n")
            if writer:
                for key, v in value.items():
                    if key not in {"step", "event"} and isinstance(v, (int, float)):
                        writer.add_scalar(f"{value['event']}/{key}", v, value["step"])
                writer.flush()

    def evaluate(step):
        model.eval()
        values = torch.zeros(6, device=device, dtype=torch.float64)
        random_state = rng_state(device)
        with torch.no_grad(), autocast():
            for i in validation_indices(
                len(val),
                limit=args.eval_batches * world,
                seed=args.seed,
                rank=rank,
                world_size=world,
            ):
                validation_batch = batch(val, [i], device)
                x, y = validation_batch
                loss, lm, _reward = (
                    validation_rollout(model, x, y)
                    if validation_rollout
                    else objective(model, x, y, **validation_batch.extras)
                )
                # Weight by supervised tokens (DPO is a per-pair objective).
                count = (
                    1 if args.stage in {"dpo", "grpo", "mopd"} else int((y[:, 1:] != -100).sum())
                )
                values += torch.stack(
                    [
                        loss.float() * count,
                        lm.float() * count,
                        loss.new_tensor(count),
                        loss.new_tensor(
                            validation_rollout.last_reward if validation_rollout else 0.0
                        ),
                        loss.new_tensor(1),
                        _reward if args.stage == "dpo" else loss.new_tensor(0),
                    ]
                )
        if world > 1:
            dist.all_reduce(values)
        metrics = dict(
            event="validation",
            step=step,
            loss=(values[0] / values[2]).item(),
            lm_loss=(values[1] / values[2]).item(),
            supervised_tokens=int(values[2]),
            reward=(values[3] / values[4]).item(),
            examples=int(values[4].item()),
            sampling="uniform_without_replacement",
        )
        if args.stage == "dpo":
            metrics["preference_accuracy"] = (values[5] / values[4]).item()
        record(metrics)
        restore_rng(random_state, device)
        model.train()

    token_budget = args.ce_tokens or args.input_tokens

    def budget_used():
        return ledger.ce_tokens if args.ce_tokens is not None else ledger.input_tokens

    def finished(step):
        return budget_used() >= token_budget if token_budget is not None else step == args.steps

    def save(step):
        if world > 1 and finished(step):
            disagreement = torch.zeros((), device=device)
            for parameter in model.parameters():
                for chunk in parameter.detach().flatten().split(1024 * 1024):
                    source = chunk.clone()
                    dist.broadcast(source, src=0)
                    disagreement += (~torch.eq(chunk, source)).sum()
            dist.all_reduce(disagreement)
            if disagreement.item():
                raise RuntimeError("model parameters differ across DDP ranks")
            record(dict(event="rank_verification", step=step, exact_parameters=True))
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state(device))
        else:
            states = [rng_state(device)]
        if rank == 0:
            payload = dict(
                schema_version=1,
                model_name=args.model,
                config=asdict(model.config),
                phase=phase,
                stage=args.stage,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                optimizer_kind=optimizer_kind,
                step=step,
                data_offset=cursor.offset,
                data_cursor=cursor.state_dict() if args.token_mixture else None,
                rng=states,
                run_spec=run_spec,
                tokenizer_sha256=data_manifest["tokenizer"]["sha256"],
                token_ledger=ledger.state_dict(),
                router_balance=balance.state_dict(),
                qk_clip=qk_clip.state_dict() if qk_clip is not None else None,
            )
            if reference is not None:
                payload["reference"] = reference.state_dict()
            atomic_save(payload, output / "checkpoint.pt")
            (output / "status.json").write_text(
                json.dumps(
                    dict(
                        step=step,
                        target_steps=args.steps,
                        stage=args.stage,
                        phase=phase,
                        state="complete" if finished(step) else "running",
                        token_budget=token_budget,
                        token_ledger=ledger.state_dict(),
                        capability_status="unassessed",
                        checkpoint=str(output / "checkpoint.pt"),
                        updated_at=time.time(),
                    )
                )
            )
        if world > 1:
            dist.barrier()

    record(
        dict(
            event="start",
            step=first_step,
            parameters=sum(p.numel() for p in model.parameters() if p.is_floating_point()),
            world_size=world,
        )
    )
    evaluate(first_step)
    model.train()
    step = first_step
    empty_windows = 0
    empty_rl_windows = 0
    profile = []
    while step < args.steps and not (token_budget is not None and budget_used() >= token_budget):
        next_step = step + 1
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(3, device=device)
        trainable_response = torch.zeros((), device=device, dtype=torch.int64)
        # Keep only token tensors (not activations) for this accumulation window.
        window = [batch(train, cursor.next(), torch.device("cpu")) for _ in range(args.grad_accum)]
        io_seconds = time.monotonic() - started
        counts = (
            window_counts(window, device, pairwise=args.stage == "dpo") if rollout is None else None
        )
        if counts is not None and counts[0] == 0 and args.stage != "dense_distill":
            ledger.skipped_windows += 1
            record(dict(event="zero_supervision_window", step=step, data_offset=cursor.offset))
            empty_windows += 1
            if empty_windows >= 32:
                save(step)
                raise ValueError("32 accumulation windows had no CE targets; checkpoint retained")
            continue
        empty_windows = 0
        from minifrontier.training.mtp import window_mtp_counts

        media_counts = torch.tensor(
            [
                sum(getattr(item, name) for item in window)
                for name in ("image_count", "video_count", "frame_count", "image_features")
            ],
            device=device,
            dtype=torch.int64,
        )
        if world > 1:
            dist.all_reduce(media_counts)
        mtp_counts = (
            window_mtp_counts(window, model.config, device)
            if model.config.mtp_enabled and rollout is None and args.stage != "dpo"
            else [0, 0, 0]
        )
        step = next_step
        factor = (
            step / max(1, args.warmup_steps)
            if step <= args.warmup_steps
            else 0.1
            + 0.9
            * 0.5
            * (
                1
                + math.cos(
                    math.pi * (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
                )
            )
        )
        if token_budget is not None:
            assert counts is not None
            progress = min(
                1.0, (budget_used() + int(counts[0 if args.ce_tokens else 1])) / token_budget
            )
            warmup = (
                args.warmup_tokens
                if args.warmup_tokens is not None
                else max(1, int(token_budget * 0.01))
            ) / token_budget
            # Start at a positive rate; the recorded schedule is driven by completed real tokens.
            if progress < warmup:
                factor = max(progress, 1 / token_budget) / warmup
            else:
                cooldown_start = 0.8 if args.schedule == "wsd" else warmup
                cooldown = max(0.0, min(1.0, (progress - cooldown_start) / (1 - cooldown_start)))
                factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cooldown))
        for group in optimizer.param_groups:
            group["lr"] = (
                args.muon_lr if optimizer_kind in {"qwen_muon", "kimi_muon"} else args.lr
            ) * factor
            if optimizer_kind in {"qwen_muon", "kimi_muon", "deepseek_muon"}:
                group["adam_lr"] = args.lr * factor
        window_balance = None
        if (
            args.model == "miniqwen4"
            and rollout is None
            and args.stage != "dpo"
            and phase != "dense_distill"
        ):
            from minifrontier.training.qwen_balance import QwenWindowBalance

            window_balance = QwenWindowBalance(model)
            before = rng_state(device)
            with torch.no_grad(), autocast():
                for cpu_batch in window:
                    prepass = cpu_batch.to(device)
                    local_x, local_y = prepass
                    with window_balance.capture(local_x.ne(0)):
                        model(
                            local_x,
                            attention_mask=local_x.ne(0),
                            labels=local_y,
                            return_logits=False,
                            **prepass.extras,
                        )
            restore_rng(before, device)
            window_balance.finalize()
        for micro in range(args.grad_accum):
            local_batch = window[micro].to(device)
            x, y = local_batch
            sync = (
                wrapped.no_sync()
                if world > 1 and micro < args.grad_accum - 1
                else contextlib.nullcontext()
            )
            with sync:
                capture = (
                    balance.capture(x.ne(0))
                    if rollout is None and phase != "dense_distill"
                    else contextlib.nullcontext()
                )
                # DPO flattens paired sequences before forwarding.
                if args.stage == "dpo":
                    capture = contextlib.nullcontext()
                clip_capture = (
                    qk_clip.capture(x.ne(0))
                    if qk_clip is not None and rollout is None and args.stage != "dpo"
                    else contextlib.nullcontext()
                )
                with autocast(), capture, clip_capture:
                    loss, lm, _auxiliary = objective(wrapped, x, y, **local_batch.extras)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}, rank {rank}")
                if rollout is not None or args.stage == "dpo":
                    weighted = loss / args.grad_accum
                elif args.stage == "dense_distill":
                    assert counts is not None
                    index_term = loss
                    if _auxiliary.mtp_aux_loss is not None:
                        index_term = index_term - _auxiliary.mtp_aux_loss
                        weighted = scaled_ce(
                            _auxiliary.mtp_aux_loss, _auxiliary.mtp_aux_count, mtp_counts[1], world
                        )
                    else:
                        weighted = loss * 0
                    weighted = weighted + scaled_ce(index_term, x.ne(0).sum(), counts[1], world)
                else:
                    assert counts is not None
                    weighted = scaled_ce(lm, y[:, 1:].ne(-100).sum(), counts[0], world)
                    auxiliary = loss - lm
                    if _auxiliary.mtp_loss is not None:
                        mtp_term = model.config.mtp_loss_coef * _auxiliary.mtp_loss
                        auxiliary = auxiliary - mtp_term
                        weighted = weighted + scaled_ce(
                            mtp_term, _auxiliary.mtp_tokens, mtp_counts[0], world
                        )
                    if _auxiliary.mtp_aux_loss is not None:
                        auxiliary = auxiliary - _auxiliary.mtp_aux_loss
                        denominator = mtp_counts[2 if args.model == "minideepseekv4" else 1]
                        weighted = weighted + scaled_ce(
                            _auxiliary.mtp_aux_loss, _auxiliary.mtp_aux_count, denominator, world
                        )
                    if args.model == "minideepseekv4":
                        seq = model.config.sequence_balance_coef * _auxiliary.aux_loss
                        auxiliary = auxiliary - seq
                        weighted = weighted + scaled_ce(
                            seq, x.ne(0).any(-1).sum(), counts[2], world
                        )
                    weighted = weighted + scaled_ce(auxiliary, x.ne(0).sum(), counts[1], world)
                weighted.backward()
            if rollout is not None:
                trainable_response += rollout.last_trainable_tokens
            metrics[0] += weighted.detach()
            metrics[1] += (
                scaled_ce(lm.detach(), y[:, 1:].ne(-100).sum(), counts[0], world)
                if counts is not None and args.stage != "dpo"
                else lm.detach() / args.grad_accum
            )
            metrics[2] += rollout.last_tokens if rollout else (y[..., 1:] != -100).sum()
        if window_balance is not None:
            window_balance.clear()
        if rollout is not None:
            if world > 1:
                dist.all_reduce(trainable_response)
            if int(trainable_response) == 0:
                if world > 1:
                    dist.all_reduce(metrics)
                ledger.response_tokens += int(metrics[2])
                ledger.skipped_windows += 1
                empty_rl_windows += 1
                step -= 1
                optimizer.zero_grad(set_to_none=True)
                record(
                    dict(
                        event="zero_advantage_window",
                        step=step,
                        zero_variance_fraction=1.0,
                        token_ledger=ledger.state_dict(),
                    )
                )
                if empty_rl_windows >= 32:
                    save(step)
                    raise ValueError(
                        "32 windows have no RL learning signal; change task curriculum before continuing"
                    )
                continue
            empty_rl_windows = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.clip_grad, error_if_nonfinite=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_started = time.monotonic()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_seconds = time.monotonic() - optimizer_started
        if qk_clip is not None and rollout is None and args.stage != "dpo":
            clip_metrics = qk_clip.update()
            if step % args.log_every == 0:
                record(dict(event="qk_clip", step=step, layers=clip_metrics))
        if phase != "dense_distill" and rollout is None and args.stage != "dpo":
            balance.update(args.router_bias_rate)
        if world > 1:
            dist.all_reduce(metrics)
        ledger.optimizer_updates += 1
        ledger.image_occurrences += int(media_counts[0])
        ledger.video_examples += int(media_counts[1])
        ledger.video_frames += int(media_counts[2])
        ledger.image_features += int(media_counts[3])
        if counts is not None:
            ledger.input_tokens += int(counts[1])
            if args.stage not in {"dense_distill", "dpo"}:
                ledger.ce_tokens += int(counts[0])
        else:
            ledger.response_tokens += int(metrics[2])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        performance = torch.tensor(
            [
                elapsed,
                io_seconds,
                optimizer_seconds,
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
                torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0,
                -torch.cuda.mem_get_info(device)[0] / 2**30 if device.type == "cuda" else 0,
            ],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(performance, op=dist.ReduceOp.MAX)
        if device.type == "cuda" and -float(performance[5]) < args.min_device_free_gib:
            save(step)
            raise RuntimeError("device reserve fell below configured minimum; checkpoint retained")
        if args.profile_warmup < step <= args.profile_warmup + args.profile_updates:
            profile.append(
                dict(
                    step=step,
                    seconds=float(performance[0]),
                    io_seconds=float(performance[1]),
                    optimizer_seconds=float(performance[2]),
                    peak_allocated_gib=float(performance[3]),
                    peak_reserved_gib=float(performance[4]),
                    min_device_free_gib=-float(performance[5]),
                    ce_tokens=int(counts[0]) if counts is not None else 0,
                    input_tokens=int(counts[1]) if counts is not None else 0,
                    images=int(media_counts[0]),
                    frames=int(media_counts[2]),
                )
            )
            if rank == 0 and len(profile) == args.profile_updates:
                seconds = sum(row["seconds"] for row in profile)
                result = dict(
                    recipe=run_spec,
                    updates=profile,
                    measured_updates=len(profile),
                    ce_tokens_per_second=sum(row["ce_tokens"] for row in profile) / seconds,
                    images_per_second=sum(row["images"] for row in profile) / seconds,
                    frames_per_second=sum(row["frames"] for row in profile) / seconds,
                    scope="this phase, context bucket, modality mixture and device count only",
                    includes=["data", "forward", "backward", "optimizer", "balance", "mtp", "ddp"],
                )
                (output / "performance.json").write_text(json.dumps(result, indent=2))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record(
                dict(
                    event="train",
                    step=step,
                    loss=(metrics[0] / world).item(),
                    lm_loss=(metrics[1] / world).item(),
                    grad_norm=grad_norm.item(),
                    lr=args.lr * factor,
                    tokens_per_second=metrics[2].item() / elapsed,
                    step_seconds=elapsed,
                    data_offset=cursor.offset,
                    token_ledger=ledger.state_dict(),
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else 0,
                )
            )
        if step % args.eval_every == 0 or finished(step):
            evaluate(step)
        if step % args.save_every == 0 or finished(step):
            save(step)
        if finished(step):
            break
    if token_budget is not None and budget_used() < token_budget:
        save(step)
        raise RuntimeError(
            "update safety limit reached before actual token budget; checkpoint retained"
        )
    if rank == 0:
        atomic_save(
            dict(
                schema_version=1,
                model_name=args.model,
                config=asdict(model.config),
                phase=phase,
                stage=args.stage,
                model=model.state_dict(),
                step=step,
                token_ledger=ledger.state_dict(),
                tokenizer_sha256=data_manifest["tokenizer"]["sha256"],
            ),
            output / "model.pt",
        )
        if writer:
            writer.close()
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
