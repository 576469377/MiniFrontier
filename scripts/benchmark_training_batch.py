"""Screen microbatches at a fixed synthetic token budget, including the optimizer.

This is an execution benchmark, not a quality comparison. Ragged real-data
accumulation windows must be checked separately before changing a training run.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from minifrontier.data import sha256
from minifrontier.models.factory import build_model
from minifrontier.provenance import source_identity
from minifrontier.training.minideepseekv4_optim import MiniDeepSeekV4Optimizer
from minifrontier.training.minikimik3_optim import MiniKimiK3Optimizer
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer


def case(model_name, config, batch, length, input_tokens, warmup, updates):
    torch.manual_seed(42)
    model = build_model(model_name, config).cuda().train()
    optimizer: torch.optim.Optimizer
    if model_name == "miniqwen4":
        optimizer = MiniQwen4Optimizer(model, lr=0.01, adam_lr=0.0003, weight_decay=0.1)
    elif model_name == "minikimik3":
        optimizer = MiniKimiK3Optimizer(model, lr=0.005, adam_lr=0.0003, weight_decay=0.1, eps=1e-8)
    else:
        optimizer = MiniDeepSeekV4Optimizer(model, lr=0.0003, weight_decay=0.1, eps=1e-8)
    ids = torch.randint(
        24, model.config.vocab_size, (input_tokens // length, length), device="cuda"
    )
    accumulation = input_tokens // (batch * length)
    rows = []
    for update in range(warmup + updates):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for offset in range(0, len(ids), batch):
            inputs = ids[offset : offset + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(inputs, labels=inputs, return_logits=False)
            (result.loss / accumulation).backward()
        torch.cuda.synchronize()
        backward_done = time.monotonic()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        row = dict(
            step=update + 1,
            microbatch=batch,
            accumulation=accumulation,
            warmup=update < warmup,
            seconds=elapsed,
            forward_backward_seconds=backward_done - started,
            optimizer_seconds=time.monotonic() - backward_done,
            input_tokens=input_tokens,
            ce_tokens=(length - 1) * len(ids),
            loss=float(result.loss.detach()),
            grad_norm=float(norm),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            free_gib=torch.cuda.mem_get_info()[0] / 1024**3,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["free_gib"] < 2:
            raise RuntimeError("GPU free memory is below the 2 GiB reserve")
    measured = rows[warmup:]
    return dict(
        microbatch=batch,
        accumulation=accumulation,
        rows=rows,
        ce_per_second=sum(row["ce_tokens"] for row in measured)
        / sum(row["seconds"] for row in measured),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["miniqwen4", "minikimik3", "minideepseekv4"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    if (
        min(args.batches) < 1
        or args.length < 2
        or args.input_tokens < 1
        or args.updates < 1
        or args.warmup < 0
    ):
        raise ValueError("invalid benchmark controls")
    if any(args.input_tokens % (args.length * batch) for batch in args.batches):
        raise ValueError("all cases must use exactly the same input token budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(
        21 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    report = dict(
        state="running",
        source=source_identity(),
        runner_sha256=sha256(__file__),
        config=json.loads(args.config.read_text()),
        config_sha256=sha256(args.config),
        controls={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        scope="short synthetic optimizer screen; no real data/media/validation/IO; not an admitted steady training profile",
        cases=[],
    )

    def write():
        p = args.output / "report.json"
        temporary = p.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(p)

    write()
    try:
        for batch in args.batches:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                result = case(
                    args.model,
                    args.config,
                    batch,
                    args.length,
                    args.input_tokens,
                    args.warmup,
                    args.updates,
                )
            except torch.cuda.OutOfMemoryError:
                result = dict(microbatch=batch, state="out_of_memory")
            report["cases"].append(result)
            write()
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write()


if __name__ == "__main__":
    main()
