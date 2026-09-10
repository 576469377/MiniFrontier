"""Isolated-device MF1 optimizer microbatch screening and optional operator profiling."""

import argparse
import gc
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import write_json
from minifrontier.hardware import query_gpus
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.processing import process_frames
from minifrontier.multimodal import move
from minifrontier.provenance import source_identity
from minifrontier.training.minifrontier1_optim import QuantileBalance, make_optimizer


def run_case(
    config, *, length, batch_size, input_tokens, warmup, updates, media_features=0, profile=False
):
    torch.manual_seed(42)
    model = MiniFrontier1ForCausalLM(config).to("cuda:0").train()
    optimizer = make_optimizer(model)
    balance = QuantileBalance(model)
    accumulation = input_tokens // (length * batch_size)
    rows = []
    # All microbatch variants use the same flattened input samples at the same update index.
    inputs = torch.randint(24, config.vocab_size, (input_tokens // length, length), device="cuda:0")
    labels = inputs.clone()
    media = None
    if media_features:
        media = process_frames(
            [Image.new("RGB", (448, 448), "red")],
            max_features=media_features,
            patch_size=config.vision_config.patch_size,
        )
        n = media["feature_count"]
        assert n + 4 < length
        inputs[:, 1] = 9
        inputs[:, 2 : 2 + n] = config.image_token_id
        inputs[:, 2 + n] = 10
        labels[:, 1 : 3 + n] = -100
    for step in range(warmup + updates):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if profile and step == warmup
            else nullcontext()
        )
        with profiler as trace:
            for index in range(accumulation):
                lo, hi = index * batch_size, (index + 1) * batch_size
                spans = (
                    []
                    if media is None
                    else [
                        dict(media, batch_index=b, start=2, resource_kind="image")
                        for b in range(batch_size)
                    ]
                )
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16),
                    balance.capture(inputs[lo:hi].ne(0)),
                ):
                    result = model(
                        inputs[lo:hi],
                        labels=labels[lo:hi],
                        media=move(spans, "cuda:0"),
                        return_logits=False,
                    )
                (result.loss / accumulation).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            balance.update()
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        row = dict(
            step=step + 1,
            warmup=step < warmup,
            profiled=profile and step == warmup,
            input_tokens=input_tokens,
            ce_tokens=int(labels[:, 1:].ne(-100).sum()),
            seconds=elapsed,
            grad_norm=float(norm),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            free_gib=torch.cuda.mem_get_info()[0] / 1024**3,
        )
        if profile and step == warmup:
            assert trace is not None
            row["operators"] = trace.key_averages().table(
                sort_by="self_device_time_total", row_limit=30
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["free_gib"] < 2:
            raise RuntimeError("device reserve below 2 GiB")
    measured = [r for r in rows if not r["warmup"] and not r["profiled"]]
    return dict(
        length=length,
        microbatch=batch_size,
        accumulation=accumulation,
        media_features=media_features,
        rows=rows,
        ce_per_second=sum(r["ce_tokens"] for r in measured) / sum(r["seconds"] for r in measured)
        if measured
        else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--media-features", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.updates < 1
        or min(args.batches) < 1
        or any(args.input_tokens % (args.length * b) for b in args.batches)
    ):
        raise ValueError("use positive divisible microbatch controls")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(
        21 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    config = MiniFrontier1Config(**json.loads(Path(args.config).read_text()))
    report = dict(
        state="running",
        source=source_identity(),
        runner_sha256=sha256(__file__),
        config=asdict(config),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        devices=[asdict(d) for d in query_gpus()],
        controls=vars(args),
        scope="synthetic equal-input-budget screen; only a >=20 warmup / >=100 unprofiled-update run meets the measurement duration requirement",
        cases=[],
    )
    write_json(output / "report.json", report)
    try:
        for size in args.batches:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                case = run_case(
                    config,
                    length=args.length,
                    batch_size=size,
                    input_tokens=args.input_tokens,
                    warmup=args.warmup,
                    updates=args.updates,
                    media_features=args.media_features,
                    profile=args.profile,
                )
            except torch.cuda.OutOfMemoryError:
                case = dict(length=args.length, microbatch=size, state="out_of_memory")
            report["cases"].append(case)
            write_json(output / "report.json", report)
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
