"""Bounded MF1 CUDA optimizer probes; synthetic inputs measure execution, not capability."""

import argparse
import json
import math
import os
import time
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
from minifrontier.storage import require_space
from minifrontier.training.minifrontier1_optim import make_optimizer, parameter_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--cases", nargs="+", default=["text-128", "text-512", "image-49", "image-196", "video-128"]
    )
    args = parser.parse_args()
    if args.steps < 2 or args.memory_gib <= 0:
        raise ValueError("at least two updates and a positive memory ceiling are required")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a new output directory")
    require_space(output, 64 * 1024**2)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        args.memory_gib * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device
    )
    config = MiniFrontier1Config(**json.loads(Path(args.config).read_text()))
    model = MiniFrontier1ForCausalLM(config).to(device).train()
    optimizer = make_optimizer(model)
    report = dict(
        state="running",
        source=source_identity(),
        runner_sha256=sha256(__file__),
        config=asdict(config),
        parameters=parameter_report(model, optimizer),
        device=torch.cuda.get_device_name(device),
        devices=[asdict(g) for g in query_gpus()],
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        memory_limit_gib=args.memory_gib,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        seed=42,
        scope="shared-GPU synthetic-input optimizer probe; not an isolated benchmark or learning experiment",
        cases=[],
    )
    write_json(output / "report.json", report)
    try:
        for case in args.cases:
            length = int(case.split("-")[1]) if case.startswith("text-") else 512
            ids = torch.randint(24, config.vocab_size, (1, length))
            labels = ids.clone()
            media = []
            if not case.startswith("text-"):
                video = case.startswith("video-")
                count = 4 if video else 1
                budget = int(case.split("-")[1])
                sample = process_frames(
                    [
                        Image.new("RGB", (448, 448), color)
                        for color in ["red", "green", "blue", "red"][:count]
                    ],
                    max_features=budget,
                    patch_size=config.vision_config.patch_size,
                    timestamps=[0.0, 0.5, 1.0, 1.5] if video else None,
                )
                n = sample["feature_count"]
                sample.update(batch_index=0, start=2, resource_kind="video" if video else "image")
                ids[0, 1] = 20 if video else 9
                ids[0, 2 : 2 + n] = config.image_token_id
                ids[0, 2 + n] = 21 if video else 10
                labels[0, 1 : 3 + n] = -100
                media = [sample]
            batch = move(dict(input_ids=ids, labels=labels, media=media), device)
            for update in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                start = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch, return_logits=False)
                result.loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                vision_grad = (
                    sum(
                        float(p.grad.detach().float().square().sum())
                        for name, p in model.named_parameters()
                        if name.startswith("vision.") and p.grad is not None
                    )
                    ** 0.5
                )
                optimizer.step()
                torch.cuda.synchronize(device)
                seconds = time.monotonic() - start
                row = dict(
                    case=case,
                    update=update + 1,
                    warmup=update == 0,
                    seconds=seconds,
                    ce_tokens=int(labels[:, 1:].ne(-100).sum()),
                    input_tokens=ids.numel(),
                    vision_tokens=sum(m["feature_count"] for m in media),
                    loss=float(result.loss.detach()),
                    lm_loss=float(result.lm_loss.detach()),
                    grad_norm=float(norm),
                    vision_grad_norm=vision_grad,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                    free_device_gib=torch.cuda.mem_get_info(device)[0] / 1024**3,
                )
                if not all(
                    math.isfinite(row[k]) for k in ("loss", "grad_norm", "vision_grad_norm")
                ):
                    raise RuntimeError("nonfinite numerical result")
                if media and vision_grad <= 0:
                    raise RuntimeError("visual input did not produce a vision gradient")
                report["cases"].append(row)
                write_json(output / "report.json", report)
                print(json.dumps(row), flush=True)
                if row["free_device_gib"] < 5:
                    raise RuntimeError(
                        "physical GPU free memory fell below the 5 GiB probe reserve"
                    )
                del result
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
