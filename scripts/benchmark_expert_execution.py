"""Source-loop versus batched experts, forward/backward only; not a training profile."""

import argparse
import json
import time
from pathlib import Path

import torch

from minifrontier.models.factory import build_model
from minifrontier.models.grouped_experts import configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(196)
    model = build_model(args.model, args.config).to(args.device).train()
    ids = torch.randint(21, model.config.vocab_size, (1, 512), device=args.device)
    report = dict(
        model=args.model,
        scope="text-only forward/backward microbenchmark; no optimizer/data/DDP; background training active",
        measurements={},
    )
    reference = {}
    for mode in ("loop", "batched"):
        configure(model, mode)
        times = []
        for step in range(3 + args.steps):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(ids, labels=ids, return_logits=False).loss
            loss.backward()
            torch.cuda.synchronize()
            if step >= 3:
                times.append(time.monotonic() - start)
        entry = dict(
            seconds_per_microbatch=sum(times) / len(times),
            loss=float(loss.detach()),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        if mode == "loop":
            reference = {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        else:
            cosine_numerator = reference_norm = actual_norm = squared_error = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                actual = parameter.grad.detach().float().cpu().double()
                expected = reference[name].double()
                cosine_numerator += float((actual * expected).sum())
                reference_norm += float(expected.square().sum())
                actual_norm += float(actual.square().sum())
                squared_error += float((actual - expected).square().sum())
            entry.update(
                gradient_cosine=cosine_numerator / (reference_norm * actual_norm) ** 0.5,
                gradient_relative_l2=(squared_error / reference_norm) ** 0.5,
            )
        report["measurements"][mode] = entry
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(mode=mode, **entry)), flush=True)


if __name__ == "__main__":
    main()
