# ruff: noqa: RUF001
"""Compare SFT/DPO checkpoints on complete held-out sets and fixed greedy prompts.

This reports measurements and raw generations, never an automatic fluency pass.
Training data, model weights and tokenizer are left unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from minifrontier.data import StageDataset, sha256
from minifrontier.inference import load_checkpoint, respond
from minifrontier.training.posttrain import token_log_probs

PROMPTS = [
    "你好，请用一句话介绍你能做什么。",
    "请解释为什么下雨后地面会湿。",
    "把“今天天气很好”翻译成英文。",
    "请只回答数字：2加3等于几？",
]


def batches(dataset, batch_size, device):
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        yield tuple(torch.stack(values).to(device) for values in zip(*rows, strict=True))


@torch.no_grad()
def measure(model, dataset, batch_size, device):
    total_nll = count = hits = 0
    sequence_logps = []
    for x, y in batches(dataset, batch_size, device):
        if x.ndim == 3:
            x, y = x.flatten(0, 1), y.flatten(0, 1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x, attention_mask=x.ne(0)).logits
        logp, mask = token_log_probs(logits, y)
        total_nll -= logp.sum().item()
        count += mask.sum().item()
        hits += ((logits[:, :-1].argmax(-1) == y[:, 1:]) & mask).sum().item()
        sequence_logps.extend(logp.sum(-1).cpu().tolist())
    return dict(
        examples=len(dataset),
        supervised_tokens=count,
        nll=total_nll / count,
        token_accuracy=hits / count,
    ), torch.tensor(sequence_logps, dtype=torch.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--run", default="educational-v1")
    p.add_argument("--data", default="data/educational-v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    root = Path("outputs") / args.model / args.run
    report = dict(
        model=args.model,
        data_sha256=sha256(Path(args.data) / "manifest.json"),
        started_at=time.time(),
        checkpoints={},
        capability_status="requires_review_of_generations",
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sft_data = StageDataset(args.data, "sft", "val")
    dpo_data = StageDataset(args.data, "dpo", "val")
    reference = None
    for stage in ["sft", "dpo"]:
        path = root / stage / "model.pt"
        if not path.exists():
            continue
        model, tokenizer, meta = load_checkpoint(path, device)
        entry = dict(**meta, checkpoint=str(path), sha256=sha256(path))
        entry["sft_validation"], _ = measure(model, sft_data, args.batch_size, device)
        entry["dpo_validation_tokens"], logps = measure(model, dpo_data, args.batch_size, device)
        logps = logps.reshape(-1, 2)
        if stage == "sft":
            reference = logps
        if reference is not None:
            ratios = logps - reference
            margins = 0.1 * (ratios[:, 0] - ratios[:, 1])
            entry["preference_validation"] = dict(
                pairs=len(margins),
                reference="sft",
                beta=0.1,
                loss=(-torch.nn.functional.logsigmoid(margins)).mean().item(),
                accuracy=(margins > 0).double().mean().item(),
                ties=(margins == 0).double().mean().item(),
            )
        entry["generations"] = [
            dict(
                prompt=prompt,
                temperature=0,
                max_new_tokens=48,
                text=respond(model, tokenizer, prompt, temperature=0, max_new_tokens=48),
            )
            for prompt in PROMPTS
        ]
        report["checkpoints"][stage] = entry
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["finished_at"] = time.time()
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
