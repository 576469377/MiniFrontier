"""Train a pooled V4.1 ViT and pass only its trained weights between decoders."""

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from minifrontier.data.text import sha256
from minifrontier.models.factory import build_model
from minifrontier.training.runtime import atomic_save
from minifrontier.training.transitions import load_previous
from minifrontier.training.v41_vision import (
    CaptionContrast,
    install_tower,
    positives,
    tower_payload,
)


def read(path):
    return json.loads(Path(path).read_text())


def status(path, values):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(values, indent=2))
    os.replace(tmp, path)


class Pairs(Dataset):
    def __init__(self, root, manifest, split, tokenizer, side, length):
        source = root / manifest["splits"][split]["file"]
        if sha256(source) != manifest["splits"][split]["sha256"]:
            raise ValueError("caption split checksum changed")
        self.root, self.side = Path(manifest["media_root"]), side
        self.rows = []
        with source.open() as fp:
            for line in fp:
                row = json.loads(line)
                ids = tokenizer.encode(row["caption"], add_special_tokens=False).ids[:length]
                if not ids or 0 in ids:
                    raise ValueError("empty caption or reserved padding in caption")
                self.rows.append((row["file"], torch.tensor(ids + [0] * (length - len(ids)))))
        if len(self.rows) != manifest["splits"][split]["images"]:
            raise ValueError("caption split count changed")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, ids = self.rows[i]
        with Image.open(self.root / name) as original:
            image = ImageOps.pad(
                original.convert("RGB"),
                (self.side, self.side),
                method=Image.Resampling.BICUBIC,
                color=(127, 127, 127),
            )
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return pixels, ids


def batches(size, batch, start, end, seed):
    per_epoch = size // batch
    if per_epoch == 0:
        raise ValueError("need at least one full batch of distinct images")
    for epoch in range(start // per_epoch, (end - 1) // per_epoch + 1):
        gen = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(size, generator=gen).tolist()
        for step in range(max(start, epoch * per_epoch), min(end, (epoch + 1) * per_epoch)):
            pos = (step % per_epoch) * batch
            yield order[pos : pos + batch]


def move(pixels, device):
    return pixels.to(device, non_blocking=True).float().div_(127.5).sub_(1)


@torch.no_grad()
def evaluate(model, dataset, device, batch, limit, seed):
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(len(dataset), limit))
    if len(indices) < 2:
        raise ValueError("held-out retrieval requires at least two unique images")
    images, texts, tokens = [], [], []
    model.eval()
    for pixels, ids in DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch):
        images.append(model.images(move(pixels, device)).cpu())
        texts.append(model.texts(ids.to(device)).cpu())
        tokens.append(ids)
    scores = torch.cat(images) @ torch.cat(texts).T
    valid = positives(torch.cat(tokens))
    idx = torch.arange(len(scores))
    return dict(
        images=len(scores),
        i2t_r1=valid[idx, scores.argmax(1)].float().mean().item(),
        t2i_r1=valid[scores.argmax(0), idx].float().mean().item(),
        chance_r1=valid.float().mean().item(),
    )


def contrastive(a):
    device = torch.device(a.device)
    if (
        min(a.batch_size, a.steps, a.eval_every, a.eval_limit, a.side, a.text_length) < 1
        or a.workers < 0
    ):
        raise ValueError("invalid training budget")
    root, out = a.data, a.output
    m = read(root / "manifest.json")
    if (
        m["schema"] != "v41-caption-pairs-v1"
        or sha256(root / "tokenizer.json") != m["tokenizer_sha256"]
    ):
        raise ValueError("invalid caption manifest/tokenizer")
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    cfg = read(a.config)["vision_config"]
    if a.side > 224 or a.side % cfg["patch_size"]:
        raise ValueError("stage A canvas must be patch-aligned and at most 224")
    train, val = [Pairs(root, m, s, tok, a.side, a.text_length) for s in ("train", "val")]
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "checkpoint.pt"
    if any(out.iterdir()) and not a.resume:
        raise ValueError("use --resume to continue an existing contrastive output")
    spec = dict(
        manifest_sha256=sha256(root / "manifest.json"),
        vision=cfg,
        seed=a.seed,
        side=a.side,
        batch_size=a.batch_size,
        text_length=a.text_length,
        learning_rate=a.lr,
        steps=a.steps,
        eval_limit=a.eval_limit,
    )
    torch.manual_seed(a.seed)
    model = CaptionContrast(cfg, tok.get_vocab_size()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    step = 0
    if a.resume:
        saved = torch.load(cp, map_location=device, weights_only=True)
        if saved["spec"] != spec:
            raise ValueError("contrastive resume recipe differs from checkpoint")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        step = saved["step"]
    start = step
    sampler = batches(len(train), a.batch_size, step, a.steps, a.seed)
    loader = DataLoader(
        train, batch_sampler=sampler, num_workers=a.workers, pin_memory=device.type == "cuda"
    )
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(out / "tensorboard"), purge_step=step + 1 if a.resume else None)
    log = out / "metrics.jsonl"

    def record(event):
        with log.open("a") as f:
            f.write(json.dumps(event) + "\n")
        for key in ("loss", "image_exposures"):
            if key in event:
                writer.add_scalar("train/" + key, event[key], event["step"])
        for key, value in event.get("validation", {}).items():
            writer.add_scalar("validation/" + key, value, event["step"])
        if device.type == "cuda":
            writer.add_scalar(
                "performance/peak_allocated_gib",
                torch.cuda.max_memory_allocated() / 2**30,
                event["step"],
            )
        writer.flush()
        print(json.dumps(event), flush=True)

    if not a.resume:
        record(
            dict(
                step=0, validation=evaluate(model, val, device, a.batch_size, a.eval_limit, a.seed)
            )
        )
    status(out / "status.json", dict(state="running", step=step, target_steps=a.steps))
    for pixels, ids in loader:
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(move(pixels, device), ids.to(device, non_blocking=True))
        if not torch.isfinite(loss):
            raise ValueError("contrastive loss is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        step += 1
        if step == start + 1 or step % 20 == 0:
            record(dict(step=step, loss=loss.item(), image_exposures=step * a.batch_size))
        if step % a.eval_every == 0 or step == a.steps:
            result = evaluate(model, val, device, a.batch_size, a.eval_limit, a.seed)
            record(dict(step=step, validation=result))
            atomic_save(
                dict(spec=spec, step=step, model=model.state_dict(), optimizer=opt.state_dict()), cp
            )
            status(out / "status.json", dict(state="running", step=step, target_steps=a.steps))
    writer.close()
    if step != a.steps:
        raise ValueError("contrastive data exhausted prematurely")
    evidence = dict(
        step=step,
        image_exposures=step * a.batch_size,
        caption_manifest=spec["manifest_sha256"],
        tokenizer_sha256=m["tokenizer_sha256"],
    )
    atomic_save(
        tower_payload(model.vit, cfg, phase="contrastive", evidence=evidence), out / "tower.pt"
    )
    status(
        out / "status.json",
        dict(
            state="complete",
            step=step,
            target_steps=a.steps,
            **{"image_exposures": step * a.batch_size, "tower_sha256": sha256(out / "tower.pt")},
            updated_at=time.time(),
        ),
    )


def attach(a):
    art = torch.load(a.tower, map_location="cpu", weights_only=True)
    if art["evidence"]["tokenizer_sha256"] != sha256(a.tokenizer):
        raise ValueError("tower/tokenizer provenance differs")
    cfg = read(a.config)
    torch.manual_seed(a.seed)
    model = build_model("minideepseekv41", cfg, phase="sparse_pretrain")
    model.bind_tokenizer(Tokenizer.from_file(str(a.tokenizer)))
    if a.text_checkpoint:
        base = torch.load(a.text_checkpoint, map_location="cpu", weights_only=True)
        if base["tokenizer_sha256"] != sha256(a.tokenizer):
            raise ValueError("text-backbone tokenizer differs")
        load_previous(model, base, transition="text-to-vision", stage="pretrain")
    install_tower(model, art, a.role)
    receipt = dict(
        role=a.role,
        tower_sha256=sha256(a.tower),
        text_sha256=sha256(a.text_checkpoint) if a.text_checkpoint else None,
        seed=a.seed,
        tower_evidence=art["evidence"],
    )
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(
        dict(
            model_name="minideepseekv41",
            stage="pretrain",
            phase="sparse_pretrain",
            step=0,
            config=asdict(model.config),
            model=model.state_dict(),
            tokenizer_sha256=sha256(a.tokenizer),
            vision_curriculum=receipt,
        ),
        a.output,
    )


def extract(a):
    src = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    ledger = src["token_ledger"]
    if (
        src["model_name"] != "minideepseekv41"
        or src.get("vision_curriculum", {}).get("role") != "temporary"
        or min(
            src["step"],
            ledger["image_occurrences"],
            ledger["ce_tokens"],
            ledger["optimizer_updates"],
        )
        <= 0
    ):
        raise ValueError("need a trained temporary visual decoder checkpoint")
    config = src["config"]["vision_config"]
    from minifrontier.models.minideepseekv41.vision import DeepSeekVision, DeepSeekVisionConfig

    vit = DeepSeekVision(DeepSeekVisionConfig(**config)).vit
    prefix = "vision.vit."
    vit.load_state_dict(
        {k[len(prefix) :]: v for k, v in src["model"].items() if k.startswith(prefix)}
    )
    evidence = dict(
        temporary_model_sha256=sha256(a.checkpoint),
        tokenizer_sha256=src["tokenizer_sha256"],
        parent=src["vision_curriculum"],
        ledger=ledger,
    )
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(tower_payload(vit, config, phase="generative", evidence=evidence), a.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    c = modes.add_parser("contrastive")
    for flag in ("data", "config", "output"):
        c.add_argument("--" + flag, type=Path, required=True)
    for name, default in dict(
        steps=4000,
        batch_size=32,
        side=224,
        text_length=96,
        workers=4,
        eval_every=100,
        eval_limit=128,
        seed=20261110,
    ).items():
        c.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--device", default="cuda")
    c.add_argument("--resume", action="store_true")
    c.add_argument(
        "--next-plan", type=Path, help="bound continuation generated before V-C completes"
    )
    at = modes.add_parser("attach")
    for name in ("tower", "config", "tokenizer", "output", "text-checkpoint"):
        at.add_argument("--" + name, type=Path, required=name != "text-checkpoint")
    at.add_argument("--role", choices=["temporary", "main"], required=True)
    at.add_argument("--seed", type=int, default=20261110)
    ex = modes.add_parser("extract")
    for name in ("checkpoint", "output"):
        ex.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    {"contrastive": contrastive, "extract": extract, "attach": attach}[args.command](args)
    if args.command == "contrastive" and args.next_plan:
        # Replace the process to release the CUDA context and all contrastive optimizer memory.
        import sys

        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "scripts.run_v41_visual_stages",
                "--plan",
                str(args.next_plan),
                "--contrastive-output",
                str(args.output),
            ],
        )


if __name__ == "__main__":
    main()
