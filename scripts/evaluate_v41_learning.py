"""Greedy held-out color answers with correct images, switched labels' images, and blank pixels."""

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import torch
from PIL import Image

from minifrontier.inference.runtime import generate_ids, load_checkpoint, respond
from minifrontier.multimodal import collate, prepare_record


@torch.inference_mode()
def evaluate(checkpoint, corpus, split, device):
    model, tok, info = load_checkpoint(checkpoint, device)
    db = sqlite3.connect((corpus / "corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        samples = [
            json.loads(row[0])
            for row in db.execute(
                "select payload from samples where stage='sft' and split=? order by id", (split,)
            )
        ]
    finally:
        db.close()
    samples = [r for r in samples if r["task"] == "shape_color"]
    by_color: dict[str, list[dict]] = {}
    for r in samples:
        by_color.setdefault(r["visual_answer"], []).append(r)
    colors = sorted(by_color)
    if len(colors) != 4 or len({len(v) for v in by_color.values()}) != 1:
        raise ValueError("color controls require a balanced four-color split")
    cases: list[tuple[dict, dict]] = []
    for i, label in enumerate(colors):
        alternate = colors[(i + 1) % len(colors)]
        cases.extend(zip(by_color[label], by_color[alternate], strict=True))

    all_outputs = {}
    with tempfile.TemporaryDirectory() as tmp:
        blank = Path(tmp) / "blank.png"
        with Image.open(cases[0][0]["media"][0]["path"]) as im:
            Image.new("RGB", im.size, (127, 127, 127)).save(blank)
        for variant in ("correct", "swap", "blank"):
            outputs: list[str] = []
            for first in range(0, len(cases), 12):
                batch = []
                for r, other in cases[first : first + 12]:
                    resource = (
                        {"path": str(blank)}
                        if variant == "blank"
                        else (other if variant == "swap" else r)["media"][0]
                    )
                    prompt = dict(stage="sft", turns=r["turns"][:1], media=[resource])
                    batch.append(
                        prepare_record(
                            prompt,
                            tok,
                            "minideepseekv41",
                            max_features=64,
                            model_vocab_size=model.config.vocab_size,
                            generation_prompt=True,
                        )
                    )
                if len({b.input_ids.shape[1] for b in batch}) != 1:
                    raise ValueError("batched prompts must have equal length")
                data = collate(batch, device)
                generated = generate_ids(
                    model,
                    data.input_ids,
                    max_new_tokens=8,
                    temperature=0,
                    vocab_size=tok.get_vocab_size(),
                    media=data.extras["media"],
                )
                outputs.extend(
                    tok.decode(s, skip_special_tokens=True).strip().lower()
                    for s in generated[:, data.input_ids.shape[1] :].tolist()
                )
            all_outputs[variant] = outputs
        r = cases[0][0]
        cli_answer = (
            respond(
                model,
                tok,
                r["visual_question"],
                images=[r["media"][0]["path"]],
                temperature=0,
                max_new_tokens=8,
            )
            .strip()
            .lower()
        )
        if cli_answer != all_outputs["correct"][0]:
            raise RuntimeError("single-image public generation disagrees with batched evaluation")
    rows = [
        dict(
            id=r["sample_id"],
            expected=r["visual_answer"],
            swapped_expected=o["visual_answer"],
            **{k: v[i] for k, v in all_outputs.items()},
        )
        for i, (r, o) in enumerate(cases)
    ]
    return dict(
        checkpoint=str(checkpoint),
        step=info["step"],
        split=split,
        total=len(rows),
        correct=sum(r["correct"] == r["expected"] for r in rows),
        blank=sum(r["blank"] == r["expected"] for r in rows),
        swap_follows_image=sum(r["swap"] == r["swapped_expected"] for r in rows),
        swap_keeps_original=sum(r["swap"] == r["expected"] for r in rows),
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "corpus", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    result = evaluate(a.checkpoint, a.corpus, a.split, a.device)
    a.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "rows"})
