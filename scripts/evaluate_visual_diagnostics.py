"""Training-set color counterfactuals: diagnostic image dependence, not generalization."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from minifrontier.models.factory import build_model
from minifrontier.multimodal import prepare_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Hash and load one open inode while the trainer atomically replaces its path.
    with args.checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["config"], phase=saved["phase"]).eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved["optimizer"]
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [
        json.loads(line)["record"]
        for line in (args.data / "pretrain.train.media.jsonl").read_text().splitlines()
    ]
    samples = []
    for color in ("red", "green", "blue", "yellow", "红色", "绿色", "蓝色", "黄色"):
        pattern = rf"\b{color}\b" if color.isascii() else color
        matching = [r for r in records if re.search(pattern, r["text"])]
        for record in matching[:4]:
            packed = prepare_record(
                record,
                tokenizer,
                saved["model_name"],
                root=manifest["media_root"],
                max_features=manifest["max_features"],
                model_vocab_size=model.config.vocab_size,
            )
            raw = tokenizer.encode(record["text"])
            start = record["text"].index(color)
            delta = packed.input_ids.shape[1] - (len(raw.ids) + 2)
            positions = [
                1 + i + delta
                for i, (a, b) in enumerate(raw.offsets)
                if a < start + len(color) and b > start
            ]
            samples.append((record, color, packed, positions))
    cases = []
    with torch.no_grad():
        for record, color, packed, positions in samples:
            other = next(p for r, c, p, _ in samples if c != color and r["lang"] == record["lang"])
            results = {}
            for condition in ("correct_image", "wrong_image", "masked_image"):
                media = [dict(span) for span in packed.extras["media"]]
                if condition == "wrong_image":
                    media[0]["patches"] = other.extras["media"][0]["patches"]
                elif condition == "masked_image":
                    media[0]["patches"] = torch.zeros_like(media[0]["patches"])
                logits = model(packed.input_ids, media=media).logits
                target = packed.input_ids[0, positions]
                selected = logits[0, [p - 1 for p in positions]]
                results[condition] = dict(
                    color_token_nll=float(
                        torch.nn.functional.cross_entropy(selected.float(), target)
                    ),
                    greedy_color_tokens=tokenizer.decode(selected.argmax(-1).tolist()),
                    expected_color_tokens=tokenizer.decode(target.tolist()),
                    exact_tokens=bool(torch.equal(selected.argmax(-1), target)),
                )
            cases.append(dict(sample_id=record["sample_id"], color=color, results=results))
    summary = {
        condition: dict(
            color_token_nll=sum(c["results"][condition]["color_token_nll"] for c in cases)
            / len(cases),
            exact_color_sequences=sum(c["results"][condition]["exact_tokens"] for c in cases),
        )
        for condition in ("correct_image", "wrong_image", "masked_image")
    }
    result = dict(
        checkpoint_sha256=digest,
        step=saved["step"],
        model=saved["model_name"],
        cases=cases,
        summary=summary,
        evaluated_images=len(cases),
        scope="training-set color counterfactual diagnostic only; not a vision release gate",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
