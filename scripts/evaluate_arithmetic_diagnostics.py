"""Greedy answers on generated diagnostic train/validation rows; no fluency claim."""

import argparse
import hashlib
import json
import re
import sqlite3

import torch
from tokenizers import Tokenizer

from minifrontier.inference import generate_ids
from minifrontier.models.factory import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    with open(args.checkpoint, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    model = build_model(saved["model_name"], saved["config"], phase=saved["phase"]).eval()
    model.load_state_dict(saved["model"], strict=True)
    model.to(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    db = sqlite3.connect(f"file:{args.corpus}/corpus.sqlite?mode=ro", uri=True)
    report = dict(
        checkpoint_sha256=digest,
        step=saved["step"],
        model=saved["model_name"],
        scope="generated diagnostic arithmetic; training recall and tiny group-heldout validation only",
        main_quality_passed=False,
        cases=[],
    )
    for split in ("train", "val"):
        for (payload,) in db.execute(
            "SELECT payload FROM samples WHERE split=? AND task='diagnostic_text' ORDER BY id LIMIT 16",
            (split,),
        ):
            row = json.loads(payload)
            raw = tokenizer.encode(row["text"])
            answer_start = row["text"].index(" = ") + 3
            # The answer's leading space may be merged into its first digit token.
            # Take the actual training prefix; appending a new standalone space
            # would measure an unseen BPE boundary rather than training recall.
            boundary = next(i for i, (_, end) in enumerate(raw.offsets) if end > answer_start)
            prefix = raw.ids[:boundary]
            prompt = tokenizer.decode(prefix)
            ids = torch.tensor([[1, *prefix]], device=args.device)
            generated = generate_ids(
                model, ids, temperature=0, max_new_tokens=12, vocab_size=tokenizer.get_vocab_size()
            )[0, ids.shape[1] :]
            answer = tokenizer.decode(generated.tolist())
            integer = re.match(r"\s*(\d+)", answer)
            case = dict(
                sample_id=row["sample_id"],
                split=split,
                prompt=prompt,
                answer=answer,
                expected=row["verifier"]["answer"],
                correct=bool(integer and int(integer[1]) == row["verifier"]["answer"]),
            )
            report["cases"].append(case)
            print(json.dumps(case, ensure_ascii=False), flush=True)
    report["summary"] = {
        split: dict(
            total=sum(c["split"] == split for c in report["cases"]),
            correct=sum(c["split"] == split and c["correct"] for c in report["cases"]),
        )
        for split in ("train", "val")
    }
    with open(args.output, "w") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
