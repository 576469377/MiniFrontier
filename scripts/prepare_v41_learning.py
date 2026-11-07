"""Build a small single-GPU learning corpus with disjoint images and paired blank controls."""

import argparse
import contextlib
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.corpus import train_tokenizer
from minifrontier.data.native import encode_native
from minifrontier.data.partitions import corpus_storage_root, open_corpus

COLORS = {
    "red": (217, 40, 46),
    "blue": (40, 85, 220),
    "green": (44, 169, 62),
    "yellow": (222, 190, 32),
}
QUESTION = "What color is the shape? Answer with one word."


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(output, tokenizer_path, natural=None, text=None, ocr=None):
    if tokenizer_path is None and any((natural, text, ocr)):
        raise ValueError("provide the frozen tokenizer when adding external source records")
    output.mkdir(parents=True, exist_ok=False)
    roots = [output / name for name in ("aligned", "blank")]
    for root in roots:
        (root / "images").mkdir(parents=True)
    counts = Counter()
    groups = {}
    blank_paths = {}
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path else None

    with contextlib.ExitStack() as stack:
        dbs = [
            stack.enter_context(contextlib.closing(sqlite3.connect(root / "corpus.sqlite")))
            for root in roots
        ]
        for db in dbs:
            db.execute(
                "CREATE TABLE samples(id TEXT PRIMARY KEY,stage TEXT,split TEXT,"
                "group_root TEXT,payload TEXT,text TEXT)"
            )

        def add(record, split, group, image_path=None):
            if groups.setdefault(group, split) != split:
                raise ValueError("group crosses data splits")
            for index, db in enumerate(dbs):
                row = dict(record)
                if image_path:
                    media = dict(row["media"][0])
                    if index:
                        with Image.open(image_path) as im:
                            size = im.size
                        if size not in blank_paths:
                            blank = roots[1] / "images" / f"blank-{size[0]}x{size[1]}.png"
                            Image.new("RGB", size, (127, 127, 127)).save(blank)
                            blank_paths[size] = (blank, sha256(blank))
                        blank, digest = blank_paths[size]
                        media = dict(kind="image", path=str(blank.resolve()), sha256=digest)
                    else:
                        media["path"] = str(image_path.resolve())
                    row["media"] = [media]
                db.execute(
                    "INSERT INTO samples VALUES(?,?,?,?,?,?)",
                    (
                        row["sample_id"],
                        row["stage"],
                        split,
                        group,
                        json.dumps(row),
                        row.get("text", ""),
                    ),
                )
            counts[(split, record["stage"], record["task"])] += 1

        # Each split and stage uses independent pixels. Every label occurs equally often.
        for stage_number, stage in enumerate(("pretrain", "sft")):
            for split_number, (split, count) in enumerate(
                (("train", 1536), ("val", 96), ("test", 96))
            ):
                rng = random.Random(12019 + stage_number * 10 + split_number)
                for i in range(count):
                    label, rgb = list(COLORS.items())[i % len(COLORS)]
                    bg = tuple(rng.randrange(214, 256) for _ in range(3))
                    color = tuple(channel + rng.randrange(-14, 15) for channel in rgb)
                    im = Image.new("RGB", (84, 84), bg)
                    x, y = rng.randint(4, 23), rng.randint(4, 23)
                    box = (x, y, x + rng.randint(38, 56), y + rng.randint(38, 56))
                    draw = ImageDraw.Draw(im)
                    (draw.ellipse if rng.randrange(2) else draw.rectangle)(box, fill=color)
                    group = hashlib.sha256(im.tobytes()).hexdigest()
                    f = roots[0] / "images" / (group + ".png")
                    im.save(f)
                    row = dict(
                        sample_id=identity(stage + group),
                        stage=stage,
                        task="shape_color",
                        source="generated_shapes",
                        group_id=group,
                        media=[dict(kind="image", sha256=sha256(f))],
                        visual_question=QUESTION,
                        visual_answer=label,
                        text="<|image|>\nUser: " + QUESTION + "\nAssistant: " + label,
                        turns=[
                            dict(role="user", content="<|image|>\n" + QUESTION),
                            dict(role="assistant", content=label),
                        ],
                    )
                    add(row, split, group, f)

        for source, limit in ((natural, 384), (ocr, 128)):
            if source is None:
                continue
            db = stack.enter_context(contextlib.closing(open_corpus(source)))
            pixels = corpus_storage_root(db)
            for split, n in (("train", limit), ("val", 16)):
                remaining = n
                for group, payload in db.execute(
                    "SELECT group_root,payload FROM samples WHERE "
                    "stage='pretrain' AND split=? ORDER BY id",
                    (split,),
                ):
                    row = json.loads(payload)
                    if len(row.get("media", [])) != 1 or row["media"][0].get("kind") != "image":
                        continue
                    f = pixels / row["media"][0]["path"]
                    if (
                        not f.is_file()
                        or group in groups
                        or len(tokenizer.encode(row["text"]).ids) > 105
                    ):
                        continue
                    add(row, split, group, f)
                    remaining -= 1
                    if not remaining:
                        break

        if text:
            db = stack.enter_context(contextlib.closing(open_corpus(text)))
            for split, n in (("train", 640), ("val", 32)):
                for group, payload in db.execute(
                    "SELECT group_root,payload FROM samples WHERE "
                    "stage='pretrain' AND split=? ORDER BY id LIMIT ?",
                    (split, n),
                ):
                    row = json.loads(payload)
                    if row.get("media") or group in groups:
                        continue
                    pieces = tokenizer.encode(row["text"], add_special_tokens=False).ids[:95]
                    row["text"] = tokenizer.decode(pieces, skip_special_tokens=False)
                    add(row, split, group)

        for db in dbs:
            db.commit()
    if tokenizer_path is None:
        tokenizer_path = output / "tokenizer.json"
        train_tokenizer(roots[0], tokenizer_path, 1024)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/strategies/minideepseekv41-learning.json"
    )
    config = json.loads(config_path.read_text())
    config["vocab_size"] = tokenizer.get_vocab_size()
    (output / "learning-config.json").write_text(json.dumps(config, indent=2))
    for root in roots:
        manifest = dict(
            kind="v41-learning",
            counts={"/".join(k): v for k, v in counts.items()},
            blank_control=root == roots[1],
            database_sha256=sha256(root / "corpus.sqlite"),
            source_manifests={
                str(s): sha256(s / "corpus-manifest.json") for s in (natural, ocr, text) if s
            },
        )
        (root / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2))
        encode_native(
            root,
            tokenizer_path,
            output / (root.name + "-encoded"),
            "minideepseekv41",
            max_length=128,
            max_features=16,
            min_pixels=3136,
            model_vocab_size=tokenizer.get_vocab_size(),
        )
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "tokenizer", "natural", "text", "ocr"):
        p.add_argument("--" + flag, type=Path, required=flag == "output")
    a = p.parse_args()
    prepare(a.output.resolve(), a.tokenizer, a.natural, a.text, a.ocr)
