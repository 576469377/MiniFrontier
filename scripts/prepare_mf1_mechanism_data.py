"""Build a bounded MF1 mechanism pool from an existing audited corpus and generated media."""

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import (
    encode_record,
    safe_text,
    train_tokenizer,
    validate_record,
    write_json,
)
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.models.minifrontier1.processing import CONTROL_VERSION, PROCESSOR_VERSION
from minifrontier.storage import require_space

SOURCES = {
    "opencsg/Fineweb-Edu-Chinese-V2.1": "zh_general",
    "HuggingFaceFW/fineweb-edu": "en_general",
    "minifrontier-generated-math-recipe": "math",
}


def text_chunks(text, tokenizer, max_tokens):
    """Split on Unicode boundaries and retain every character; never split a target in place."""
    start = 0
    while start < len(text):
        remaining = text[start:]
        encoded = tokenizer.encode(remaining, add_special_tokens=False)
        end = (
            len(remaining) if len(encoded.ids) <= max_tokens else encoded.offsets[max_tokens - 1][1]
        )
        while end and len(safe_text(tokenizer, remaining[:end])) > max_tokens:
            end -= 1
        if not end:
            raise ValueError("token budget cannot represent one Unicode character")
        yield start, start + end, remaining[:end]
        start += end


def native_text(payload, group):
    return dict(
        sample_id=payload["sample_id"],
        split_group=group,
        language=payload["lang"],
        domain=SOURCES[payload["source"]],
        source=dict(
            dataset=payload["source"], revision=payload["revision"], record_id=payload["item_id"]
        ),
        provenance=dict(
            license_record=payload["license"], original_content_sha256=payload["content_hash"]
        ),
        supervision=dict(type="continuation_ce"),
        media=[],
        messages=[dict(role="user", content=[dict(type="text", text=payload["text"])])],
    )


def read_candidates(database, documents_per_source, seed):
    """Inherit the existing group split; never read sealed test payloads."""
    rng = random.Random(seed)
    result: dict[str, list[dict]] = {"train": [], "val": []}
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        for split in result:
            for source in SOURCES:
                rows = connection.execute(
                    "SELECT id, group_root FROM samples WHERE source=? AND split=? ORDER BY id",
                    (source, split),
                ).fetchall()
                rng.shuffle(rows)
                limit = documents_per_source if split == "train" else 32
                for identifier, group in rows[:limit]:
                    raw = connection.execute(
                        "SELECT payload FROM samples WHERE id=?", (identifier,)
                    ).fetchone()[0]
                    result[split].append(native_text(json.loads(raw), group))
    if not all(result.values()):
        raise ValueError("the source pool needs nonempty inherited train and validation splits")
    if {r["split_group"] for r in result["train"]} & {r["split_group"] for r in result["val"]}:
        raise ValueError("source group crosses the inherited train/validation split")
    return result


def unique_chunks(records, other_split_hashes):
    """Drop normalized duplicate text chunks and reject new cross-split leakage."""
    kept, hashes = [], set()
    for record in records:
        text = record["messages"][0]["content"][0]["text"]
        fingerprint = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
        if fingerprint in other_split_hashes:
            raise ValueError("normalized text chunk crosses train/validation split")
        if fingerprint not in hashes:
            kept.append(record)
            hashes.add(fingerprint)
    return kept, hashes, len(records) - len(kept)


def generated_media(root, split, count, seed):
    """Controlled colour and temporal tasks; unique media, separate seeds, no capability claim."""
    rng = random.Random(seed + (0 if split == "train" else 100000))
    rows = []
    for kind in ("image", "video"):
        for index in range(count):
            identity = f"mf1-mechanism-v1-{split}-{kind}-{index}"
            colours = [
                rng.choice(["red", "green", "blue"]) for _ in range(4 if kind == "video" else 1)
            ]
            if kind == "video" and colours[-1] == colours[0]:
                colours[-1] = "blue" if colours[0] != "blue" else "red"
            paths = []
            for frame, colour in enumerate(colours):
                path = root / "media" / f"{identity}-{frame}.png"
                picture = Image.new("RGB", (96, 96), colour)
                draw = ImageDraw.Draw(picture)
                for _ in range(96):
                    draw.point(
                        (rng.randrange(96), rng.randrange(96)),
                        fill=tuple(rng.randrange(256) for _ in range(3)),
                    )
                picture.save(path)
                paths.append(path.relative_to(root).as_posix())
            media = dict(
                media_id="m0",
                uri=paths[0],
                sha256=sha256(root / paths[0]),
                width=96,
                height=96,
                max_features=32,
            )
            if kind == "video":
                media.update(
                    frames=paths,
                    frame_sha256=[sha256(root / p) for p in paths],
                    timestamps=[0.0, 0.5, 1.0, 1.5],
                )
            rows.append(
                dict(
                    sample_id=identity,
                    split_group=identity,
                    language="en",
                    domain="video" if kind == "video" else "vision",
                    source=dict(
                        dataset="minifrontier-generated-colour-mechanism",
                        revision="1",
                        record_id=identity,
                    ),
                    provenance=dict(license_record="Apache-2.0", generation_seed=seed),
                    supervision=dict(type="answer_ce"),
                    media=[media],
                    expected=colours[-1],
                    messages=[
                        dict(
                            role="user",
                            content=[
                                dict(type=kind, media_id="m0"),
                                dict(
                                    type="text", text="Last color?" if kind == "video" else "Color?"
                                ),
                            ],
                        ),
                        dict(
                            role="assistant",
                            channel="final",
                            content=[dict(type="text", text=colours[-1])],
                        ),
                    ],
                )
            )
    return rows


def prepare(database, output, *, documents_per_source=2000, media_per_kind=128, seed=42):
    database, output = Path(database), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a new immutable dataset directory")
    require_space(output, 2 * 1024**3)
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir()
    corpus_manifest = json.loads((database.parent / "corpus-manifest.json").read_text())
    database_hash = sha256(database)
    if database_hash != corpus_manifest["database_sha256"]:
        raise ValueError("source database differs from its audited manifest")
    candidates = read_candidates(database, documents_per_source, seed)
    media = {
        split: generated_media(output, split, media_per_kind if split == "train" else 16, seed)
        for split in candidates
    }
    tokenizer = train_tokenizer(
        [*candidates["train"], *media["train"]], output / "tokenizer.json", 32768
    )
    if tokenizer.get_vocab_size() != 32768:
        raise ValueError(
            "mechanism pool was too small to build the configured 32K candidate vocabulary"
        )
    config = MiniFrontier1Config()
    counts, domain_tokens, domain_inputs, fingerprints = {}, {}, {}, {}
    text_hashes: set[str] = set()
    for split, records in candidates.items():
        expanded = []
        for record in records:
            text = record["messages"][0]["content"][0]["text"]
            for number, (start, end, chunk) in enumerate(text_chunks(text, tokenizer, 509)):
                copied = dict(record, sample_id=f"{record['sample_id']}/chunk-{number}")
                copied["messages"] = [dict(role="user", content=[dict(type="text", text=chunk)])]
                copied["provenance"] = dict(
                    record["provenance"],
                    transform="unicode-contiguous-chunks-max509-v1",
                    source_char_span=[start, end],
                )
                expanded.append(copied)
        expanded, split_hashes, duplicates = unique_chunks(expanded, text_hashes)
        text_hashes.update(split_hashes)
        expanded.extend(media[split])
        random.Random(seed).shuffle(expanded)
        # The reference evaluator reads a bounded prefix. Put two examples of every domain first.
        if split == "val":
            prefix, remainder = [], []
            seen: Counter[str] = Counter()
            for record in expanded:
                if seen[record["domain"]] < 2:
                    prefix.append(record)
                    seen[record["domain"]] += 1
                else:
                    remainder.append(record)
            expanded = prefix + remainder
        ce: Counter[str] = Counter()
        inputs: Counter[str] = Counter()
        media_hashes: set[str] = set()
        with (output / f"{split}.jsonl").open("w") as handle:
            for record in expanded:
                validate_record(record, output)
                item = encode_record(record, tokenizer, config, output)
                if item["input_ids"].numel() > 512:
                    raise ValueError("mechanism input exceeded the 512-token limit")
                ce[record["domain"]] += int(item["labels"][:, 1:].ne(-100).sum())
                inputs[record["domain"]] += item["input_ids"].numel()
                media_hashes.update(m["sha256"] for m in record["media"])
                for entry in record["media"]:
                    media_hashes.update(entry.get("frame_sha256", []))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = dict(
            records=len(expanded),
            source_documents=len(records),
            normalized_duplicate_chunks_removed=duplicates,
            domains=dict(Counter(r["domain"] for r in expanded)),
        )
        domain_tokens[split], domain_inputs[split] = dict(ce), dict(inputs)
        fingerprints[split] = media_hashes
    if fingerprints["train"] & fingerprints["val"]:
        raise ValueError("identical generated media crossed the split")
    manifest = dict(
        format="mf1-native-records-v1",
        kind="public-text-and-generated-media-mechanism",
        formal_admission=False,
        main_budget_eligible=False,
        seed=seed,
        media_root=".",
        tokenizer_sha256=sha256(output / "tokenizer.json"),
        vocab_size=32768,
        tokenizer_scope="candidate built on selected training records only; not the formal 5-10GB tokenizer comparison",
        control_template=CONTROL_VERSION,
        processor=PROCESSOR_VERSION,
        source_database_sha256=database_hash,
        source_manifest_sha256=sha256(database.parent / "corpus-manifest.json"),
        preparation_script_sha256=sha256(__file__),
        counts=counts,
        ce_tokens=domain_tokens,
        input_tokens=domain_inputs,
        files={split: sha256(output / f"{split}.jsonl") for split in candidates},
        limitations=[
            "bounded mechanism experiment only",
            "original train/val split inherited; sealed test not read",
            "source-level formal review and formal tokenizer comparison incomplete",
            "generated colour tasks do not measure general vision/video capability",
            "Python-Edu and externally sourced visual candidates excluded",
        ],
    )
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--documents-per-source", type=int, default=2000)
    parser.add_argument("--media-per-kind", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args())
    if args["documents_per_source"] < 1 or args["media_per_kind"] < 1:
        raise ValueError("positive source and media counts are required")
    prepare(**args)


if __name__ == "__main__":
    main()
