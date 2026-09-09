"""Versioned native records, safe control encoding and bounded local data preparation."""

import hashlib
import json
import math
import random
import sqlite3
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from minifrontier.data import sha256
from minifrontier.data.corpus import STRATEGY_SPECIAL_TOKENS
from minifrontier.models.minifrontier1.processing import (
    CONTROL_VERSION,
    PROCESSOR_VERSION,
    process_document,
    process_frames,
)
from minifrontier.storage import require_space

SPECIAL_TOKENS = [
    *STRATEGY_SPECIAL_TOKENS[:20],
    "<|video_begin|>",
    "<|video_end|>",
    "<|frame|>",
    "<|time|>",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require_space(path, len(raw.encode()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def safe_text(tokenizer, text):
    if not isinstance(text, str):
        raise ValueError("ordinary message content must be text")
    # Explicit escape policy: the inserted WORD JOINER prevents a user string from
    # being parsed as a control token. Ordinary code/whitespace/numbers stay intact.
    for token in SPECIAL_TOKENS:
        text = text.replace(token, token[0] + "\u2060" + token[1:])
    return tokenizer.encode(text, add_special_tokens=False).ids


def validate_record(record, root, *, allow_sources=None):
    for key in ("sample_id", "split_group", "language", "domain"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"record requires nonempty {key}")
    source = record.get("source", {})
    if any(not source.get(k) for k in ("dataset", "revision", "record_id")):
        raise ValueError("source dataset/revision/record_id must be pinned")
    license_record = record.get("provenance", {}).get("license_record")
    if not license_record or (
        allow_sources is not None
        and (source["dataset"], source["revision"], license_record) not in allow_sources
    ):
        raise ValueError("source/revision/license is not admitted")
    if not record.get("messages") or record.get("supervision", {}).get("type") not in {
        "answer_ce",
        "continuation_ce",
    }:
        raise ValueError("record needs messages and an explicit CE supervision type")
    available = {}
    for media in record.get("media", []):
        if (
            not media.get("media_id")
            or media["media_id"] in available
            or min(media.get("width", 0), media.get("height", 0)) <= 0
        ):
            raise ValueError("media IDs must be unique and dimensions nonzero")
        frames = media.get("frames", [media.get("uri")])
        hashes = media.get("frame_sha256", [media.get("sha256")])
        if (
            not frames
            or len(frames) != len(hashes)
            or any(not isinstance(h, str) or len(h) != 64 for h in hashes)
        ):
            raise ValueError("each local media/frame requires SHA256")
        if len(frames) > 1 and (
            len(media.get("timestamps", [])) != len(frames)
            or any(
                not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0
                for t in media["timestamps"]
            )
            or any(b <= a for a, b in pairwise(media["timestamps"]))
        ):
            raise ValueError("video requires ordered source timestamps")
        for uri, expected in zip(frames, hashes, strict=True):
            path = (Path(root) / str(uri)).resolve()
            if (
                not path.is_relative_to(Path(root).resolve())
                or not path.is_file()
                or sha256(path) != expected
            ):
                raise ValueError("missing media, path outside dataset or hash mismatch")
            with Image.open(path) as image:
                if image.size != (media["width"], media["height"]):
                    raise ValueError("decoded media dimensions differ from manifest")
        available[media["media_id"]] = media
    used = []
    for message in record["messages"]:
        if message.get("role") not in {"system", "user", "assistant", "tool"} or not isinstance(
            message.get("content"), list
        ):
            raise ValueError("messages need a known role and typed content")
        if message.get("channel", "final") not in {"final", "thinking", "tool_call"}:
            raise ValueError("unknown assistant channel")
        for part in message["content"]:
            if part.get("type") == "text":
                if not isinstance(part.get("text"), str) or "\x00" in part["text"]:
                    raise ValueError("invalid text payload")
            elif part.get("type") in {"image", "video"} and part.get("media_id") in available:
                resource = available[part["media_id"]]
                if part["type"] == "video" and len(resource.get("frames", [])) < 2:
                    raise ValueError("video needs at least two hashed source frames and timestamps")
                if part["type"] == "image" and len(resource.get("frames", [])) > 1:
                    raise ValueError("image content cannot silently consume a video resource")
                if resource.get("representation", "standard") not in {"standard", "document"} or (
                    resource.get("representation") == "document" and part["type"] != "image"
                ):
                    raise ValueError("document crops require an image resource")
                used.append(part["media_id"])
            else:
                raise ValueError("unknown/missing content or media reference")
    if set(used) != set(available) or len(used) != len(set(used)):
        raise ValueError("each media resource must occur once in expanded messages")
    return record


def encode_record(record, tokenizer, config, root, *, generation_prompt=False):
    if any(tokenizer.token_to_id(s) != i for i, s in enumerate(SPECIAL_TOKENS)):
        raise ValueError("MF1 needs its own frozen control-token mapping")
    ids, labels = [1], [-100]
    spans: list[dict[str, Any]] = []
    resources = {m["media_id"]: m for m in record.get("media", [])}
    roles = {"system": 3, "user": 4, "assistant": 5, "tool": 6}
    for message in record["messages"]:
        role = message["role"]
        if generation_prompt and role == "assistant":
            break
        ids.append(roles[role])
        labels.append(-100)
        if role == "assistant":
            marker = {"final": 17, "thinking": 15, "tool_call": 18}[message.get("channel", "final")]
            ids.append(marker)
            labels.append(marker)
        if role == "tool":
            ids.append(19)
            labels.append(-100)
        for part in message["content"]:
            if part["type"] == "text":
                tokens = safe_text(tokenizer, part["text"])
                ids.extend(tokens)
                supervise = (
                    role == "assistant" or record["supervision"]["type"] == "continuation_ce"
                )
                labels.extend(tokens if supervise else [-100] * len(tokens))
            else:
                resource = resources[part["media_id"]]
                frames = []
                for uri in resource.get("frames", [resource.get("uri")]):
                    with Image.open(Path(root) / uri) as image:
                        frames.append(image.convert("RGB"))
                remaining = config.protected_media_tokens - sum(s["feature_count"] for s in spans)
                samples = (
                    process_document(frames[0], patch_size=config.vision_config.patch_size)
                    if resource.get("representation") == "document"
                    else [
                        process_frames(
                            frames,
                            max_features=min(resource.get("max_features", remaining), remaining),
                            patch_size=config.vision_config.patch_size,
                            timestamps=resource.get("timestamps"),
                        )
                    ]
                )
                if sum(s["feature_count"] for s in samples) > remaining:
                    raise ValueError("document global view and source crops exceed media budget")
                video = part["type"] == "video"
                for sample in samples:
                    ids.append(20 if video else 9)
                    labels.append(-100)
                    sample.update(
                        batch_index=0,
                        start=len(ids),
                        resource_kind="video" if video else "image",
                        media_id=resource["media_id"],
                        source_sha256=digest(resource["frame_sha256"])
                        if video
                        else resource["sha256"],
                    )
                    ids.extend([7] * sample["feature_count"])
                    labels.extend([-100] * sample["feature_count"])
                    ids.append(21 if video else 10)
                    labels.append(-100)
                    spans.append(sample)
        ids.append(2)
        labels.append(
            2 if role == "assistant" or record["supervision"]["type"] == "continuation_ce" else -100
        )
    if generation_prompt:
        thinking = record.get("mode") == "thinking" or any(
            m.get("channel") == "thinking" for m in record["messages"]
        )
        prefix = [5] if record.get("tool_environment") else [5, 15 if thinking else 17]
        ids.extend(prefix)
        labels.extend([-100] * len(prefix))
    if len(ids) > config.max_position_embeddings:
        raise ValueError(
            "expanded sample exceeds context; truncation requires a traceable data transform"
        )
    return dict(
        input_ids=torch.tensor([ids]),
        labels=torch.tensor([labels]),
        media=spans,
        sample_id=record["sample_id"],
        domain=record["domain"],
        split_group=record["split_group"],
        media_hashes=[s["source_sha256"] for s in spans],
        media_exposures=len({s["media_id"] for s in spans}),
    )


def train_tokenizer(records, path, vocab_size=32768):
    if vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError("vocabulary must include byte fallback and all control tokens")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        (
            p["text"]
            for r in records
            for m in r["messages"]
            for p in m["content"]
            if p["type"] == "text"
        ),
        trainer,
    )
    tokenizer.save(str(path))
    return tokenizer


def prepare_records(input_path, output, source_allowlist, *, seed=42, max_gib=8):
    """Stream JSONL through a disk index; deduplicate before group-level splitting.

    Media stays beside the source JSONL. The output records preserve a bound root;
    publication/export must copy admitted assets explicitly, never fetch mutable URLs.
    """
    source = Path(input_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("data versions are immutable; choose a fresh output")
    require_space(output, int(max_gib * 1024**3))
    output.mkdir(parents=True)
    allowed = {
        (x["dataset"], x["revision"], x["license_record"])
        for x in source_allowlist
        if x.get("status") == "admitted"
    }
    db = sqlite3.connect(output / "prepare.sqlite")
    db.executescript(
        "CREATE TABLE rows(id INTEGER PRIMARY KEY, group_id TEXT, payload TEXT); CREATE TABLE identities(hash TEXT PRIMARY KEY, group_id TEXT); CREATE TABLE parents(id TEXT PRIMARY KEY, parent TEXT);"
    )
    counts: Counter[str] = Counter()

    def root(group):
        while True:
            row = db.execute("SELECT parent FROM parents WHERE id=?", (group,)).fetchone()
            if row is None:
                db.execute("INSERT INTO parents VALUES(?,?)", (group, group))
                return group
            if row[0] == group:
                return group
            group = row[0]

    with source.open() as handle, (output / "quarantine.jsonl").open("w") as quarantine:
        for line in handle:
            counts["read"] += 1
            try:
                record = validate_record(json.loads(line), source.parent, allow_sources=allowed)
            except (ValueError, KeyError, OSError) as error:
                quarantine.write(json.dumps(dict(line=counts["read"], reason=str(error))) + "\n")
                counts["quarantined"] += 1
                continue
            text = "\n".join(
                p["text"] for m in record["messages"] for p in m["content"] if p["type"] == "text"
            )
            identity = digest(
                dict(
                    messages=record["messages"],
                    media=[m.get("sha256", m.get("frame_sha256")) for m in record.get("media", [])],
                )
            )
            # Shared images/documents and whitespace-near-identical text link groups transitively.
            links = ["record:" + identity, "text:" + digest(" ".join(text.split()))]
            for media in record.get("media", []):
                links += ["media:" + h for h in media.get("frame_sha256", [media.get("sha256")])]
            group = root(record["split_group"])
            duplicate = db.execute("SELECT 1 FROM identities WHERE hash=?", (links[0],)).fetchone()
            for key in links:
                previous = db.execute(
                    "SELECT group_id FROM identities WHERE hash=?", (key,)
                ).fetchone()
                if previous:
                    other = root(previous[0])
                    if other != group:
                        db.execute("UPDATE parents SET parent=? WHERE id=?", (group, other))
                else:
                    db.execute("INSERT INTO identities VALUES(?,?)", (key, group))
            if duplicate:
                counts["exact_duplicate"] += 1
                continue
            record["raw_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            record["normalized_text_sha256"] = digest(" ".join(text.split()))
            db.execute(
                "INSERT INTO rows(group_id,payload) VALUES(?,?)",
                (group, json.dumps(record, ensure_ascii=False)),
            )
            if counts["read"] % 1000 == 0:
                db.commit()
                used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                if used > max_gib * 1024**3:
                    raise ValueError("preparation reached its explicit disk budget")
    db.commit()
    files = {split: (output / f"{split}.jsonl").open("w") for split in ("train", "val", "test")}
    try:
        for group, payload in db.execute("SELECT group_id,payload FROM rows ORDER BY id"):
            fraction = int(digest([seed, root(group)])[:8], 16) % 10000
            split = "train" if fraction < 9850 else "val" if fraction < 9900 else "test"
            record = json.loads(payload)
            record["split_group"] = root(group)
            files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in files.values():
            handle.close()
        db.close()
    manifest = dict(
        format="mf1-native-records-v1",
        kind="candidate",
        seed=seed,
        media_root=str(source.parent),
        source_sha256=sha256(source),
        counts=dict(counts),
        source_allowlist=source_allowlist,
        cleaning_version="mf1-groups-v1",
        formal_admission=False,
        pending=[
            "perceptual/semantic near-duplicate audit",
            "evaluation contamination audit",
            "100/300-record source review",
            "token/domain/media coverage",
        ],
        files={s: sha256(output / f"{s}.jsonl") for s in files},
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def make_fixture(output, *, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError("fixture output already exists")
    require_space(output, 32 * 1024**2)
    (output / "media").mkdir(parents=True)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count in (("train", 72), ("val", 12), ("test", 12), ("demo", 16)):
        records = []
        for i in range(count):
            group = f"{split}-{i}"
            record: dict[str, Any] = dict(
                sample_id=group,
                split_group=group,
                language="en",
                domain="math",
                source=dict(
                    dataset="mf1-generated-mechanism-fixture", revision="2", record_id=group
                ),
                provenance=dict(
                    license_record="generated-CC0", teacher=None, transform_version="mf1-fixture-v2"
                ),
                supervision=dict(type="answer_ce", verifier="exact"),
                media=[],
            )
            if i < (32 if split == "train" else count // 2):
                a, b = i + 1 + {"train": 0, "val": 70, "test": 170, "demo": 270}[split], i % 7
                prompt, answer = f"{a}+{b}=", str(a + b)
                content = [dict(type="text", text=prompt)]
            else:
                video = i >= (64 if split == "train" else count - 2)
                colors = [rng.choice(["red", "green", "blue"]) for _ in range(4 if video else 1)]
                if video:
                    colors[-1] = "blue" if colors[0] != "blue" else "red"
                paths = []
                for frame, color in enumerate(colors):
                    path = output / "media" / f"{group}-{frame}.png"
                    image = Image.new("RGB", (16, 16), color)
                    draw = ImageDraw.Draw(image)
                    for _ in range(10):
                        draw.point(
                            (rng.randrange(16), rng.randrange(16)),
                            fill=tuple(rng.randrange(256) for _ in range(3)),
                        )
                    image.save(path)
                    paths.append(path.relative_to(output).as_posix())
                resource = dict(
                    media_id="m0",
                    uri=paths[0],
                    sha256=sha256(output / paths[0]),
                    width=16,
                    height=16,
                    max_features=8,
                )
                if video:
                    resource.update(
                        frames=paths,
                        frame_sha256=[sha256(output / p) for p in paths],
                        timestamps=[0.0, 0.4, 1.2, 2.0],
                    )
                record["media"] = [resource]
                record["domain"] = "video" if video else "vision"
                prompt = "Last color?" if video else "Color?"
                answer = colors[-1]
                content = [
                    dict(type="video" if video else "image", media_id="m0"),
                    dict(type="text", text=prompt),
                ]
            record["messages"] = [
                dict(role="user", content=content),
                dict(role="assistant", channel="final", content=[dict(type="text", text=answer)]),
            ]
            record["expected"] = answer
            validate_record(record, output)
            records.append(record)
        splits[split] = records
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
    tokenizer = train_tokenizer(splits["train"], output / "tokenizer.json", vocab_size=320)
    manifest = dict(
        format="mf1-native-records-v1",
        kind="mechanism_fixture",
        formal_admission=False,
        seed=seed,
        media_root=".",
        tokenizer_sha256=sha256(output / "tokenizer.json"),
        vocab_size=tokenizer.get_vocab_size(),
        control_template=CONTROL_VERSION,
        processor=PROCESSOR_VERSION,
        counts={k: len(v) for k, v in splits.items()},
        unique_media=len({m["sha256"] for rs in splits.values() for r in rs for m in r["media"]}),
        files={k: sha256(output / f"{k}.jsonl") for k in splits},
        limitations=[
            "generated mechanism checks only",
            "not formal training data or a capability benchmark",
        ],
    )
    write_json(output / "manifest.json", manifest)
    return manifest


class RecordDataset:
    def __init__(self, root, split, config):
        self.root, self.config = Path(root).resolve(), config
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.media_root = (self.root / self.manifest.get("media_root", ".")).resolve()
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        if sha256(self.root / "tokenizer.json") != self.manifest["tokenizer_sha256"]:
            raise ValueError("tokenizer differs from dataset manifest")
        self.path = self.root / f"{split}.jsonl"
        if sha256(self.path) != self.manifest["files"][split]:
            raise ValueError("dataset shard differs from manifest")
        self.offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            record = json.loads(handle.readline())
        return validate_record(record, self.media_root)

    def __getitem__(self, index):
        return encode_record(self.record(index), self.tokenizer, self.config, self.media_root)
