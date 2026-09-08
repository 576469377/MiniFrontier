"""Reproducible public-corpus acquisition, tokenizer training and stage datasets.

Raw sources and processed artifacts stay outside the source distribution. A pinned
Hub revision, source/sample hashes, filtering counts, split seed and tokenizer hash
are recorded; no private flagship training corpus is implied.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
]
ROLES = {name: i + 3 for i, name in enumerate(("system", "user", "assistant", "tool"))}
REPO = "jingyaogong/minimind_dataset"
SOURCE_FILES = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return unicodedata.normalize("NFKC", text).replace("\x00", "").strip()


def fingerprint(text):
    return hashlib.sha256(re.sub(r"\s+", " ", normalized(text)).encode()).hexdigest()


def split_for(key, seed):
    return (
        "val"
        if int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100 < 2
        else "train"
    )


def chat_tokens(messages, tokenizer, *, generation_prompt=False):
    """Explicit role boundaries; only assistant contents and EOS are supervised."""
    ids, labels = [1], [-100]
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"unsupported conversation role: {role}")
        content = message.get("content") or ""
        # Preserve structured tool messages, rather than silently dropping them.
        if message.get("tool_calls"):
            content += "\n" + json.dumps(message["tool_calls"], ensure_ascii=False)
        if message.get("tools"):
            content += "\n" + json.dumps(message["tools"], ensure_ascii=False)
        value = [*tokenizer.encode(content, add_special_tokens=False).ids, 2]
        ids += [ROLES[role], *value]
        labels += [-100, *(value if role == "assistant" else [-100] * len(value))]
    if generation_prompt:
        ids += [ROLES["assistant"]]
        labels += [-100]
    return ids, labels


def download_prefix(output, stage, limit, revision, *, sampling="prefix", seed=42):
    """Sample a pinned source. Reservoir sampling reads the full file without retaining it."""
    if sampling not in {"prefix", "reservoir"}:
        raise ValueError("sampling must be prefix or reservoir")
    filename = SOURCE_FILES[stage]
    path = output / f"{stage}.source.jsonl"
    meta_path = path.with_suffix(".meta.json")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{filename}"
    expected = dict(
        url=url, requested_rows=limit, revision=revision, sampling="deterministic file prefix"
    )
    if sampling == "reservoir":
        expected.update(sampling="uniform reservoir over full source", sampling_seed=seed)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in expected.items()) and meta["sha256"] == sha256(path):
            return path, meta
        raise ValueError(f"existing source does not match requested source: {path}")
    temp = path.with_suffix(".download")
    rows = 0
    source_hash = hashlib.sha256()
    reservoir: list[tuple[int, bytes]] = []
    rng = random.Random(seed)
    with urllib.request.urlopen(url, timeout=120) as source, temp.open("wb") as dest:
        for line in source:
            source_hash.update(line)
            if not line.strip():
                continue
            json.loads(line)
            rows += 1
            if sampling == "prefix":
                dest.write(line)
                if rows >= limit:
                    break
            elif len(reservoir) < limit:
                reservoir.append((rows, line))
            else:
                slot = rng.randrange(rows)
                if slot < limit:
                    reservoir[slot] = (rows, line)
        if sampling == "reservoir":
            for _, line in sorted(reservoir):
                dest.write(line)
    temp.replace(path)
    meta = dict(
        **expected,
        rows=min(rows, limit),
        sha256=sha256(path),
        dataset_card=f"https://huggingface.co/datasets/{REPO}/blob/{revision}/README.md",
        declared_licenses=["apache-2.0", "cc-by-nc-2.0"],
    )
    if sampling == "reservoir":
        meta.update(source_rows=rows, full_source_sha256=source_hash.hexdigest())
    meta_path.write_text(json.dumps(meta, indent=2))
    return path, meta


def prepare_data(
    output,
    *,
    pretrain_rows=60000,
    sft_rows=30000,
    dpo_rows=10000,
    vocab_size=65536,
    sequence_length=256,
    seed=42,
    revision=None,
    local_sources=None,
    sampling="reservoir",
):
    output = Path(output).resolve()
    if min(pretrain_rows, sft_rows, dpo_rows) < 1 or vocab_size < 263 or sequence_length < 8:
        raise ValueError("invalid data capacity")
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "prepared corpus already exists; use it or choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    if revision is None and local_sources is None:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=30) as r:
            revision = json.load(r)["sha"]
    manifest = dict(
        schema_version=1,
        seed=seed,
        sequence_length=sequence_length,
        source_revision=revision,
        sources={},
        stages={},
        purpose="small-scale educational text training; not flagship data reproduction",
    )
    records = {}
    for stage, limit in [("pretrain", pretrain_rows), ("sft", sft_rows), ("dpo", dpo_rows)]:
        if local_sources:
            path = Path(local_sources[stage])
            meta = dict(path=str(path.resolve()), sha256=sha256(path), provenance="user supplied")
        else:
            path, meta = download_prefix(
                sources, stage, limit, revision, sampling=sampling, seed=seed
            )
        manifest["sources"][stage] = meta
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
        seen = set()
        rejected = duplicates = 0
        with path.open() as f:
            for index, line in enumerate(f):
                if index >= limit:
                    break
                row = json.loads(line)
                if stage == "pretrain":
                    text = normalized(row.get("text", ""))
                    key = text
                    row = {"text": text}
                elif stage == "sft":
                    messages = row.get("conversations", row.get("messages", []))
                    row = {"messages": messages}
                    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                    key = "\n".join(
                        m.get("content", "") or "" for m in messages if m.get("role") == "user"
                    )
                    if not any(m.get("role") == "assistant" and m.get("content") for m in messages):
                        rejected += 1
                        continue
                else:
                    if not isinstance(row.get("chosen"), list) or not isinstance(
                        row.get("rejected"), list
                    ):
                        rejected += 1
                        continue
                    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
                    key = "\n".join(
                        m.get("content", "") or "" for m in row["chosen"] if m.get("role") == "user"
                    )
                    if (
                        not row["chosen"]
                        or not row["rejected"]
                        or row["chosen"][:-1] != row["rejected"][:-1]
                        or row["chosen"][-1].get("role") != "assistant"
                        or row["rejected"][-1].get("role") != "assistant"
                    ):
                        rejected += 1
                        continue
                    if row["chosen"] == row["rejected"]:
                        rejected += 1
                        continue
                if not 20 <= len(text) <= 24000 or not key.strip():
                    rejected += 1
                    continue
                identity = fingerprint(text)
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                splits[split_for(fingerprint(key), seed)].append(row)
        if not all(splits.values()):
            raise ValueError(f"{stage} has an empty split; provide a larger corpus")
        records[stage] = splits
        manifest["stages"][stage] = dict(
            raw_accepted={k: len(v) for k, v in splits.items()},
            rejected=rejected,
            duplicates=duplicates,
        )
        print(json.dumps(dict(stage=stage, **manifest["stages"][stage])), flush=True)
    # Tokenizer sees training text only; validation and rejected preference answers are excluded.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def corpus():
        for row in records["pretrain"]["train"]:
            yield row["text"]
        for row in records["sft"]["train"]:
            for message in row["messages"]:
                yield message.get("content") or ""

    tokenizer.train_from_iterator(corpus(), trainer)
    tokenizer.save(str(output / "tokenizer.json"))
    manifest["tokenizer"] = dict(
        vocab_size=tokenizer.get_vocab_size(),
        requested_vocab_size=vocab_size,
        sha256=sha256(output / "tokenizer.json"),
        special_tokens=SPECIAL_TOKENS,
    )
    for stage, splits in records.items():
        for split, rows in splits.items():
            # Persist cleaned text for transparent review and re-tokenization.
            text_path = output / f"{stage}.{split}.jsonl"
            with text_path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if stage == "pretrain":
                path = output / f"{stage}.{split}.bin"
                tokens = 0
                with path.open("wb") as f:
                    for row in rows:
                        ids = [*tokenizer.encode(row["text"]).ids, 2]
                        np.asarray(ids, dtype=np.int32).tofile(f)
                        tokens += len(ids)
                count = (tokens - 1) // (sequence_length - 1)
            else:
                encoded = []
                for row in rows:
                    messages = (
                        [row["messages"]] if stage == "sft" else [row["chosen"], row["rejected"]]
                    )
                    pair = []
                    for chat in messages:
                        ids, labels = chat_tokens(chat, tokenizer)
                        ids, labels = ids[:sequence_length], labels[:sequence_length]
                        if all(x == -100 for x in labels[1:]):
                            break
                        pad = sequence_length - len(ids)
                        pair.append([ids + [0] * pad, labels + [-100] * pad])
                    if len(pair) == len(messages):
                        encoded.append(pair)
                if not encoded:
                    raise ValueError(
                        f"{stage}/{split} has no usable responses at this sequence length"
                    )
                path = output / f"{stage}.{split}.npy"
                np.save(path, np.asarray(encoded, dtype=np.int32))
                count = len(encoded)
                tokens = sum(
                    int((np.asarray(p[1])[1:] != -100).sum()) for pair in encoded for p in pair
                )
            manifest["stages"][stage][split] = dict(
                file=path.name, sha256=sha256(path), examples=count, supervised_tokens=tokens
            )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(dict(prepared=str(output), tokenizer=manifest["tokenizer"])), flush=True)
    return manifest


class StageDataset:
    def __init__(self, root, stage, split="train", sequence_length=None):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        self.stage = stage
        self.length = sequence_length or manifest["sequence_length"]
        record = manifest["stages"][stage][split]
        self.documents: Any = None
        if record.get("format") == "hybrid-native-v2":
            from minifrontier.native_data import NativeDataset

            self.documents = NativeDataset(root, record, stage, self.length)
            return
        if record.get("format") == "document-ragged-v2":
            from minifrontier.data_v2 import DocumentDataset

            self.documents = DocumentDataset(root, record, stage, self.length)
            return
        path = root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"dataset hash mismatch: {path}")
        self.values = (
            np.memmap(path, dtype=np.int32, mode="r")
            if stage == "pretrain"
            else np.load(path, mmap_mode="r")
        )
        if stage != "pretrain" and self.values.shape[-1] != self.length:
            raise ValueError("SFT/DPO length must match prepared corpus")

    def __len__(self):
        if self.documents is not None:
            return len(self.documents)
        return (
            (len(self.values) - self.length) // (self.length - 1) + 1
            if self.stage == "pretrain"
            else len(self.values)
        )

    def __getitem__(self, index):
        import torch

        if self.documents is not None:
            return self.documents[index]

        if self.stage == "pretrain":
            start = index * (self.length - 1)
            x = torch.from_numpy(self.values[start : start + self.length].astype(np.int64))
            return x, x.clone()
        row = torch.from_numpy(self.values[index].astype(np.int64))
        return (row[0, 0], row[0, 1]) if self.stage == "sft" else (row[:, 0], row[:, 1])
