"""Bounded, auditable corpus construction for the 2026-09-08 strategies.

Normalized records retain provenance. Deduplication precedes group splitting;
transitive prompt/media/document/repository groups cannot leak across splits.
Token storage is ragged: PT sequences stay inside one document, SFT answers are
never truncated, and changing a curriculum length does not rewrite token files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from minifrontier.chat_controls import record_template, semantic_content, update_manifest
from minifrontier.data import SPECIAL_TOKENS, chat_tokens, fingerprint, normalized, sha256
from minifrontier.storage import GIB, require_space, reserve_write

STRATEGY_SPECIAL_TOKENS = [
    *SPECIAL_TOKENS,
    "<|image|>",
    "<|video|>",
    "<|media_begin|>",
    "<|media_end|>",
    "<|media_content|>",
    "<|effort_low|>",
    "<|effort_high|>",
    "<|effort_max|>",
    "<think>",
    "</think>",
    "<|final|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|draft_noise|>",
]
IDENTITY = re.compile(
    r"自我意识|自身存在|存在的意义|存在的目的|真实来源|your (?:creator|identity)|are you (?:conscious|sentient)",
    re.I,
)


def canonical_question(text):
    return re.sub(r"[\s，。！？,.!?]+", "", normalized(text).casefold())  # noqa: RUF001


def text_shingles(text):
    clean = re.sub(r"\s+", "", normalized(text).casefold())
    return {clean[i : i + 5] for i in range(max(1, len(clean) - 4))}


def simhash(text):
    digests = b"".join(
        hashlib.blake2b(s.encode(), digest_size=8).digest() for s in text_shingles(text)
    )
    rows = np.frombuffer(digests, dtype=np.uint8).reshape(-1, 8)
    bits = np.unpackbits(rows, axis=1)
    return int.from_bytes(np.packbits(bits.sum(0) * 2 >= len(bits)).tobytes(), "big")


class CorpusBuilder:
    def __init__(self, root, *, seed=42, max_gib=48):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "corpus-manifest.json").exists():
            raise FileExistsError("corpus is immutable; choose a new data version")
        self.seed, self.max_bytes = seed, int(max_gib * GIB)
        require_space(self.root, 16 * 1024**2)
        self.db = sqlite3.connect(self.root / "corpus.sqlite")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY, stage TEXT, source TEXT, task TEXT, first_question TEXT,
                text TEXT, payload TEXT, simhash TEXT, group_root TEXT, split TEXT);
            CREATE INDEX IF NOT EXISTS first_question_idx ON samples(first_question);
            CREATE TABLE IF NOT EXISTS bands (band INTEGER, value INTEGER, id TEXT);
            CREATE INDEX IF NOT EXISTS band_idx ON bands(band,value);
            CREATE TABLE IF NOT EXISTS links (id TEXT, key TEXT);
            CREATE INDEX IF NOT EXISTS link_key_idx ON links(key);
            CREATE TABLE IF NOT EXISTS image_bands (band INTEGER, value INTEGER, phash TEXT, id TEXT);
            CREATE INDEX IF NOT EXISTS image_band_idx ON image_bands(band,value);
        """)
        self.counts: Counter[str] = Counter()
        self.approximate_bytes = (self.root / "corpus.sqlite").stat().st_size

    def add(self, record):
        required = ("source", "revision", "item_id", "group_id", "license", "lang", "task", "stage")
        if any(not record.get(key) for key in required):
            raise ValueError(f"record lacks provenance fields: {required}")
        if record["stage"] not in {"pretrain", "sft"}:
            raise ValueError("preference/RL corpora have separate schemas and budgets")
        for media in record.get("media", []):
            if not media.get("rgb_sha256"):
                raise ValueError("media must be decoded and hashed before corpus admission")
            if media.get("kind") == "video" and (
                not media.get("video_id")
                or not media.get("frame_rgb_sha256")
                or len(media["frame_rgb_sha256"]) != len(media.get("frames", []))
            ):
                raise ValueError("video admission needs video id and decoded hash for every frame")
        if record.get("official_split", "train").startswith(("test", "validation")):
            self.counts["reserved_official_split"] += 1
            return False
        turns = record.get("turns", [])
        if record["stage"] == "sft":
            if not turns or turns[-1].get("role") != "assistant" or not turns[-1].get("content"):
                self.counts["incomplete_response"] += 1
                return False
            if any(
                t.get("role") not in {"system", "user", "assistant", "tool"}
                or (t.get("reasoning") is not None and not isinstance(t["reasoning"], str))
                or not (
                    isinstance(t.get("content"), str)
                    or (t.get("content") is None and t.get("tool_calls"))
                )
                for t in turns
            ):
                self.counts["invalid_turns"] += 1
                return False
            first = next((t["content"] for t in turns if t["role"] == "user"), "")
            media_context = "|".join(m["rgb_sha256"] for m in record.get("media", []))
            question = (
                fingerprint(
                    canonical_question(first) + ("|media:" + media_context if media_context else "")
                )
                if first
                else ""
            )
            text = "\n".join(t["role"] + ":" + semantic_content(t) for t in turns)
        else:
            text, question = normalized(record.get("text", "")), ""
            record = dict(record, text=text)
        if not 20 <= len(text) <= 2_000_000 or "\ufffd" in text:
            self.counts["invalid_text"] += 1
            return False
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        if lines and 1 - len(set(lines)) / len(lines) > 0.4:
            self.counts["repeated_boilerplate"] += 1
            return False
        # Cross-source exact content duplicates share the same identity.
        media_context = "|".join(m["rgb_sha256"] for m in record.get("media", []))
        identity = fingerprint(text + ("|media:" + media_context if media_context else ""))
        if self.db.execute("SELECT 1 FROM samples WHERE id=?", (identity,)).fetchone():
            self.counts["exact_duplicates"] += 1
            return False
        if question:
            existing = self.db.execute(
                "SELECT payload FROM samples WHERE first_question=?", (question,)
            ).fetchall()
            if len(existing) >= 3:
                self.counts["first_question_cap"] += 1
                return False
            answer = text_shingles(semantic_content(turns[-1]))
            for (payload,) in existing:
                other = text_shingles(semantic_content(json.loads(payload)["turns"][-1]))
                if len(answer & other) / max(1, len(answer | other)) >= 0.8:
                    self.counts["answer_template_duplicate"] += 1
                    return False
        code = simhash(text)
        candidates: set[str] = set()
        for band in range(4):
            candidates.update(
                row[0]
                for row in self.db.execute(
                    "SELECT id FROM bands WHERE band=? AND value=?",
                    (band, code >> (band * 16) & 65535),
                )
            )
        own_shingles = None
        for candidate in candidates:
            other_code, other_text, other_payload = self.db.execute(
                "SELECT simhash,text,payload FROM samples WHERE id=?", (candidate,)
            ).fetchone()
            other_media = "|".join(
                m["rgb_sha256"] for m in json.loads(other_payload).get("media", [])
            )
            if media_context != other_media:
                continue
            if (code ^ int(other_code, 16)).bit_count() <= 3:
                own_shingles = own_shingles or text_shingles(text)
                other = text_shingles(other_text)
                if len(own_shingles & other) / max(1, len(own_shingles | other)) >= 0.85:
                    self.counts["near_duplicates"] += 1
                    return False
        record = dict(
            record,
            sample_id=identity,
            content_hash=identity,
            quality_flags=record.get("quality_flags", []),
        )
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        incoming = 4 * len(payload.encode()) + 1024
        self.approximate_bytes += incoming
        if self.approximate_bytes > self.max_bytes:
            raise ValueError("corpus storage budget reached; accepted rows retained")
        if self.counts["accepted"] % 1000 == 0:
            self.db.commit()
            require_space(self.root, 64 * 1024**2)
        self.db.execute(
            "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
            (
                identity,
                record["stage"],
                record["source"],
                record["task"],
                question,
                text,
                payload,
                f"{code:016x}",
            ),
        )
        self.db.executemany(
            "INSERT INTO bands VALUES (?,?,?)",
            [(b, code >> (b * 16) & 65535, identity) for b in range(4)],
        )
        keys = ["source-group:" + record["source"] + ":" + record["group_id"]]
        if question:
            keys.append("question:" + question)
            # The question identity above includes ordered media hashes. Generic
            # "describe this image" prompts are different contexts on different
            # images; merging their templates would connect the whole visual corpus.
        for media in record.get("media", []):
            if not media.get("rgb_sha256"):
                raise ValueError("media must be decoded and hashed before corpus admission")
            keys.append("rgb:" + media["rgb_sha256"])
            keys.extend("rgb:" + value for value in media.get("frame_rgb_sha256", []))
            if media.get("phash"):
                code = int(media["phash"], 16)
                # Seven disjoint bands guarantee a candidate for <=6 bit changes.
                for band, (offset, width) in enumerate(
                    ((0, 10), (10, 9), (19, 9), (28, 9), (37, 9), (46, 9), (55, 9))
                ):
                    value = (code >> offset) & ((1 << width) - 1)
                    for old_hash, old_id in self.db.execute(
                        "SELECT phash,id FROM image_bands WHERE band=? AND value=?", (band, value)
                    ):
                        if (code ^ int(old_hash, 16)).bit_count() <= 6:
                            keys.append("phash-neighbor:" + old_id)
                            self.db.execute(
                                "INSERT INTO links VALUES (?,?)",
                                (old_id, "phash-neighbor:" + old_id),
                            )
                    self.db.execute(
                        "INSERT INTO image_bands VALUES (?,?,?,?)",
                        (band, value, media["phash"], identity),
                    )
            for field in ("document_id", "video_id", "origin_id"):
                if media.get(field):
                    keys.append(field + ":" + media[field])
        if record.get("repo_id"):
            keys.append("repo:" + record["repo_id"])
        self.db.executemany("INSERT INTO links VALUES (?,?)", [(identity, key) for key in keys])
        self.counts["accepted"] += 1
        return True

    def finalize(self):
        self.db.commit()
        # Union identifiers before splitting; same-image questions and translated
        # variants form one transitive component, including cross-source links.
        parents: dict[str, str] = {}

        def find(key):
            parents.setdefault(key, key)
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        first: dict[str, str] = {}
        for identity, key in self.db.execute("SELECT id,key FROM links ORDER BY key,id"):
            a = find(identity)
            if key in first:
                b = find(first[key])
                parents[max(a, b)] = min(a, b)
            else:
                first[key] = identity
        split_counts: Counter[str] = Counter()
        for identity in parents:
            root = find(identity)
            bucket = (
                int(hashlib.sha256(f"{self.seed}:{root}".encode()).hexdigest()[:16], 16) % 10000
            )
            split = "val" if bucket < 50 else "test" if bucket < 100 else "train"
            self.db.execute(
                "UPDATE samples SET group_root=?, split=? WHERE id=?", (root, split, identity)
            )
            split_counts[split] += 1
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        manifest = dict(
            schema_version=2,
            seed=self.seed,
            counts=dict(self.counts),
            splits=dict(split_counts),
            split_rule="connected groups; sha256 buckets val 0.5%, sealed test 0.5%",
            dedup="exact hash, first-question cap3, answer shingles0.8, simhash64+Jaccard0.85",
            database_sha256=sha256(self.root / "corpus.sqlite"),
        )
        (self.root / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def train_tokenizer(corpus_root, output, vocab_size, *, byte_budget=64 * 1024**2):
    corpus_root, output = Path(corpus_root), Path(output)
    if output.exists():
        raise FileExistsError("tokenizer is immutable")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    digest, bytes_seen = hashlib.sha256(), 0

    def corpus():
        nonlocal bytes_seen
        # tokenizers may consume the iterator on its worker thread. Open and
        # close the read-only connection on that same thread.
        db = sqlite3.connect(f"file:{corpus_root / 'corpus.sqlite'}?mode=ro", uri=True)
        try:
            for (text,) in db.execute("SELECT text FROM samples WHERE split='train' ORDER BY id"):
                encoded = text.encode()
                if bytes_seen + len(encoded) > byte_budget:
                    break
                digest.update(encoded)
                bytes_seen += len(encoded)
                yield text
        finally:
            db.close()

    tokenizer.train_from_iterator(
        corpus(),
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=STRATEGY_SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return dict(
        requested_vocab=vocab_size,
        actual_vocab=tokenizer.get_vocab_size(),
        training_bytes=bytes_seen,
        training_byte_sha256=digest.hexdigest(),
        sha256=sha256(output),
    )


def compare_tokenizers(corpus_root, output, *, byte_budget=64 * 1024**2):
    output = Path(output)
    if output.exists():
        raise FileExistsError("tokenizer comparison is immutable; choose a new version")
    output.mkdir(parents=True)
    candidates = [
        train_tokenizer(
            corpus_root, output / f"tokenizer-{size}.json", size, byte_budget=byte_budget
        )
        for size in (32768, 65536)
    ]
    if len({row["training_byte_sha256"] for row in candidates}) != 1:
        raise ValueError("tokenizer candidates did not use identical training bytes")
    result = dict(
        candidates=candidates,
        selected=65536,
        frozen=True,
        reason="strategy default; no controlled quality pilot has favored 32K",
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def encode_corpus(corpus_root, tokenizer_path, output, *, max_length=4096):
    """Encode text records once; media records require the native processor path."""
    root, output = Path(corpus_root), Path(output)
    if output.exists():
        raise FileExistsError("encoded corpus is immutable; select a new path")
    output.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    (output / "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    db = sqlite3.connect(f"file:{root / 'corpus.sqlite'}?mode=ro", uri=True)
    manifest = dict(
        schema_version=2,
        sequence_length=max_length,
        format="document-ragged-v2",
        corpus_sha256=sha256(root / "corpus-manifest.json"),
        stages={},
        tokenizer=dict(vocab_size=tokenizer.get_vocab_size(), sha256=sha256(tokenizer_path)),
    )
    for stage in ("pretrain", "sft"):
        manifest["stages"][stage] = {}
        for split in ("train", "val", "test"):
            prefix = f"{stage}.{split}"
            token_path, label_path = output / f"{prefix}.bin", output / f"{prefix}.labels.bin"
            index, token_count, supervised, identity_tokens = [], 0, 0, 0
            rejected: Counter[str] = Counter()
            template_counts: Counter[str] = Counter()
            records = db.execute(
                "SELECT payload FROM samples WHERE stage=? AND split=? ORDER BY id", (stage, split)
            )
            with (
                token_path.open("wb") as tokens,
                label_path.open("wb") as labels,
                (output / f"{prefix}.jsonl").open("w") as metadata,
            ):
                for (payload,) in records:
                    row = json.loads(payload)
                    if row.get("media"):
                        rejected["requires_native_media_encoder"] += 1
                        continue
                    if stage == "pretrain":
                        ids = [1, *tokenizer.encode(row["text"]).ids, 2]
                        targets = ids.copy()
                        targets[0] = -100
                    else:
                        ids, targets = chat_tokens(
                            row["turns"], tokenizer, mode=row.get("mode"), effort=row.get("effort")
                        )
                        if len(ids) > max_length:
                            rejected["complete_answer_exceeds_largest_bucket"] += 1
                            continue
                    ce = sum(value != -100 for value in targets[1:])
                    if stage == "sft" and IDENTITY.search(
                        next((t["content"] for t in row["turns"] if t["role"] == "user"), "")
                    ):
                        # Conservative streaming quota: may retain less than 0.5%, never more.
                        if identity_tokens + ce > 0.005 * (supervised + ce):
                            rejected["identity_token_cap"] += 1
                            continue
                        identity_tokens += ce
                    incoming = len(ids) * 8 + len(payload.encode()) + 512
                    with reserve_write(token_path, incoming):
                        np.asarray(ids, dtype=np.int32).tofile(tokens)
                        np.asarray(targets, dtype=np.int32).tofile(labels)
                        metadata.write(payload + "\n")
                    index.append((token_count, len(ids)))
                    if stage == "sft":
                        template_counts[record_template(row)] += 1
                    token_count += len(ids)
                    supervised += ce
            index_path = output / f"{prefix}.index.npy"
            np.save(index_path, np.asarray(index, dtype=np.int64).reshape(-1, 2))
            manifest["stages"][stage][split] = dict(
                format="document-ragged-v2",
                file=token_path.name,
                sha256=sha256(token_path),
                labels_file=label_path.name,
                labels_sha256=sha256(label_path),
                index_file=index_path.name,
                index_sha256=sha256(index_path),
                examples=len(index),
                supervised_tokens=supervised,
                stored_positions=token_count,
                rejected=dict(rejected),
                identity_tokens=identity_tokens,
                metadata_file=f"{prefix}.jsonl",
                metadata_sha256=sha256(output / f"{prefix}.jsonl"),
                chat_template_counts=dict(template_counts),
            )
    db.close()
    update_manifest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class DocumentDataset:
    def __init__(self, root, record, stage, length):
        self.length, self.stage = length, stage
        root = Path(root)
        for key, digest_key in (
            ("file", "sha256"),
            ("labels_file", "labels_sha256"),
            ("index_file", "index_sha256"),
        ):
            if sha256(root / record[key]) != record[digest_key]:
                raise ValueError("document dataset hash mismatch")
        self.tokens = (
            np.memmap(root / record["file"], mode="r", dtype=np.int32)
            if record["stored_positions"]
            else np.empty(0, dtype=np.int32)
        )
        self.labels = (
            np.memmap(root / record["labels_file"], mode="r", dtype=np.int32)
            if record["stored_positions"]
            else np.empty(0, dtype=np.int32)
        )
        self.documents = np.load(root / record["index_file"], mmap_mode="r")
        self.rows: list[tuple[int, int]] = []
        self.domains: list[str] = []
        self.ce_counts: list[int] = []
        self.input_counts: list[int] = []
        domains = []
        if record.get("metadata_file"):
            metadata_path = root / record["metadata_file"]
            if sha256(metadata_path) != record["metadata_sha256"]:
                raise ValueError("document metadata hash mismatch")
            with metadata_path.open() as metadata:
                domains = [json.loads(line)["task"] for line in metadata]
        for doc_index, (start, size) in enumerate(self.documents):
            first = len(self.rows)
            if stage == "pretrain":
                self.rows.extend(
                    (int(start + offset), int(min(length, size - offset)))
                    for offset in range(0, int(size) - 1, length - 1)
                )
            elif size <= length:
                self.rows.append((int(start), int(size)))

            for row_start, row_size in self.rows[first:]:
                self.domains.append(domains[doc_index] if domains else "unspecified")
                self.ce_counts.append(
                    int(np.count_nonzero(self.labels[row_start + 1 : row_start + row_size] != -100))
                )
                self.input_counts.append(row_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        start, size = self.rows[index]
        x = torch.zeros(self.length, dtype=torch.long)
        y = torch.full((self.length,), -100, dtype=torch.long)
        x[:size] = torch.from_numpy(self.tokens[start : start + size].astype(np.int64))
        y[:size] = torch.from_numpy(self.labels[start : start + size].astype(np.int64))
        return x, y
