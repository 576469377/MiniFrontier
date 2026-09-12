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
from minifrontier.data.media_hash import PHASH_BANDS
from minifrontier.data.partitions import open_corpus
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
    def __init__(
        self, root, *, seed=42, max_gib=48, val_buckets=50, test_buckets=50, group_image_phash=True
    ):
        if type(group_image_phash) is not bool:
            raise ValueError("image pHash grouping must be explicitly enabled or deferred")
        self.group_image_phash = group_image_phash
        if min(val_buckets, test_buckets) <= 0 or val_buckets + test_buckets >= 10000:
            raise ValueError("held-out hash buckets must leave a nonempty training fraction")
        self.val_buckets, self.test_buckets = val_buckets, test_buckets
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "corpus-manifest.json").exists():
            raise FileExistsError("corpus is immutable; choose a new data version")
        self.seed, self.max_bytes = seed, int(max_gib * GIB)
        require_space(self.root, 16 * 1024**2)
        self.db = sqlite3.connect((self.root / "corpus.sqlite").as_uri(), uri=True)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY, stage TEXT, source TEXT, task TEXT, first_question TEXT,
                text TEXT, payload TEXT, simhash TEXT, group_root TEXT, split TEXT);
            CREATE INDEX IF NOT EXISTS first_question_idx ON samples(first_question);
            CREATE TABLE IF NOT EXISTS bands (band INTEGER, value INTEGER, id TEXT);
            CREATE INDEX IF NOT EXISTS band_idx ON bands(band,value);
            CREATE TABLE IF NOT EXISTS links (
                id TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(key,id)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS image_bands (band INTEGER, value INTEGER, phash TEXT, id TEXT);
            CREATE INDEX IF NOT EXISTS image_band_idx ON image_bands(band,value);
        """)
        if (
            not group_image_phash
            and self.db.execute("SELECT 1 FROM image_bands LIMIT 1").fetchone()
        ):
            self.db.close()
            raise ValueError("cannot defer image grouping on a partially grouped corpus")
        # Existing interrupted databases retain their schema until explicitly compacted.
        link_schema = self.db.execute(
            "SELECT sql FROM sqlite_schema WHERE name='links'"
        ).fetchone()[0]
        if "WITHOUT ROWID" not in link_schema.upper():
            self.db.execute("CREATE INDEX IF NOT EXISTS link_key_idx ON links(key)")
        self.counts: Counter[str] = Counter()
        self.last_retained_id: str | None = None
        self.approximate_bytes = (self.root / "corpus.sqlite").stat().st_size

    def add(self, record):
        self.last_retained_id = None
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
            text = (
                record.get("text", "")
                if record.get("text_format") == "source_code"
                else normalized(record.get("text", ""))
            )
            question = ""
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
        identity = (
            hashlib.sha256(b"source-code\0" + text.encode()).hexdigest()
            if record.get("text_format") == "source_code"
            else fingerprint(text + ("|media:" + media_context if media_context else ""))
        )
        if self.db.execute("SELECT 1 FROM samples WHERE id=?", (identity,)).fetchone():
            self.last_retained_id = identity
            self._link_origin(identity, record)
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
            other_record = json.loads(other_payload)
            other_media = "|".join(m["rgb_sha256"] for m in other_record.get("media", []))
            if (
                record.get("text_format") == "source_code"
                or other_record.get("text_format") == "source_code"
            ) and (
                not record.get("syntax_sha256")
                or record["syntax_sha256"] != other_record.get("syntax_sha256")
            ):
                continue  # Whitespace/Unicode changes can change code semantics.
            if media_context != other_media:
                continue
            if (code ^ int(other_code, 16)).bit_count() <= 3:
                own_shingles = own_shingles or text_shingles(text)
                other = text_shingles(other_text)
                if len(own_shingles & other) / max(1, len(own_shingles | other)) >= 0.85:
                    self.last_retained_id = candidate
                    self._link_origin(candidate, record)
                    self.counts["near_duplicates"] += 1
                    return False
        record = dict(
            record,
            sample_id=identity,
            content_hash=identity,
            quality_flags=record.get("quality_flags", []),
        )
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        keys = ["source-group:" + record["source"] + ":" + record["group_id"]]
        if question:
            keys.append("question:" + question)
            # The question identity above includes ordered media hashes. Generic
            # "describe this image" prompts are different contexts on different
            # images; merging their templates would connect the whole visual corpus.
        links = set()
        image_bands = []
        for media in record.get("media", []):
            keys.append("rgb:" + media["rgb_sha256"])
            keys.extend("rgb:" + value for value in media.get("frame_rgb_sha256", []))
            if media.get("phash") and self.group_image_phash:
                image_code = int(media["phash"], 16)
                # Seven disjoint bands guarantee a candidate for <=6 bit changes.
                for band, (offset, width) in enumerate(PHASH_BANDS):
                    value = (image_code >> offset) & ((1 << width) - 1)
                    for old_hash, old_id in self.db.execute(
                        "SELECT phash,id FROM image_bands WHERE band=? AND value=?", (band, value)
                    ):
                        if (image_code ^ int(old_hash, 16)).bit_count() <= 6:
                            keys.append("phash-neighbor:" + old_id)
                            links.add((old_id, "phash-neighbor:" + old_id))
                    image_bands.append((band, value, media["phash"], identity))
            for field in ("document_id", "video_id", "origin_id"):
                if media.get(field):
                    keys.append(field + ":" + media[field])
        if record.get("repo_id"):
            keys.append("repo:" + record["repo_id"])
        if record.get("document_id"):
            keys.append("document:" + record["document_id"])
        links.update((identity, key) for key in keys)
        # Include aliases and image indexes, whose fan-out is unrelated to text size.
        incoming = 4 * len(payload.encode()) + 1024 + 256 * len(image_bands)
        incoming += 4 * sum(len(i.encode()) + len(k.encode()) + 32 for i, k in links)
        self._reserve_metadata(incoming)
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
        self.db.executemany("INSERT INTO image_bands VALUES (?,?,?,?)", image_bands)
        self.db.executemany("INSERT OR IGNORE INTO links VALUES (?,?)", sorted(links))
        self.counts["accepted"] += 1
        self.last_retained_id = identity
        return True

    def _reserve_metadata(self, incoming):
        page_bytes = (
            self.db.execute("PRAGMA page_count").fetchone()[0]
            * self.db.execute("PRAGMA page_size").fetchone()[0]
        )
        proposed = max(self.approximate_bytes, page_bytes) + incoming
        if proposed > self.max_bytes:
            raise ValueError("corpus storage budget reached; accepted rows retained")
        self.approximate_bytes = proposed

    def _link_origin(self, identity, record):
        # Rejected duplicates still connect their origin groups to the retained
        # document, so other variants of that origin cannot leak into held-out.
        keys = ["source-group:" + record["source"] + ":" + record["group_id"]]
        if record.get("document_id"):
            keys.append("document:" + record["document_id"])
        if record.get("repo_id"):
            keys.append("repo:" + record["repo_id"])
        links = [
            (identity, key)
            for key in set(keys)
            if not self.db.execute(
                "SELECT 1 FROM links WHERE key=? AND id=?", (key, identity)
            ).fetchone()
        ]
        self._reserve_metadata(4 * sum(len(i.encode()) + len(k.encode()) + 32 for i, k in links))
        self.db.executemany("INSERT OR IGNORE INTO links VALUES (?,?)", links)

    def finalize(self, *, split_locks=None, excluded_ids=()):
        controlled_split = split_locks is not None or bool(excluded_ids)
        split_locks = split_locks or {}
        if any(split not in {"val", "test"} for split in split_locks.values()):
            raise ValueError("prior split locks may only preserve validation or sealed test")
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
        protected: dict[str, str] = {}
        for key, split in split_locks.items():
            if key in first:
                group = find(first[key])
                if protected.get(group) != "test":
                    protected[group] = split
        if set(excluded_ids) - parents.keys():
            raise ValueError("excluded identity does not belong to the corpus")
        excluded_groups = {find(identity) for identity in excluded_ids}
        removed = []
        split_counts: Counter[str] = Counter()
        for identity in parents:
            root = find(identity)
            if root in excluded_groups:
                removed.append((identity,))
                continue
            bucket = (
                int(hashlib.sha256(f"{self.seed}:{root}".encode()).hexdigest()[:16], 16) % 10000
            )
            split = (
                "val"
                if bucket < self.val_buckets
                else "test"
                if bucket < self.val_buckets + self.test_buckets
                else "train"
            )
            split = protected.get(root, split)
            self.db.execute(
                "UPDATE samples SET group_root=?, split=? WHERE id=?", (root, split, identity)
            )
            split_counts[split] += 1
        if removed:
            self.db.execute("CREATE TEMP TABLE excluded_records(id TEXT PRIMARY KEY)")
            self.db.executemany("INSERT INTO excluded_records VALUES (?)", removed)
            for table in ("samples", "links", "bands", "image_bands"):
                self.db.execute(
                    f"DELETE FROM {table} WHERE id IN (SELECT id FROM excluded_records)"
                )
            self.db.execute("DROP TABLE excluded_records")
            self.counts["accepted"] -= len(removed)
            self.counts["excluded_group_records"] += len(removed)
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        manifest = dict(
            schema_version=2,
            seed=self.seed,
            counts=dict(self.counts),
            splits=dict(split_counts),
            split_rule=(
                f"connected groups; sha256 buckets val {self.val_buckets / 100:g}%, "
                f"sealed test {self.test_buckets / 100:g}%"
            ),
            dedup="exact hash, first-question cap3, answer shingles0.8, simhash64+Jaccard0.85",
            database_sha256=sha256(self.root / "corpus.sqlite"),
        )
        if controlled_split:
            manifest["retained_split_controls"] = dict(
                prior_holdout_keys=len(split_locks),
                protected_groups=len(protected),
                excluded_groups=len(excluded_groups),
                excluded_records=len(removed),
                policy="prior test takes precedence over prior val; whole matched groups removed",
            )
        if not self.group_image_phash:
            manifest["image_phash_grouping"] = False
            manifest["image_near_duplicate_audit"] = (
                "deferred; requires a bound cross-corpus audit before admission"
            )
        (self.root / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def train_tokenizer(
    corpus_root, output, vocab_size, *, byte_budget=64 * 1024**2, special_tokens=None
):
    corpus_root, output = Path(corpus_root), Path(output)
    if output.exists():
        raise FileExistsError("tokenizer is immutable")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    digest, bytes_seen = hashlib.sha256(), 0

    def corpus():
        nonlocal bytes_seen
        # Keep database ownership on the calling thread. Tokenizers can hand
        # successive iterator batches to different Rayon workers.
        db = open_corpus(corpus_root)
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

    # Materialize only the explicit byte-budget sample, never the full corpus.
    training_texts = list(corpus())
    tokenizer.train_from_iterator(
        training_texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=STRATEGY_SPECIAL_TOKENS if special_tokens is None else special_tokens,
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


def _text_encoding_identity(row):
    fields = (
        "sample_id",
        "content_hash",
        "source",
        "revision",
        "item_id",
        "license",
        "stage",
        "task",
        "group_id",
        "text_format",
    )
    text_hash = (
        hashlib.sha256(row["text"].encode()).hexdigest()
        if "text" in row
        else row.get("encoded_text_sha256")
    )
    if not text_hash:
        raise ValueError("reused text metadata lacks its exact text identity")
    return hashlib.sha256(
        json.dumps([text_hash, *[row.get(key) for key in fields]], ensure_ascii=False).encode()
    ).digest()


def _reusable_text_spans(root, tokenizer_path):
    """Verify a prior standalone encoding and index immutable document spans."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("format") != "document-ragged-v2"
        or manifest.get("pretrain_text_encoding")
        != "literal_text_without_special_token_matching_v1"
        or any(node["examples"] for node in manifest["stages"]["sft"].values())
    ):
        raise ValueError("reuse requires a standalone literal pretraining text encoding")
    checksum = sha256(tokenizer_path)
    if sha256(root / "tokenizer.json") != checksum or manifest["tokenizer"]["sha256"] != checksum:
        raise ValueError("reused encoding tokenizer differs")
    spans = {}
    for node in manifest["stages"]["pretrain"].values():
        if node.get("format") != "document-ragged-v2":
            raise ValueError("reuse requires complete document spans")
        paths = {}
        for key, digest_key in (
            ("file", "sha256"),
            ("labels_file", "labels_sha256"),
            ("index_file", "index_sha256"),
            ("metadata_file", "metadata_sha256"),
        ):
            path = (root / node[key]).resolve()
            if not path.is_relative_to(root) or sha256(path) != node[digest_key]:
                raise ValueError("reused encoding file path/hash differs")
            paths[key] = path
        positions = node["stored_positions"]
        if any(paths[key].stat().st_size != positions * 4 for key in ("file", "labels_file")):
            raise ValueError("reused token storage size differs")
        indexes = np.load(paths["index_file"], allow_pickle=False)
        if indexes.dtype != np.dtype("int64") or indexes.shape != (node["examples"], 2):
            raise ValueError("reused document index shape/type differs")
        tokens: np.ndarray | list[int] = (
            np.memmap(paths["file"], mode="r", dtype=np.int32) if positions else []
        )
        labels: np.ndarray | list[int] = (
            np.memmap(paths["labels_file"], mode="r", dtype=np.int32) if positions else []
        )
        end = 0
        with paths["metadata_file"].open() as stream:
            for (offset, length), line in zip(indexes, stream, strict=True):
                offset, length = int(offset), int(length)
                if offset != end or length < 2 or offset + length > positions:
                    raise ValueError("reused documents have gaps, overlaps or invalid lengths")
                row = json.loads(line)
                identity = row["sample_id"]
                if (
                    identity in spans
                    or row["stage"] != "pretrain"
                    or row.get("media")
                    or row.get("turns")
                ):
                    raise ValueError("reused documents must be distinct pure pretraining text")
                spans[identity] = (_text_encoding_identity(row), tokens, labels, offset, length)
                end += length
        if end != positions:
            raise ValueError("reused token storage has unindexed positions")
    return spans, dict(
        path=str(root),
        manifest_sha256=sha256(root / "manifest.json"),
        corpus_sha256=manifest["corpus_sha256"],
        tokenizer_sha256=checksum,
    )


def encode_corpus(
    corpus_root,
    tokenizer_path,
    output,
    *,
    max_length=4096,
    max_gib=None,
    reuse_encoding=None,
    compact_metadata=False,
):
    """Encode text records once; media records require the native processor path."""
    root, output = Path(corpus_root), Path(output)
    if output.exists():
        raise FileExistsError("encoded corpus is immutable; select a new path")
    if max_gib is not None and max_gib <= 0:
        raise ValueError("encoding disk budget must be positive")
    reusable, reuse_binding = (
        ({}, None)
        if reuse_encoding is None
        else (_reusable_text_spans(reuse_encoding, tokenizer_path))
    )
    max_bytes = int(max_gib * GIB) if max_gib is not None else None
    # Include tokenizer, index headers and final manifests in a conservative bound.
    accounted_bytes = Path(tokenizer_path).stat().st_size + 1024**2
    if max_bytes is not None:
        if accounted_bytes > max_bytes:
            raise ValueError("encoding disk budget cannot hold tokenizer and metadata")
        require_space(output, max_bytes, reserve_bytes=80 * GIB)
    output.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    (output / "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    db = open_corpus(root)
    manifest = dict(
        schema_version=2,
        sequence_length=max_length,
        format="document-ragged-v2",
        pretrain_text_encoding="literal_text_without_special_token_matching_v1",
        corpus_sha256=sha256(root / "corpus-manifest.json"),
        stages={},
        tokenizer=dict(vocab_size=tokenizer.get_vocab_size(), sha256=sha256(tokenizer_path)),
    )
    if reuse_binding is not None:
        manifest["reused_encoding"] = reuse_binding
    if compact_metadata:
        manifest["text_metadata"] = "identity_and_provenance_without_text_v1"
    for stage in ("pretrain", "sft"):
        # Raw documents can quote protocol spellings. Encode their original bytes
        # as ordinary BPE pieces; only this writer inserts document BOS/EOS.
        # This option is distinct from add_special_tokens (post-processing).
        tokenizer.encode_special_tokens = stage == "pretrain"
        manifest["stages"][stage] = {}
        for split in ("train", "val", "test"):
            prefix = f"{stage}.{split}"
            token_path, label_path = output / f"{prefix}.bin", output / f"{prefix}.labels.bin"
            index, token_count, supervised, identity_tokens = [], 0, 0, 0
            rejected: Counter[str] = Counter()
            template_counts: Counter[str] = Counter()
            reused_documents = encoded_documents = 0
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
                        old = reusable.get(row["sample_id"])
                        if old is not None:
                            identity, old_tokens, old_labels, offset, length = old
                            if _text_encoding_identity(row) != identity:
                                raise ValueError("reused document text or provenance differs")
                            ids = old_tokens[offset : offset + length]
                            targets = old_labels[offset : offset + length]
                            if (
                                ids[0] != 1
                                or ids[-1] != 2
                                or targets[0] != -100
                                or not np.array_equal(targets[1:], ids[1:])
                            ):
                                raise ValueError("reused document boundaries or labels differ")
                            reused_documents += 1
                        else:
                            ids = [
                                1,
                                *tokenizer.encode(row["text"], add_special_tokens=False).ids,
                                2,
                            ]
                            targets = ids.copy()
                            targets[0] = -100
                            encoded_documents += 1
                    else:
                        ids, targets = chat_tokens(
                            row["turns"], tokenizer, mode=row.get("mode"), effort=row.get("effort")
                        )
                        if len(ids) > max_length:
                            rejected["complete_answer_exceeds_largest_bucket"] += 1
                            continue
                    ce = (
                        int(np.count_nonzero(targets[1:] != -100))
                        if isinstance(targets, np.ndarray)
                        else sum(value != -100 for value in targets[1:])
                    )
                    if stage == "sft" and IDENTITY.search(
                        next((t["content"] for t in row["turns"] if t["role"] == "user"), "")
                    ):
                        # Conservative streaming quota: may retain less than 0.5%, never more.
                        if identity_tokens + ce > 0.005 * (supervised + ce):
                            rejected["identity_token_cap"] += 1
                            continue
                        identity_tokens += ce
                    if stage == "pretrain" and compact_metadata:
                        kept = {
                            key: value
                            for key, value in row.items()
                            if key not in {"text", "turns", "source_metadata"}
                        }
                        kept["encoded_text_sha256"] = hashlib.sha256(
                            row["text"].encode()
                        ).hexdigest()
                        payload = json.dumps(kept, ensure_ascii=False)
                    incoming = len(ids) * 8 + len(payload.encode()) + 512
                    if max_bytes is not None and accounted_bytes + incoming > max_bytes:
                        db.close()
                        raise ValueError("encoding disk budget reached; output remains unadmitted")
                    with reserve_write(token_path, incoming):
                        np.asarray(ids, dtype=np.int32).tofile(tokens)
                        np.asarray(targets, dtype=np.int32).tofile(labels)
                        metadata.write(payload + "\n")
                    accounted_bytes += incoming
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
            if reuse_binding is not None and stage == "pretrain":
                manifest["stages"][stage][split]["document_reuse"] = dict(
                    copied=reused_documents,
                    tokenized=encoded_documents,
                )
    db.close()
    if max_bytes is not None:
        manifest["storage_bound"] = dict(max_bytes=max_bytes, accounted_bytes=accounted_bytes)
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
