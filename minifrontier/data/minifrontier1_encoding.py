"""Disk-bounded native token/label/position shards and per-record CE accounting."""

import contextlib
import json
import shutil
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import (
    SPECIAL_TOKENS,
    RecordDataset,
    digest,
    encode_record,
    prepare_media,
    safe_text,
    validate_record,
    write_json,
)
from minifrontier.data.partitions import open_corpus
from minifrontier.models.minifrontier1.processing import PROCESSOR_VERSION, token_metadata
from minifrontier.storage import require_space


def encode_dataset(data, output, config, *, max_gib=16, compact=False, shard_tokens=64_000_000):
    if compact:
        return encode_compact_dataset(
            data, output, config, max_gib=max_gib, shard_tokens=shard_tokens
        )
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or max_gib <= 0:
        raise ValueError("choose a new encoding version and positive disk budget")
    output.mkdir(parents=True)
    require_space(output, 16 * 1024**2)
    token_dtype = "uint16" if config.vocab_size <= 65536 else "uint32"
    splits = {}
    used_bytes = 0
    for split in ("train", "val", "test"):
        dataset = RecordDataset(data, split, config)
        counts: Counter[str] = Counter()
        unique_media = set()
        domains: Counter[str] = Counter()
        paths = {
            key: output / f"{split}.{key}"
            for key in ("ids.bin", "labels.bin", "positions.bin", "index.jsonl")
        }
        with (
            paths["ids.bin"].open("wb") as ids_file,
            paths["labels.bin"].open("wb") as labels_file,
            paths["positions.bin"].open("wb") as positions_file,
            paths["index.jsonl"].open("w") as index_file,
        ):
            for i in range(len(dataset)):
                item = dataset[i]
                metadata = token_metadata(item["input_ids"], config, item["media"])
                ids = item["input_ids"][0].numpy().astype(token_dtype)
                labels = item["labels"][0].numpy().astype("int32")
                positions = metadata["position_ids"][:, 0].T.numpy().astype("int32")
                spans = [
                    {
                        key: value.tolist() if hasattr(value, "tolist") else value
                        for key, value in span.items()
                        if key != "patches"
                    }
                    for span in item["media"]
                ]
                entry = dict(
                    sample_id=item["sample_id"],
                    split_group=item["split_group"],
                    domain=item["domain"],
                    offset=counts["input_tokens"],
                    input_tokens=len(ids),
                    ce_tokens=int((labels[1:] != -100).sum()),
                    vision_tokens=sum(s["feature_count"] for s in spans),
                    media=spans,
                    segment_ids=metadata["segment_ids"][0].tolist(),
                    linear_positions=metadata["linear_positions"][0].tolist(),
                    modality=metadata["modality"][0].tolist(),
                    media_ids=metadata["media_ids"][0].tolist(),
                    encoded_sha256=digest([ids.tolist(), labels.tolist(), positions.tolist()]),
                )
                raw = json.dumps(entry) + "\n"
                incoming = ids.nbytes + labels.nbytes + positions.nbytes + len(raw.encode())
                if used_bytes + incoming > max_gib * 1024**3:
                    raise ValueError(
                        "encoding reached its disk budget; incomplete shards remain unadmitted"
                    )
                require_space(output, incoming)
                ids.tofile(ids_file)
                labels.tofile(labels_file)
                positions.tofile(positions_file)
                index_file.write(raw)
                used_bytes += incoming
                counts.update(
                    records=1,
                    input_tokens=len(ids),
                    ce_tokens=entry["ce_tokens"],
                    vision_tokens=entry["vision_tokens"],
                    media_exposures=item["media_exposures"],
                )
                domains[item["domain"]] += entry["ce_tokens"]
                unique_media.update(item["media_hashes"])
        splits[split] = dict(
            counts=counts,
            domain_ce=dict(domains),
            unique_media=len(unique_media),
            files={k: dict(name=p.name, sha256=sha256(p)) for k, p in paths.items()},
        )
    result = dict(
        format="mf1-native-shards-v1",
        source_manifest_sha256=sha256(data / "manifest.json"),
        tokenizer_sha256=sha256(data / "tokenizer.json"),
        token_dtype=token_dtype,
        label_dtype="int32",
        position_dtype="int32",
        position_axes=["t", "h", "w"],
        bytes=used_bytes,
        splits=splits,
        formal_admission=False,
    )
    write_json(output / "manifest.json", result)
    return result


# Per-token storage is IDs (2 or 4 bytes) plus one loss-mask bit. There are no
# dense labels, position axes, segment IDs or modality arrays on disk.
INDEX = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u4"),
        ("mask_offset", "<u8"),
        ("metadata_offset", "<u8"),
        ("metadata_bytes", "<u4"),
        ("domain", "<u2"),
    ]
)
FORMAT = "mf1-compact-shards-v2"


def _span_metadata(span):
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v
        for k, v in span.items()
        if k != "patches"
    }


def _close_part(parts, handles, paths, counts):
    if not handles:
        return
    for handle in handles.values():
        handle.close()
    parts.append(
        dict(
            counts=dict(counts),
            files={
                k: dict(name=p.name, bytes=p.stat().st_size, sha256=sha256(p))
                for k, p in paths.items()
            },
        )
    )
    handles.clear()


def encode_compact_dataset(data, output, config, *, max_gib=16, shard_tokens=64_000_000):
    """Encode immutable bounded shards, retaining only sparse media descriptors."""
    data = Path(data).resolve()
    source_manifest = json.loads((data / "manifest.json").read_text())

    def datasets():
        for split in ("train", "val", "test"):
            dataset = RecordDataset(data, split, config)
            yield split, ((dataset.record(i), dataset[i]) for i in range(len(dataset)))

    return _encode_compact_items(
        datasets(),
        output,
        config,
        data / "tokenizer.json",
        source_manifest,
        source_manifest_sha256=sha256(data / "manifest.json"),
        media_root=(data / source_manifest.get("media_root", ".")).resolve(),
        max_gib=max_gib,
        shard_tokens=shard_tokens,
    )


def encode_canonical_text(
    corpus, tokenizer_path, output, config, *, max_gib=3, shard_tokens=64_000_000
):
    """Store each complete canonical document once, without a JSONL/raw-text copy.

    Text windows are chosen at consumption time; no EOS is inserted inside a
    document and source groups retain their effective train/val/test partition.
    This text component remains unadmitted until the full phase data is assembled.
    """
    corpus, tokenizer_path = Path(corpus).resolve(), Path(tokenizer_path).resolve()
    audit = json.loads((corpus / "source-audit.json").read_text())
    if (
        not audit.get("formal_admission")
        and audit.get("status") != "candidate_slice_complete_pending_admission"
    ):
        raise ValueError("canonical text construction must finish before encoding")
    if (corpus / "corpus.sqlite").exists():
        manifest = json.loads((corpus / "corpus-manifest.json").read_text())
        if sha256(corpus / "corpus.sqlite") != manifest["database_sha256"]:
            raise ValueError("canonical database checksum differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > config.vocab_size or any(
        tokenizer.token_to_id(s) != i for i, s in enumerate(SPECIAL_TOKENS)
    ):
        raise ValueError(
            "canonical MF1 encoding needs its control mapping and a fitting vocabulary"
        )
    domain_map = dict(
        zh_edu="zh_general",
        en_edu="en_general",
        code="code",
        verified_math_science="math",
        dialogue="structured",
    )
    source_manifest = dict(
        kind="canonical_text_component",
        formal_admission=False,
        source_corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        source_audit_sha256=sha256(corpus / "source-audit.json"),
        boundary_protocol="BOS + control-escaped text + EOS; one-token context overlap between windows",
        raw_text_copied=False,
    )

    def records(db, split):
        for payload, group in db.execute(
            "SELECT payload,group_root FROM samples WHERE split=? ORDER BY id", (split,)
        ):
            row = json.loads(payload)
            if row["stage"] != "pretrain" or row.get("media") or row.get("turns"):
                raise ValueError("canonical text component only accepts pure pretraining documents")
            ids = torch.tensor([[1, *safe_text(tokenizer, row["text"]), 2]])
            labels = ids.clone()
            labels[:, 0] = -100
            origin = {
                k: row[k] for k in ("source", "revision", "item_id", "license", "content_hash")
            }
            yield (
                dict(origin=origin),
                dict(
                    input_ids=ids,
                    labels=labels,
                    sample_id=row["sample_id"],
                    split_group=group,
                    domain=domain_map[row["task"]],
                    media=[],
                    media_exposures=0,
                    text_document=True,
                ),
            )

    with contextlib.closing(open_corpus(corpus)) as db:
        return _encode_compact_items(
            ((split, records(db, split)) for split in ("train", "val", "test")),
            output,
            config,
            tokenizer_path,
            source_manifest,
            source_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
            media_root=corpus,
            max_gib=max_gib,
            shard_tokens=shard_tokens,
        )


def canonical_image_record(row, group, *, max_features, document_tiles=False):
    """Adapt a complete canonical image QA without rebuilding text or source pixels."""
    if (
        row.get("stage") != "pretrain"
        or row.get("task") not in {"caption", "vqa", "ocr_document", "chart_table"}
        or len(row.get("media", [])) != 1
        or row["media"][0].get("kind") != "image"
        or any(
            not isinstance(row.get(k), str) or not row[k].strip()
            for k in ("visual_question", "visual_answer")
        )
        or max_features < 1
    ):
        raise ValueError("canonical image encoding requires one image and a complete grounded QA")
    original = row["media"][0]
    resource = dict(
        media_id=original["rgb_sha256"],
        uri=original["path"],
        sha256=original["sha256"],
        rgb_sha256=original["rgb_sha256"],
        width=original["width"],
        height=original["height"],
        max_features=max_features,
        representation="document"
        if document_tiles and row["task"] == "ocr_document"
        else "standard",
    )
    return dict(
        sample_id=row["sample_id"],
        split_group=group,
        language=row["lang"],
        domain=row["task"],
        source=dict(dataset=row["source"], revision=row["revision"], record_id=row["item_id"]),
        provenance=dict(license_record=row["license"], transform="canonical-image-qa-to-mf1-v1"),
        origin={k: row[k] for k in ("source", "revision", "item_id", "license", "content_hash")},
        supervision=dict(type="answer_ce"),
        media=[resource],
        messages=[
            dict(
                role="user",
                content=[
                    dict(type="image", media_id=resource["media_id"]),
                    dict(type="text", text=row["visual_question"]),
                ],
            ),
            dict(
                role="assistant",
                channel="final",
                content=[dict(type="text", text=row["visual_answer"])],
            ),
        ],
    )


def encode_canonical_images(
    corpus,
    tokenizer_path,
    output,
    config,
    *,
    max_features,
    document_tiles=False,
    max_gib=1,
    shard_tokens=64_000_000,
):
    """Keep complete grounded answers in compact shards, with source pixels shared.

    The result is an unadmitted media component. P0 uses max_features=49 and no
    document tiles; later stages explicitly rebuild their media spans at higher
    resolution. Context overflow is an error, never silent answer truncation.
    """
    corpus, tokenizer_path = Path(corpus).resolve(), Path(tokenizer_path).resolve()
    audit = json.loads((corpus / "source-audit.json").read_text())
    if audit.get("status") not in {
        "candidate_slice_complete_pending_admission",
        "candidate_inventory_below_target",
    } and not audit.get("formal_admission"):
        raise ValueError("canonical media construction must finish before encoding")
    manifest = json.loads((corpus / "corpus-manifest.json").read_text())
    if sha256(corpus / "corpus.sqlite") != manifest["database_sha256"]:
        raise ValueError("canonical media database checksum differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > config.vocab_size or any(
        tokenizer.token_to_id(s) != i for i, s in enumerate(SPECIAL_TOKENS)
    ):
        raise ValueError("canonical media needs the MF1 control mapping and fitting vocabulary")
    if not 1 <= max_features <= config.protected_media_tokens:
        raise ValueError("image feature limit exceeds the model media budget")
    source_manifest = dict(
        kind="canonical_image_component",
        formal_admission=False,
        source_corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        source_audit_sha256=sha256(corpus / "source-audit.json"),
        raw_media_copied=False,
        raw_text_copied=False,
        image_transform=dict(max_features=max_features, document_tiles=document_tiles),
        encoding_processor_sha256=sha256(__file__),
        complete_record_length_buckets={},
    )

    def records(db, split):
        buckets: dict[str, dict[str, Counter[str]]] = {}
        source_manifest["complete_record_length_buckets"][split] = buckets
        for payload, group in db.execute(
            "SELECT payload,group_root FROM samples WHERE split=? ORDER BY id", (split,)
        ):
            record = canonical_image_record(
                json.loads(payload), group, max_features=max_features, document_tiles=document_tiles
            )
            validate_record(record, corpus)
            item = encode_record(record, tokenizer, config, corpus)
            length = item["input_ids"].shape[1]
            bucket = str(
                next((n for n in (512, 1024, 2048, 4096, 8192) if length <= n), "over_8192")
            )
            counts = buckets.setdefault(record["domain"], {}).setdefault(bucket, Counter())
            counts.update(
                records=1,
                input_tokens=length,
                ce_tokens=int(item["labels"][:, 1:].ne(-100).sum()),
                media_exposures=item["media_exposures"],
                vision_tokens=sum(s["feature_count"] for s in item["media"]),
            )
            yield record, item

    with contextlib.closing(open_corpus(corpus)) as db:
        return _encode_compact_items(
            ((split, records(db, split)) for split in ("train", "val", "test")),
            output,
            config,
            tokenizer_path,
            source_manifest,
            source_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
            media_root=corpus,
            max_gib=max_gib,
            shard_tokens=shard_tokens,
        )


def _encode_compact_items(
    datasets,
    output,
    config,
    tokenizer_path,
    source_manifest,
    *,
    source_manifest_sha256,
    media_root,
    max_gib,
    shard_tokens,
):
    output = Path(output).resolve()
    if output.exists() or max_gib <= 0 or shard_tokens < 1:
        raise ValueError("choose a new output and positive shard/storage budgets")
    require_space(output, min(int(max_gib * 1024**3), 16 * 1024**2))
    output.mkdir(parents=True)
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")
    dtype = np.dtype("<u2" if config.vocab_size <= 65536 else "<u4")
    domains: list[str] = []
    splits = {}
    used = (output / "tokenizer.json").stat().st_size
    for split, items in datasets:
        parts: list[dict[str, Any]] = []
        handles: dict[str, BinaryIO] = {}
        paths: dict[str, Path] = {}
        counts: Counter[str] = Counter()
        domain_ce: Counter[str] = Counter()

        try:
            for record, item in items:
                ids, labels = item["input_ids"][0].numpy(), item["labels"][0].numpy()
                if ids.min() < 0 or ids.max() >= config.vocab_size:
                    raise ValueError(
                        "token ID outside the model vocabulary; refuse integer truncation"
                    )
                if not np.all((labels == -100) | (labels == ids)):
                    raise ValueError("compact format needs labels equal to IDs or -100")
                text_document = bool(item.get("text_document"))
                if handles and (
                    counts["input_tokens"] + len(ids) > shard_tokens
                    or text_document != bool(counts["text_documents"])
                ):
                    _close_part(parts, handles, paths, counts)
                if not handles:
                    counts = Counter()
                    paths = {
                        k: output / f"{split}-{len(parts):05d}.{k}"
                        for k in ("tokens.bin", "mask.bin", "index.bin", "metadata.jsonl")
                    }
                    handles = {k: p.open("wb") for k, p in paths.items()}
                domain = item["domain"]
                if domain not in domains:
                    if len(domains) >= 65536:
                        raise ValueError("too many domains for compact index")
                    domains.append(domain)
                spans = [_span_metadata(s) for s in item["media"]]
                entry = dict(
                    sample_id=item["sample_id"],
                    split_group=item["split_group"],
                    media=spans,
                    resources=record.get("media", []),
                )
                if "origin" in record:
                    entry["origin"] = record["origin"]
                raw = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                mask = np.packbits(labels != -100, bitorder="little")
                index = np.array(
                    [
                        (
                            counts["input_tokens"],
                            len(ids),
                            handles["mask.bin"].tell(),
                            handles["metadata.jsonl"].tell(),
                            len(raw),
                            domains.index(domain),
                        )
                    ],
                    dtype=INDEX,
                )
                incoming = len(ids) * dtype.itemsize + mask.nbytes + len(raw) + index.nbytes
                if used + incoming > max_gib * 1024**3:
                    raise ValueError(
                        "compact encoding disk budget reached; output remains unadmitted"
                    )
                require_space(output, incoming)
                ids.astype(dtype).tofile(handles["tokens.bin"])
                mask.tofile(handles["mask.bin"])
                index.tofile(handles["index.bin"])
                handles["metadata.jsonl"].write(raw)
                used += incoming
                counts.update(
                    records=1,
                    input_tokens=len(ids),
                    ce_tokens=int((labels[1:] != -100).sum()),
                    media_exposures=item["media_exposures"],
                    vision_tokens=sum(s["feature_count"] for s in spans),
                )
                if text_document:
                    counts["text_documents"] += 1
                domain_ce[domain] += int((labels[1:] != -100).sum())
            _close_part(parts, handles, paths, counts)
        finally:
            for handle in handles.values():
                handle.close()
        splits[split] = dict(
            parts=parts,
            counts=dict(sum((Counter(p["counts"]) for p in parts), Counter())),
            domain_ce=dict(domain_ce),
        )
    manifest = dict(
        source_manifest,
        format=FORMAT,
        source_manifest_sha256=source_manifest_sha256,
        config_sha256=digest(asdict(config)),
        processor_version=PROCESSOR_VERSION,
        tokenizer_sha256=sha256(output / "tokenizer.json"),
        token_dtype=dtype.str,
        domains=domains,
        splits=splits,
        shard_tokens=shard_tokens,
        bytes=used,
        media_root=str(media_root),
        formal_admission=False,
    )
    # Original record file paths are provenance, not dependencies of this loader.
    manifest.pop("files", None)
    write_json(output / "manifest.json", manifest)
    return manifest


class CompactDataset:
    """Bound token and index mappings separately; validate immutable files once per identity."""

    def __init__(self, root, split, config):
        self.root, self.config = Path(root).resolve(), config
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("format") != FORMAT
            or self.manifest["config_sha256"] != digest(asdict(config))
            or self.manifest["processor_version"] != PROCESSOR_VERSION
        ):
            raise ValueError("compact data config/processor identity differs")
        if sha256(self.root / "tokenizer.json") != self.manifest["tokenizer_sha256"]:
            raise ValueError("compact tokenizer checksum differs")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.media_root = Path(self.manifest["media_root"]).resolve()
        self.parts = self.manifest["splits"][split]["parts"]
        self.ends = np.cumsum([p["counts"]["records"] for p in self.parts]).tolist()
        self._verified_files: dict[tuple[int, str], tuple[int, ...]] = {}
        self._open_index = lru_cache(maxsize=4)(self._map_index)
        self._open_part = lru_cache(maxsize=4)(self._map_part)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _file(self, part, name):
        path = (self.root / part["files"][name]["name"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("compact shard path escapes dataset")
        return path

    def _validated_file(self, number, key):
        part = self.parts[number]
        entry, path = part["files"][key], self._file(part, key)

        def identity():
            stat = path.stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        current = identity()
        if self._verified_files.get((number, key)) != current:
            if current[2] != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError("compact shard checksum differs")
            if identity() != current:
                raise ValueError("compact shard changed during checksum validation")
            self._verified_files[number, key] = current
        return path

    def _map_index(self, number):
        return np.memmap(self._validated_file(number, "index.bin"), mode="r", dtype=INDEX)

    def _map_part(self, number):
        return {
            key: np.memmap(self._validated_file(number, key), mode="r", dtype=dtype)
            for key, dtype in (
                ("tokens.bin", self.manifest["token_dtype"]),
                ("mask.bin", np.uint8),
                ("metadata.jsonl", np.uint8),
            )
        }

    def _entry(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        row = index - (self.ends[part - 1] if part else 0)
        return part, self._open_index(part)[row]

    def length_at(self, index):
        return int(self._entry(index)[1]["length"])

    def domain_at(self, index):
        return self.manifest["domains"][int(self._entry(index)[1]["domain"])]

    def windowable_at(self, index):
        counts = self.parts[self._entry(index)[0]]["counts"]
        return counts.get("text_documents", 0) == counts["records"]

    def window_at(self, index, start, capacity):
        if not self.windowable_at(index):
            raise ValueError("only canonical continuation documents may be windowed")
        if not 2 <= capacity <= self.config.max_position_embeddings:
            raise ValueError("text window capacity is outside the model context")
        if not 0 <= start < self.length_at(index) - 1:
            raise ValueError("text window starts outside its document's CE positions")
        item = self._read_item(index, start=start, capacity=capacity)
        if item["media"]:
            raise ValueError("a text window cannot contain media spans")
        item["labels"][:, 0] = -100
        item["source_sample_id"] = item["sample_id"]
        item["sample_id"] += f"@{start}:{start + item['input_ids'].shape[1]}"
        item["document_offset"] = start
        return item

    def __getitem__(self, index):
        return self._read_item(index)

    def _read_item(self, index, *, start=0, capacity=None):
        part, entry = self._entry(index)
        arrays = self._open_part(part)
        length = int(entry["length"]) - start
        if capacity is not None:
            length = min(length, capacity)
        offset = int(entry["offset"]) + start
        ids = torch.from_numpy(arrays["tokens.bin"][offset : offset + length].astype(np.int64))[
            None
        ]
        mask_offset, bit_offset = int(entry["mask_offset"]) + start // 8, start % 8
        mask = np.unpackbits(
            arrays["mask.bin"][mask_offset : mask_offset + (bit_offset + length + 7) // 8],
            bitorder="little",
            count=bit_offset + length,
        )[bit_offset:].astype(bool)
        labels = ids.clone().masked_fill(~torch.from_numpy(mask)[None], -100)
        metadata_offset = int(entry["metadata_offset"])
        metadata = json.loads(
            bytes(
                arrays["metadata.jsonl"][
                    metadata_offset : metadata_offset + int(entry["metadata_bytes"])
                ]
            )
        )
        resources = {r["media_id"]: r for r in metadata["resources"]}
        spans, loaded = [], {}
        remaining = self.config.protected_media_tokens
        for saved in metadata["media"]:
            identity = saved["media_id"]
            if identity not in loaded:
                resource = resources[identity]
                for uri, expected in zip(
                    resource.get("frames", [resource.get("uri")]),
                    resource.get("frame_sha256", [resource.get("sha256")]),
                    strict=True,
                ):
                    path = (self.media_root / uri).resolve()
                    if not path.is_relative_to(self.media_root) or sha256(path) != expected:
                        raise ValueError("compact media source/hash differs")
                loaded[identity] = iter(
                    prepare_media(resource, self.config, self.media_root, remaining)
                )
            sample = next(loaded[identity])
            if any(_span_metadata(sample)[k] != saved[k] for k in _span_metadata(sample)):
                raise ValueError("media transform differs from frozen compact span")
            restored = dict(saved, patches=sample["patches"], grid_thw=sample["grid_thw"])
            spans.append(restored)
            remaining -= sample["feature_count"]
        return dict(
            input_ids=ids,
            labels=labels,
            media=spans,
            sample_id=metadata["sample_id"],
            split_group=metadata["split_group"],
            domain=self.manifest["domains"][int(entry["domain"])],
            media_hashes=[s["source_sha256"] for s in spans],
            media_exposures=len(resources),
        )


def open_dataset(root, split, config):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    return (CompactDataset if manifest.get("format") == FORMAT else RecordDataset)(
        root, split, config
    )


def evaluation_items(dataset, *, limit=0, max_length=None):
    """Evaluate each document's CE positions once at a declared context length."""
    for index in range(min(len(dataset), limit or len(dataset))):
        if hasattr(dataset, "windowable_at") and dataset.windowable_at(index):
            capacity = max_length or dataset.config.max_position_embeddings
            if capacity < 2:
                raise ValueError("document evaluation needs at least two token positions")
            for start in range(0, dataset.length_at(index) - 1, capacity - 1):
                yield dataset.window_at(index, start, capacity)
        else:
            item = dataset[index]
            if max_length is not None and item["input_ids"].shape[1] > max_length:
                raise ValueError("complete validation record exceeds the declared context")
            yield item
