"""Immutable native-media records alongside shared memory-mapped text documents."""

import contextlib
import importlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from minifrontier.chat_controls import record_template, update_manifest
from minifrontier.data import sha256
from minifrontier.data.corpus import DocumentDataset, encode_corpus
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.multimodal import prepare_record
from minifrontier.storage import GIB, require_space, reserve_write


def processor_identity(family):
    return {
        name: sha256(importlib.import_module(name).__file__)
        for name in (
            "minifrontier.multimodal",
            f"minifrontier.models.{family}.processing",
            f"minifrontier.models.{family}.upstream_processing",
        )
    }


def _shared_text_partitions(root, expected_hash):
    """Bind OCR source identities to the actual shared text partition, including after merges."""
    root = Path(root).resolve()
    if sha256(root / "manifest.json") != expected_hash:
        raise ValueError("shared text manifest changed during derivative validation")
    manifest = json.loads((root / "manifest.json").read_text())
    result = {}
    for stage, splits in manifest["stages"].items():
        for split, node in splits.items():
            path = (root / node["metadata_file"]).resolve()
            if not path.is_relative_to(root) or sha256(path) != node["metadata_sha256"]:
                raise ValueError("shared text identity metadata hash/path differs")
            count = 0
            with path.open() as stream:
                for line in stream:
                    identity = json.loads(line)["sample_id"]
                    if identity in result:
                        raise ValueError("duplicate shared text source identity")
                    result[identity] = (stage, split)
                    count += 1
            if count != node["examples"]:
                raise ValueError("shared text identity count differs")
    return result


def _check_text_origin(row, split, partitions):
    origin = row["text_origin"]
    if origin.get("split") != split or partitions.get(origin.get("sample_id")) != (
        "pretrain",
        split,
    ):
        raise ValueError("OCR source text identity is missing or crosses shared text splits")


def encode_native(
    corpus_root,
    tokenizer_path,
    output,
    family,
    *,
    max_length=4096,
    max_features=256,
    media_root=None,
    model_vocab_size=None,
    text_encoding=None,
    max_gib=None,
    min_pixels=None,
):
    output, corpus_root = Path(output).resolve(), Path(corpus_root).resolve()
    if family not in {"minikimik3", "miniqwen4", "minideepseekv4"}:
        raise ValueError("unknown native media family")
    if output.exists():
        raise FileExistsError("native encoding requires a new output directory")
    if max_length < 2 or max_features < 1 or (max_gib is not None and max_gib <= 0):
        raise ValueError("invalid native encoding bounds")
    if min_pixels is not None and (min_pixels < 1 or family == "minikimik3"):
        raise ValueError("min_pixels is a positive Qwen/DeepSeek resize bound")
    maximum = None if max_gib is None else int(max_gib * GIB)
    header_reserve = 128 * 1024
    reserve = (
        int(float(os.environ.get("MINIFRONTIER_MIN_FREE_GIB", "80")) * GIB)
        if maximum is not None
        else None
    )
    initial = Path(tokenizer_path).stat().st_size + header_reserve
    if maximum is not None:
        if initial > maximum:
            raise ValueError("native encoding disk budget cannot hold tokenizer and metadata")
        require_space(output, maximum, reserve_bytes=reserve)
    if text_encoding is None:
        manifest = encode_corpus(
            corpus_root, tokenizer_path, output, max_length=max_length, max_gib=max_gib
        )
    else:
        shared = Path(text_encoding).resolve()
        manifest = json.loads((shared / "manifest.json").read_text())
        if manifest.get("format") != "document-ragged-v2" or any(
            split.get("format") != "document-ragged-v2"
            for stage in manifest["stages"].values()
            for split in stage.values()
        ):
            raise ValueError("shared native text must be a standalone document encoding")
        checksum = sha256(tokenizer_path)
        if (
            sha256(shared / "tokenizer.json") != checksum
            or manifest["tokenizer"]["sha256"] != checksum
        ):
            raise ValueError("shared text tokenizer differs from native encoding")
        manifest["text_source"] = dict(
            path=os.path.relpath(shared, output), manifest_sha256=sha256(shared / "manifest.json")
        )
        output.mkdir(parents=True)
        (output / "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    # Publish the hybrid manifest only after the complete media encoding succeeds.
    (output / "manifest.json").unlink(missing_ok=True)
    accounted = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = model_vocab_size or tokenizer.get_vocab_size()
    text_partitions = None
    with contextlib.closing(open_corpus(corpus_root)) as db:
        pixels = Path(media_root).resolve() if media_root is not None else corpus_storage_root(db)
        for stage in ("pretrain", "sft"):
            for split in ("train", "val", "test"):
                path = output / f"{stage}.{split}.media.jsonl"
                offsets = []
                rejected: Counter[str] = Counter()
                template_counts: Counter[str] = Counter()
                domain_ce: Counter[str] = Counter()
                examples = ce = image_count = video_count = frame_count = features = 0
                records = db.execute(
                    "SELECT payload,group_root FROM samples WHERE stage=? AND split=? ORDER BY id",
                    (stage, split),
                )
                with path.open("wb") as stream:
                    for payload, group in records:
                        row = json.loads(payload)
                        if not row.get("media"):
                            continue
                        if "text_origin" in row and text_encoding is not None:
                            if text_partitions is None:
                                text_partitions = _shared_text_partitions(
                                    shared, manifest["text_source"]["manifest_sha256"]
                                )
                            _check_text_origin(row, split, text_partitions)
                        prepared = prepare_record(
                            row,
                            tokenizer,
                            family,
                            root=pixels,
                            max_features=max_features,
                            model_vocab_size=vocab,
                            min_pixels=min_pixels,
                        )
                        if prepared.input_ids.shape[1] > max_length:
                            rejected["complete_media_answer_exceeds_bucket"] += 1
                            continue
                        serialized = (
                            json.dumps(
                                dict(
                                    record=row,
                                    split_group=group,
                                    expected_ids=prepared.input_ids[0].tolist(),
                                    expected_labels=prepared.labels[0].tolist(),
                                ),
                                ensure_ascii=False,
                            ).encode()
                            + b"\n"
                        )
                        incoming = len(serialized) + 24  # Three int64 index columns.
                        if maximum is not None and accounted + incoming + header_reserve > maximum:
                            raise ValueError(
                                "native encoding disk budget reached; output remains unadmitted"
                            )
                        with reserve_write(path, incoming, reserve_bytes=reserve):
                            offsets.append(
                                (stream.tell(), len(serialized), prepared.input_ids.shape[1])
                            )
                            stream.write(serialized)
                        accounted += incoming
                        examples += 1
                        if stage == "sft":
                            template_counts[record_template(row)] += 1
                        count = int(prepared.labels[:, 1:].ne(-100).sum())
                        ce += count
                        domain_ce[row["task"]] += count
                        image_count += prepared.image_count
                        video_count += prepared.video_count
                        frame_count += prepared.frame_count
                        features += prepared.image_features
                index = path.with_suffix(".index.npy")
                with reserve_write(index, 128 + 24 * len(offsets), reserve_bytes=reserve):
                    np.save(index, np.asarray(offsets, dtype=np.int64).reshape(-1, 3))
                text = manifest["stages"][stage][split]
                media = dict(
                    file=path.name,
                    sha256=sha256(path),
                    index_file=index.name,
                    index_sha256=sha256(index),
                    examples=examples,
                    supervised_tokens=ce,
                    domain_ce=dict(domain_ce),
                    image_occurrences=image_count,
                    video_examples=video_count,
                    frames=frame_count,
                    image_features=features,
                    rejected=dict(rejected),
                    chat_template_counts=dict(template_counts),
                )
                manifest["stages"][stage][split] = dict(
                    format="hybrid-native-v2",
                    text=text,
                    media=media,
                    examples=text["examples"] + examples,
                    supervised_tokens=text["supervised_tokens"] + ce,
                )
    manifest.update(
        format="hybrid-native-v2",
        family=family,
        max_features=max_features,
        min_pixels=min_pixels,
        media_root=str(pixels),
        model_vocab_size=vocab,
        sequence_length=max_length,
        native_corpus_sha256=sha256(corpus_root / "corpus-manifest.json"),
        native_pretrain_encoding="literal_question_answer_next_token_v1",
        native_pretrain_objective="all textual next-token targets including question and answer; media features and media markers masked",
        native_processor_sha256=processor_identity(family),
        shared_text_files_copied=0 if text_encoding is not None else None,
        raw_media_copied=False,
        formal_admission=False,
        main_budget_eligible=False,
        max_encoded_gib=max_gib,
    )
    update_manifest(manifest)
    manifest_text = json.dumps(manifest, indent=2)
    actual = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if maximum is not None and actual + len(manifest_text.encode()) > maximum:
        raise ValueError("native manifest exceeds the encoding disk budget")
    with reserve_write(
        output / "manifest.json", len(manifest_text.encode()), reserve_bytes=reserve
    ):
        (output / "manifest.json").write_text(manifest_text)
    return manifest


def audit_native_encoding(corpus_root, encoded, output):
    """Check every canonical media record, immutable pixel input and complete text target."""
    corpus_root, encoded, output = Path(corpus_root), Path(encoded), Path(output)
    manifest = json.loads((encoded / "manifest.json").read_text())
    report: dict[str, Any] = dict(
        kind="canonical_native_encoding_audit",
        formal_admission=False,
        model=manifest["family"],
        manifest_sha256=sha256(encoded / "manifest.json"),
        corpus_manifest_sha256=sha256(corpus_root / "corpus-manifest.json"),
        tokenizer_sha256=sha256(encoded / "tokenizer.json"),
        splits={},
        errors=[],
        model_forward_executed=False,
    )
    try:
        if (
            manifest["native_corpus_sha256"] != report["corpus_manifest_sha256"]
            or manifest["tokenizer"]["sha256"] != report["tokenizer_sha256"]
            or manifest["native_processor_sha256"] != processor_identity(manifest["family"])
        ):
            raise ValueError("native corpus/tokenizer/processor binding differs")
        tokenizer = Tokenizer.from_file(str(encoded / "tokenizer.json"))
        if reference := manifest.get("text_source"):
            shared = (encoded / reference["path"]).resolve()
            if sha256(shared / "manifest.json") != reference["manifest_sha256"]:
                raise ValueError("shared native text manifest changed")
            text_manifest = json.loads((shared / "manifest.json").read_text())
            if sha256(shared / "tokenizer.json") != report["tokenizer_sha256"] or any(
                node["text"] != text_manifest["stages"][stage][split]
                for stage, splits in manifest["stages"].items()
                for split, node in splits.items()
            ):
                raise ValueError("shared native text mapping or splits differ")
            report["shared_text_manifest_sha256"] = reference["manifest_sha256"]
        groups: dict[str, str] = {}
        text_partitions = None
        with contextlib.closing(open_corpus(corpus_root)) as db:
            for stage in ("pretrain", "sft"):
                for split in ("train", "val", "test"):
                    media = manifest["stages"][stage][split]["media"]
                    path, index_path = encoded / media["file"], encoded / media["index_file"]
                    if (
                        sha256(path) != media["sha256"]
                        or sha256(index_path) != media["index_sha256"]
                    ):
                        raise ValueError("native media/index checksum differs")
                    index = np.load(index_path)
                    counts: Counter[str] = Counter(
                        dict.fromkeys(
                            (
                                "source_media_records",
                                "records",
                                "input_tokens",
                                "ce_tokens",
                                "image_occurrences",
                                "video_examples",
                                "frames",
                                "image_features",
                                "complete_media_answer_exceeds_bucket",
                            ),
                            0,
                        )
                    )
                    domains: Counter[str] = Counter()
                    with path.open("rb") as stream:
                        for identity, payload, group in db.execute(
                            "SELECT id,payload,group_root FROM samples WHERE stage=? AND split=? ORDER BY id",
                            (stage, split),
                        ):
                            row = json.loads(payload)
                            if not row.get("media"):
                                continue
                            if "text_origin" in row and reference:
                                if text_partitions is None:
                                    text_partitions = _shared_text_partitions(
                                        shared, reference["manifest_sha256"]
                                    )
                                _check_text_origin(row, split, text_partitions)
                            if groups.setdefault(group, split) != split:
                                raise ValueError("canonical media group crosses native splits")
                            counts["source_media_records"] += 1
                            item = prepare_record(
                                row,
                                tokenizer,
                                manifest["family"],
                                root=manifest["media_root"],
                                max_features=manifest["max_features"],
                                min_pixels=manifest.get("min_pixels"),
                                model_vocab_size=manifest["model_vocab_size"],
                            )
                            if item.input_ids.shape[1] > manifest["sequence_length"]:
                                counts["complete_media_answer_exceeds_bucket"] += 1
                                continue
                            number = counts["records"]
                            if number >= len(index):
                                raise ValueError(
                                    "native encoding omitted an eligible canonical record"
                                )
                            offset = stream.tell()
                            raw = stream.readline()
                            saved = json.loads(raw)
                            if tuple(map(int, index[number])) != (
                                offset,
                                len(raw),
                                item.input_ids.shape[1],
                            ):
                                raise ValueError(
                                    "native index does not describe its complete record"
                                )
                            if saved["record"] != row or saved["split_group"] != group:
                                raise ValueError(
                                    f"native record/group differs from canonical {identity}"
                                )
                            if (
                                saved["expected_ids"] != item.input_ids[0].tolist()
                                or saved["expected_labels"] != item.labels[0].tolist()
                            ):
                                raise ValueError(
                                    "native ids/labels differ from complete canonical supervision"
                                )
                            targets = item.labels[0][item.labels[0].ne(-100)].tolist()
                            if stage == "pretrain" and "visual_answer" in row:
                                expected_text = (
                                    "\nUser: "
                                    + row["visual_question"]
                                    + "\nAssistant: "
                                    + row["visual_answer"]
                                )
                                if (
                                    tokenizer.decode(targets, skip_special_tokens=True)
                                    != expected_text
                                    or targets[-1] != 2
                                ):
                                    raise ValueError(
                                        "native text target no longer preserves the complete question/answer"
                                    )
                            count = int(item.labels[:, 1:].ne(-100).sum())
                            counts.update(
                                records=1,
                                input_tokens=item.input_ids.numel(),
                                ce_tokens=count,
                                image_occurrences=item.image_count,
                                video_examples=item.video_count,
                                frames=item.frame_count,
                                image_features=item.image_features,
                            )
                            domains[row["task"]] += count
                        if stream.read(1) or len(index) != counts["records"]:
                            raise ValueError("native encoding has extra records or index entries")
                    for key, field in (
                        ("records", "examples"),
                        ("ce_tokens", "supervised_tokens"),
                        ("image_occurrences", "image_occurrences"),
                        ("video_examples", "video_examples"),
                        ("frames", "frames"),
                        ("image_features", "image_features"),
                    ):
                        if counts[key] != media[field]:
                            raise ValueError("native split totals differ from actual inputs")
                    if dict(domains) != media["domain_ce"] or counts[
                        "complete_media_answer_exceeds_bucket"
                    ] != media["rejected"].get("complete_media_answer_exceeds_bucket", 0):
                        raise ValueError("native domains or complete-record rejections differ")
                    report["splits"][f"{stage}.{split}"] = dict(
                        counts=dict(counts), domain_ce=dict(domains)
                    )
        report.update(
            status="mechanical_checks_passed_pending_quality_admission",
            independent_groups=len(groups),
            raw_media_copied=False,
        )
    except (ValueError, OSError, KeyError, IndexError) as error:
        report["status"] = "failed"
        report["errors"].append(type(error).__name__ + ": " + str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["errors"]:
        raise ValueError("native encoding audit failed: " + "; ".join(report["errors"]))
    return report


class NativeDataset:
    def __init__(self, root, record, stage, length):
        self.root, self.length = Path(root), length
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.family, self.max_features = manifest["family"], manifest["max_features"]
        if manifest.get("native_processor_sha256") is not None and (
            manifest["native_processor_sha256"] != processor_identity(self.family)
        ):
            raise ValueError("native processor sources changed since encoding")
        self.media_root, self.vocab = manifest["media_root"], manifest["model_vocab_size"]
        self.min_pixels = manifest.get("min_pixels")
        if sha256(self.root / "tokenizer.json") != manifest["tokenizer"]["sha256"]:
            raise ValueError("native tokenizer checksum differs")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        text_root = self.root
        if reference := manifest.get("text_source"):
            text_root = (self.root / reference["path"]).resolve()
            if sha256(text_root / "manifest.json") != reference["manifest_sha256"]:
                raise ValueError("shared native text manifest changed")
            source = json.loads((text_root / "manifest.json").read_text())
            if record["text"] not in source["stages"][stage].values():
                raise ValueError("native text split differs from the shared manifest")
        self.text = DocumentDataset(text_root, record["text"], stage, length)
        media = record["media"]
        for path, digest in (
            (media["file"], media["sha256"]),
            (media["index_file"], media["index_sha256"]),
        ):
            if sha256(self.root / path) != digest:
                raise ValueError("native dataset hash mismatch")
        self.path = self.root / media["file"]
        index = np.load(self.root / media["index_file"], mmap_mode="r")
        self.index = index[index[:, 2] <= length]
        self.domains = list(self.text.domains)
        self.ce_counts = list(self.text.ce_counts)
        self.input_counts = list(self.text.input_counts)
        self.image_counts = [0] * len(self.text)
        self.video_counts = [0] * len(self.text)
        with self.path.open("rb") as source:
            for offset, size, positions in self.index:
                source.seek(int(offset))
                row = json.loads(source.read(int(size)))
                self.domains.append(row["record"]["task"])
                self.ce_counts.append(sum(value != -100 for value in row["expected_labels"][1:]))
                self.input_counts.append(int(positions))
                resources = row["record"].get("media", [])
                self.image_counts.append(sum(m.get("kind", "image") != "video" for m in resources))
                self.video_counts.append(sum(m.get("kind") == "video" for m in resources))

    def __len__(self):
        return len(self.text) + len(self.index)

    def __getitem__(self, index):
        if index < len(self.text):
            from minifrontier.multimodal import TrainingBatch

            x, y = self.text[index]
            return TrainingBatch(x[None], y[None])
        offset, size, _ = self.index[index - len(self.text)]
        with self.path.open("rb") as stream:
            stream.seek(int(offset))
            row = json.loads(stream.read(int(size)))
        prepared = prepare_record(
            row["record"],
            self.tokenizer,
            self.family,
            root=self.media_root,
            max_features=self.max_features,
            model_vocab_size=self.vocab,
            min_pixels=self.min_pixels,
        )
        if (
            prepared.input_ids[0].tolist() != row["expected_ids"]
            or prepared.labels[0].tolist() != row["expected_labels"]
        ):
            raise ValueError("native processor/template changed since immutable encoding")
        return prepared
