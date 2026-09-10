"""Verify encoded text against its immutable canonical partition, without model training."""

import argparse
import contextlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.corpus import STRATEGY_SPECIAL_TOKENS
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, digest, write_json
from minifrontier.data.minifrontier1_encoding import FORMAT, INDEX
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.models.minifrontier1 import MiniFrontier1Config

DOMAINS = dict(
    zh_edu="zh_general",
    en_edu="en_general",
    code="code",
    verified_math_science="math",
    dialogue="structured",
)


def _checked(root, name, expected, size=None):
    path = (root / name).resolve()
    if (
        not path.is_relative_to(root)
        or sha256(path) != expected
        or (size is not None and path.stat().st_size != size)
    ):
        raise ValueError("encoded file path/hash/size differs")
    return path


def _source_records(root, manifest, split):
    row = manifest["stages"]["pretrain"][split]
    paths = {
        k: _checked(root, row[k], row[h])
        for k, h in (
            ("file", "sha256"),
            ("labels_file", "labels_sha256"),
            ("index_file", "index_sha256"),
            ("metadata_file", "metadata_sha256"),
        )
    }
    indexes = np.load(paths["index_file"], allow_pickle=False)
    if indexes.dtype != np.dtype("int64") or indexes.shape != (row["examples"], 2):
        raise ValueError("source document index shape/type differs")
    if (
        paths["file"].stat().st_size != row["stored_positions"] * 4
        or paths["labels_file"].stat().st_size != row["stored_positions"] * 4
    ):
        raise ValueError("source token/label storage size differs")
    if not row["examples"]:
        if paths["metadata_file"].stat().st_size:
            raise ValueError("empty source split has nonempty metadata")
        return
    ids = np.memmap(paths["file"], mode="r", dtype=np.int32)
    labels = np.memmap(paths["labels_file"], mode="r", dtype=np.int32)
    end = 0
    with paths["metadata_file"].open() as metadata:
        for (offset, length), line in zip(indexes, metadata, strict=True):
            offset, length = int(offset), int(length)
            if offset != end or length < 2 or offset + length > len(ids):
                raise ValueError("source documents have gaps, overlap or invalid lengths")
            x, y = ids[offset : offset + length], labels[offset : offset + length]
            mask = y != -100
            if not np.array_equal(y[mask], x[mask]):
                raise ValueError("source labels differ from their token IDs")
            record = json.loads(line)
            yield record["sample_id"], record["task"], record, None, x, mask
            end += length
    if end != len(ids):
        raise ValueError("source token storage has unindexed positions")


def _compact_records(root, manifest, split, *, images=False):
    if manifest["token_dtype"] not in ("<u2", "<u4"):
        raise ValueError("unsupported compact token type")
    for part in manifest["splits"][split]["parts"]:
        paths = {
            k: _checked(root, v["name"], v["sha256"], v["bytes"]) for k, v in part["files"].items()
        }
        indexes = np.memmap(paths["index.bin"], mode="r", dtype=INDEX)
        ids = np.memmap(paths["tokens.bin"], mode="r", dtype=manifest["token_dtype"])
        masks = np.memmap(paths["mask.bin"], mode="r", dtype=np.uint8)
        counts = part["counts"]
        if len(indexes) != counts["records"] or counts.get("text_documents", 0) != (
            0 if images else len(indexes)
        ):
            raise ValueError("compact component has an unexpected record type")
        end = mask_end = meta_end = part_ce = 0
        with paths["metadata.jsonl"].open("rb") as metadata:
            for index in indexes:
                offset, length = int(index["offset"]), int(index["length"])
                if (
                    offset != end
                    or length < 2
                    or offset + length > len(ids)
                    or int(index["mask_offset"]) != mask_end
                    or int(index["metadata_offset"]) != meta_end
                ):
                    raise ValueError("compact document offsets have gaps or overlap")
                raw = metadata.read(int(index["metadata_bytes"]))
                if len(raw) != int(index["metadata_bytes"]):
                    raise ValueError("compact document metadata was truncated")
                row = json.loads(raw)
                if not images and (row["media"] or row["resources"]):
                    raise ValueError("text component contains media")
                mask_bytes = (length + 7) // 8
                if mask_end + mask_bytes > len(masks):
                    raise ValueError("compact mask was truncated")
                mask = np.unpackbits(
                    masks[mask_end : mask_end + mask_bytes], bitorder="little", count=length
                ).astype(bool)
                yield (
                    row["sample_id"],
                    manifest["domains"][int(index["domain"])],
                    row["origin"],
                    row["split_group"],
                    ids[offset : offset + length],
                    mask,
                    row,
                )
                part_ce += int(mask[1:].sum())
                end += length
                mask_end += mask_bytes
                meta_end += len(raw)
        if (
            end != len(ids)
            or mask_end != len(masks)
            or meta_end != paths["metadata.jsonl"].stat().st_size
            or end != counts["input_tokens"]
            or part_ce != counts["ce_tokens"]
        ):
            raise ValueError("compact part size or CE counters differ")


def _mf1_records(root, manifest, split):
    for record in _compact_records(root, manifest, split):
        yield record[:6]


def audit_image_encoding(corpus, encoded, output, config):
    """Check every standard single-image QA's source, protocol, labels and geometry.

    Raw files are hashed once per distinct image; no vision features are cached or
    inferred from the labels. Source quality and model capability remain separate.
    """
    corpus, root, output = Path(corpus).resolve(), Path(encoded).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("encoding audit is immutable; choose a new report")
    values = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
    model = MiniFrontier1Config(**values)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("kind") != "canonical_image_component"
        or manifest["config_sha256"] != digest(asdict(model))
        or manifest["image_transform"]["document_tiles"]
        or manifest["source_corpus_manifest_sha256"] != sha256(corpus / "corpus-manifest.json")
        or manifest["source_manifest_sha256"] != manifest["source_corpus_manifest_sha256"]
    ):
        raise ValueError("image audit needs a bound standard-image component and model config")
    canonical = json.loads((corpus / "corpus-manifest.json").read_text())
    tokenizer = Tokenizer.from_file(
        str(_checked(root, "tokenizer.json", manifest["tokenizer_sha256"]))
    )
    if tokenizer.get_vocab_size() > model.vocab_size or any(
        tokenizer.token_to_id(token) != i for i, token in enumerate(SPECIAL_TOKENS)
    ):
        raise ValueError("image component vocabulary/control mapping differs")
    report = dict(
        kind="canonical_image_encoding_audit",
        status="checking",
        formal_admission=False,
        started_unix=time.time(),
        encoded_manifest_sha256=sha256(root / "manifest.json"),
        corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        tokenizer_sha256=manifest["tokenizer_sha256"],
        config_sha256=manifest["config_sha256"],
        processor_sha256=sha256(__file__),
        splits={},
        scope="all records and raw image hashes; QA identity, complete answer CE, protocol, span geometry and counts; not source quality or model capability",
    )
    checked_media = set()
    try:
        with contextlib.closing(open_corpus(corpus)) as db:
            media_root = corpus_storage_root(db)
            if sha256(media_root / "corpus.sqlite") != canonical["database_sha256"]:
                raise ValueError("canonical media database checksum differs")
            remaining = dict(db.execute("SELECT id,split FROM samples"))
            for split in ("train", "val", "test"):
                counts: Counter[str] = Counter()
                domains: Counter[str] = Counter()
                buckets: dict[str, dict[str, Counter[str]]] = {}
                for identity, domain, origin, group, ids, mask, meta in _compact_records(
                    root, manifest, split, images=True
                ):
                    if remaining.pop(identity, None) != split:
                        raise ValueError(
                            "encoded image duplicated, absent or in the wrong partition"
                        )
                    payload, expected_group = db.execute(
                        "SELECT payload,group_root FROM samples WHERE id=?", (identity,)
                    ).fetchone()
                    row = json.loads(payload)
                    if (
                        row["stage"] != "pretrain"
                        or domain != row["task"]
                        or group != expected_group
                        or origin.get("text_origin") != row.get("text_origin")
                        or any(
                            origin[k] != row[k]
                            for k in ("source", "revision", "item_id", "license", "content_hash")
                        )
                    ):
                        raise ValueError("image source, domain or connected group differs")
                    if (
                        len(row["media"]) != 1
                        or len(meta["resources"]) != 1
                        or len(meta["media"]) != 1
                    ):
                        raise ValueError("standard image QA needs one original image and one span")
                    original, resource, span = (
                        row["media"][0],
                        meta["resources"][0],
                        meta["media"][0],
                    )
                    if any(
                        resource[k] != original[k]
                        for k in ("sha256", "rgb_sha256", "width", "height")
                    ) or (
                        resource["uri"] != original["path"]
                        or resource["media_id"] != original["rgb_sha256"]
                        or resource["representation"] != "standard"
                        or resource["max_features"] != manifest["image_transform"]["max_features"]
                    ):
                        raise ValueError(
                            "encoded image resource differs from its canonical original"
                        )
                    path = (media_root / resource["uri"]).resolve()
                    key = (str(path), resource["sha256"])
                    if key not in checked_media:
                        _checked(media_root, resource["uri"], resource["sha256"])
                        checked_media.add(key)
                    features = span["feature_count"]
                    grid = np.asarray(span["grid_thw"])
                    if (
                        grid.shape != (1, 3)
                        or grid[0, 0] != 1
                        or np.any(grid < 1)
                        or np.any(grid[0, 1:] % 2)
                        or int(grid.prod()) // 4 != features
                        or not 1
                        <= features
                        <= min(resource["max_features"], model.protected_media_tokens)
                        or span["source_size"] != [original["width"], original["height"]]
                        or span["resized_size"]
                        != [
                            int(grid[0, 2]) * model.vision_config.patch_size,
                            int(grid[0, 1]) * model.vision_config.patch_size,
                        ]
                        or span["start"] != 3
                        or span["batch_index"] != 0
                        or span["resource_kind"] != "image"
                        or span["media_id"] != resource["media_id"]
                        or span["source_sha256"] != original["sha256"]
                    ):
                        raise ValueError("image span geometry or source identity differs")
                    texts = [row["visual_question"], row["visual_answer"]]
                    for i, text in enumerate(texts):
                        for control in SPECIAL_TOKENS:
                            text = text.replace(control, control[0] + "\u2060" + control[1:])
                        texts[i] = text
                    question, answer = [
                        tokenizer.encode(s, add_special_tokens=False).ids for s in texts
                    ]
                    expected = [1, 4, 9, *([7] * features), 10, *question, 2, 5, 17, *answer, 2]
                    expected_mask = np.zeros(len(expected), dtype=bool)
                    expected_mask[-len(answer) - 2 :] = True
                    if not np.array_equal(ids, expected) or not np.array_equal(mask, expected_mask):
                        raise ValueError("image QA protocol or complete answer labels differ")
                    if np.any(ids >= model.vocab_size) or len(ids) > model.max_position_embeddings:
                        raise ValueError("image QA exceeds model vocabulary/context")
                    delta = dict(
                        records=1,
                        input_tokens=len(ids),
                        ce_tokens=len(answer) + 2,
                        media_exposures=1,
                        vision_tokens=features,
                    )
                    counts.update(delta)
                    domains[domain] += delta["ce_tokens"]
                    bucket = str(
                        next(
                            (n for n in (512, 1024, 2048, 4096, 8192) if len(ids) <= n), "over_8192"
                        )
                    )
                    buckets.setdefault(domain, {}).setdefault(bucket, Counter()).update(delta)
                if (
                    dict(counts) != manifest["splits"][split]["counts"]
                    or dict(domains) != manifest["splits"][split]["domain_ce"]
                    or buckets != manifest["complete_record_length_buckets"][split]
                ):
                    raise ValueError("image split or complete-record length counts differ")
                report["splits"][split] = dict(
                    counts=dict(counts),
                    domain_ce=dict(domains),
                    complete_record_length_buckets=buckets,
                )
                print(
                    json.dumps(
                        dict(split=split, records=counts["records"], ce_tokens=counts["ce_tokens"])
                    ),
                    flush=True,
                )
            if remaining:
                raise ValueError("canonical image QAs are missing from encoding")
        report.update(
            status="mechanical_checks_passed_pending_quality_admission",
            raw_media_files=len(checked_media),
        )
    except BaseException as error:
        report.update(
            status="failed",
            error=type(error).__name__ + ": " + str(error),
            updated_unix=time.time(),
        )
        write_json(output, report)
        raise
    report["updated_unix"] = time.time()
    write_json(output, report)
    return report


def audit_text_encoding(corpus, encoded, output):
    corpus, root, output = Path(corpus).resolve(), Path(encoded).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("encoding audit is immutable; choose a new report")
    manifest = json.loads((root / "manifest.json").read_text())
    compact = manifest.get("format") == FORMAT
    if compact:
        if manifest.get("kind") != "canonical_text_component":
            raise ValueError("this audit is for a canonical text component")
        corpus_hash = manifest["source_corpus_manifest_sha256"]
        if manifest["source_manifest_sha256"] != corpus_hash:
            raise ValueError("compact source-manifest bindings disagree")
        tokenizer_hash = manifest["tokenizer_sha256"]
        controls = SPECIAL_TOKENS
    else:
        if manifest.get("format") != "document-ragged-v2" or any(
            v["examples"] for v in manifest["stages"]["sft"].values()
        ):
            raise ValueError("this audit requires pure pretraining text")
        corpus_hash = manifest["corpus_sha256"]
        tokenizer_hash = manifest["tokenizer"]["sha256"]
        controls = STRATEGY_SPECIAL_TOKENS
    if corpus_hash != sha256(corpus / "corpus-manifest.json"):
        raise ValueError("encoding refers to a different canonical partition")
    tokenizer = Tokenizer.from_file(str(_checked(root, "tokenizer.json", tokenizer_hash)))
    if any(tokenizer.token_to_id(s) != i for i, s in enumerate(controls)):
        raise ValueError("encoded control mapping differs")
    if not compact and tokenizer.get_vocab_size() != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("declared source vocabulary size differs")
    report = dict(
        schema_version=1,
        kind="canonical_text_encoding_audit",
        status="checking",
        started_unix=time.time(),
        formal_admission=False,
        encoded_manifest_sha256=sha256(root / "manifest.json"),
        corpus_manifest_sha256=corpus_hash,
        tokenizer_sha256=tokenizer_hash,
        processor_sha256=sha256(__file__),
        splits={},
        errors={},
        error_samples=[],
        scope="all documents, file hashes, partition membership, labels, boundaries and decoded text; not source quality review",
    )
    errors: Counter[str] = Counter()
    try:
        with contextlib.closing(open_corpus(corpus)) as db:
            remaining = dict(db.execute("SELECT id,split FROM samples"))
            for split in ("train", "val", "test"):
                counts: Counter[str] = Counter()
                domains: Counter[str] = Counter()
                lengths: Counter[str] = Counter()
                records = (_mf1_records if compact else _source_records)(root, manifest, split)
                for identity, domain, origin, group, ids, mask in records:
                    if remaining.pop(identity, None) != split:
                        raise ValueError(
                            "encoded record duplicated, absent or in the wrong partition"
                        )
                    payload, expected_group = db.execute(
                        "SELECT payload,group_root FROM samples WHERE id=?", (identity,)
                    ).fetchone()
                    reference = json.loads(payload)
                    if (
                        reference["stage"] != "pretrain"
                        or reference.get("media")
                        or reference.get("turns")
                    ):
                        raise ValueError("reference is not a pure pretraining document")
                    if domain != (DOMAINS[reference["task"]] if compact else reference["task"]) or (
                        compact and group != expected_group
                    ):
                        raise ValueError("encoded domain or connected group differs")
                    if any(
                        origin[k] != reference[k]
                        for k in ("source", "revision", "item_id", "license", "content_hash")
                    ):
                        raise ValueError("encoded source provenance differs")
                    if (
                        int(ids.min()) < 0
                        or int(ids.max()) >= tokenizer.get_vocab_size()
                        or ids[0] != 1
                        or ids[-1] != 2
                    ):
                        raise ValueError("token bounds or complete document boundary differs")
                    if mask[0] or not mask[1:].all():
                        raise ValueError(
                            "continuation loss mask does not cover exactly the document CE positions"
                        )
                    failures = []
                    if np.any(ids[1:-1] < len(controls)):
                        failures.append("literal_control_encoded_as_protocol")
                    text = reference["text"]
                    if compact:
                        for control in controls:
                            text = text.replace(control, control[0] + "\u2060" + control[1:])
                    if tokenizer.decode(ids[1:-1].tolist(), skip_special_tokens=False) != text:
                        failures.append("decoded_text_differs")
                    errors.update(failures)
                    if failures and len(report["error_samples"]) < 20:
                        report["error_samples"].append(
                            dict(sample_id=identity, split=split, errors=failures)
                        )
                    counts.update(records=1, input_tokens=len(ids), ce_tokens=len(ids) - 1)
                    domains[domain] += len(ids) - 1
                    lengths[
                        str(
                            next(
                                (n for n in (512, 1024, 2048, 4096, 8192) if len(ids) <= n),
                                "over_8192",
                            )
                        )
                    ] += 1
                declared = (
                    manifest["splits"][split]["counts"]
                    if compact
                    else manifest["stages"]["pretrain"][split]
                )
                if (
                    counts["records"] != declared.get("records" if compact else "examples", 0)
                    or counts["input_tokens"]
                    != declared.get("input_tokens" if compact else "stored_positions", 0)
                    or counts["ce_tokens"]
                    != declared.get("ce_tokens" if compact else "supervised_tokens", 0)
                    or (compact and dict(domains) != manifest["splits"][split]["domain_ce"])
                ):
                    raise ValueError("encoded split counters differ")
                report["splits"][split] = dict(
                    counts=counts, domain_ce=domains, document_length_buckets=lengths
                )
                report.update(updated_unix=time.time(), errors=dict(errors))
                print(
                    json.dumps(
                        dict(
                            split=split,
                            records=counts["records"],
                            ce=counts["ce_tokens"],
                            errors=dict(errors),
                        )
                    ),
                    flush=True,
                )
            if remaining:
                raise ValueError("canonical documents are missing from encoding")
        report["status"] = (
            "failed" if errors else "mechanical_checks_passed_pending_quality_admission"
        )
    except BaseException as error:
        report.update(
            status="failed",
            error=type(error).__name__ + ": " + str(error),
            errors=dict(errors),
            updated_unix=time.time(),
        )
        write_json(output, report)
        raise
    report["updated_unix"] = time.time()
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--encoded", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["text", "image"], default="text")
    parser.add_argument("--config", help="model config for image audits")
    args = vars(parser.parse_args())
    kind, config = args.pop("kind"), args.pop("config")
    if kind == "image" and not config:
        parser.error("image audits require --config")
    result = (
        audit_image_encoding(**args, config=config)
        if kind == "image"
        else audit_text_encoding(**args)
    )
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
