"""Verify encoded text against its immutable canonical partition, without model training."""

import argparse
import contextlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.corpus import STRATEGY_SPECIAL_TOKENS
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, write_json
from minifrontier.data.minifrontier1_encoding import FORMAT, INDEX
from minifrontier.data.partitions import open_corpus

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


def _mf1_records(root, manifest, split):
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
        if len(indexes) != counts["records"] or counts.get("text_documents") != len(indexes):
            raise ValueError("compact component includes non-document records")
        end = mask_end = meta_end = 0
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
                if row["media"] or row["resources"]:
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
                )
                end += length
                mask_end += mask_bytes
                meta_end += len(raw)
        if (
            end != len(ids)
            or mask_end != len(masks)
            or meta_end != paths["metadata.jsonl"].stat().st_size
            or end != counts["input_tokens"]
            or end - len(indexes) != counts["ce_tokens"]
        ):
            raise ValueError("compact part size or CE counters differ")


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
    result = audit_text_encoding(**vars(parser.parse_args()))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
