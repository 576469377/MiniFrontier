"""Replace dialogue spans in an encoded text corpus while retaining every other training document."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from minifrontier.chat_controls import update_manifest
from minifrontier.data import sha256

SOURCE = "HuggingFaceH4/ultrachat_200k"


def read(path):
    return json.loads(path.read_text())


def view(directory, record):
    for field, check in (
        ("file", "sha256"),
        ("labels_file", "labels_sha256"),
        ("index_file", "index_sha256"),
        ("metadata_file", "metadata_sha256"),
    ):
        if sha256(directory / record[field]) != record[check]:
            raise ValueError(f"source changed: {directory / record[field]}")
    ids = np.memmap(directory / record["file"], mode="r", dtype=np.int32)
    labels = np.memmap(directory / record["labels_file"], mode="r", dtype=np.int32)
    index = np.load(directory / record["index_file"], mmap_mode="r")
    return ids, labels, index


def merge_split(out, split, previous, old, restored, new):
    x, y, ix = view(previous, old)
    xx, yy, nx = view(restored, new)
    idx = []
    pos = 0
    counts: Counter[str] = Counter()
    checks = [[hashlib.sha256(), hashlib.sha256()] for _ in range(2)]
    with (
        (previous / old["metadata_file"]).open("rb") as old_meta,
        (restored / new["metadata_file"]).open("rb") as new_meta,
        (out / old["metadata_file"]).open("wb") as metadata,
        (out / old["file"]).open("wb") as tokens,
        (out / old["labels_file"]).open("wb") as labels,
    ):
        newer = iter(zip(nx, new_meta, strict=True))
        verify_spans = []
        for (start, length), line in zip(ix, old_meta, strict=True):
            record = json.loads(line)
            is_dialogue = record["source"] == SOURCE
            src_x, src_y = x, y
            if is_dialogue:
                (start, length), line = next(newer)
                altered = json.loads(line)
                if (record["sample_id"], record["item_id"]) != (
                    altered["sample_id"],
                    altered["item_id"],
                ):
                    raise ValueError(f"restored order or source differs in {split}")
                src_x, src_y = xx, yy
            a, b = int(start), int(start + length)
            tok = memoryview(src_x[a:b])
            target = memoryview(src_y[a:b])
            idx.append((pos, int(length)))
            pos += int(length)
            tokens.write(tok)
            labels.write(target)
            metadata.write(line)
            checks[is_dialogue][0].update(tok)
            checks[is_dialogue][1].update(target)
            verify_spans.append(is_dialogue)
            counts["restored_dialogues" if is_dialogue else "unchanged_documents"] += 1
            counts["loss"] += int(np.count_nonzero(src_y[a + 1 : b] != -100))
        if next(newer, None) is not None:
            raise ValueError("unused dialogue in restored partition")
    if counts["restored_dialogues"] != new["examples"] or len(idx) != old["examples"]:
        raise ValueError("partition count changed")
    np.save(out / old["index_file"], np.asarray(idx, dtype=np.int64).reshape(-1, 2))
    actual = [[hashlib.sha256(), hashlib.sha256()] for _ in range(2)]
    for k, key in enumerate(("file", "labels_file")):
        written = np.memmap(out / old[key], mode="r", dtype=np.int32)
        for (offset, n), changed in zip(idx, verify_spans, strict=True):
            actual[changed][k].update(memoryview(written[offset : offset + n]))
    if [[d.digest() for d in row] for row in actual] != [
        [d.digest() for d in row] for row in checks
    ]:
        raise ValueError("output lost an ordinary or restored token/label span")
    merged = dict(old)
    for key, hash_key in (
        ("file", "sha256"),
        ("labels_file", "labels_sha256"),
        ("metadata_file", "metadata_sha256"),
        ("index_file", "index_sha256"),
    ):
        merged[hash_key] = sha256(out / merged[key])
    merged.update(
        stored_positions=pos,
        supervised_tokens=counts["loss"],
        chat_template_counts=new["chat_template_counts"],
    )
    merged.pop("document_reuse", None)
    if "stored_dialogues_over_context" in new:
        merged["stored_dialogues_over_context"] = new["stored_dialogues_over_context"]
    return merged, dict(
        counts,
        unchanged_ids_sha256=actual[0][0].hexdigest(),
        unchanged_labels_sha256=actual[0][1].hexdigest(),
        restored_ids_sha256=actual[1][0].hexdigest(),
        restored_labels_sha256=actual[1][1].hexdigest(),
    )


def merge(previous, restored, out):
    old, new = (read(root / "manifest.json") for root in (previous, restored))
    if old["format"] != "document-ragged-v2" or old["tokenizer"] != new["tokenizer"]:
        raise ValueError("ragged corpus or tokenizer differs")
    if old["sequence_length"] != new["sequence_length"]:
        raise ValueError("maximum context differs")
    out.mkdir(exist_ok=False, parents=True)
    if sha256(previous / "tokenizer.json") != old["tokenizer"]["sha256"]:
        raise ValueError("predecessor tokenizer changed")
    shutil.copyfile(previous / "tokenizer.json", out / "tokenizer.json")
    manifest = {
        k: v
        for k, v in old.items()
        if k not in ("reused_encoding", "storage_bound", "text_metadata")
    }
    receipt = {
        "predecessor": dict(path=str(previous), sha256=sha256(previous / "manifest.json")),
        "updated_dialogue": dict(path=str(restored), sha256=sha256(restored / "manifest.json")),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        data, evidence = merge_split(
            out,
            split,
            previous,
            old["stages"]["pretrain"][split],
            restored,
            new["stages"]["pretrain"][split],
        )
        receipt["splits"][split] = evidence
        manifest["stages"]["pretrain"][split] = data
        print(split, evidence, flush=True)
    for spec in old["stages"]["sft"].values():
        for k in ("file", "labels_file", "index_file", "metadata_file"):
            shutil.copyfile(previous / spec[k], out / spec[k])
    origin = out / "corpus-manifest.json"
    origin.write_text(json.dumps(receipt, indent=2))
    manifest["corpus_sha256"] = sha256(origin)
    update_manifest(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    merge(opts.previous, opts.restored, opts.output)
