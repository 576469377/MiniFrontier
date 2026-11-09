"""Restore UltraChat turn boundaries from a pinned Parquet shard and preserve the retained split."""

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from minifrontier.data import chat_tokens, fingerprint, normalized, sha256
from minifrontier.data.corpus import DocumentDataset, encode_corpus
from minifrontier.data.partitions import open_corpus

SOURCE = "HuggingFaceH4/ultrachat_200k"
REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SHARD = "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"
SHARD_SHA = "6fe7d2a5e95cf660f972ecaf304aa5632d7f53384e4b0e0d0f44d9c96733c03e"
SPLITS = ("train", "val", "test")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get_upstream(path):
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    if path.name != SHARD or sha256(path) != SHARD_SHA:
        raise ValueError("UltraChat Parquet does not match the pinned Git LFS object")
    source = pq.ParquetFile(path)

    @lru_cache(maxsize=3)
    def group(index):
        return source.read_row_group(index, columns=["messages", "prompt_id"]).to_pylist()

    def row(index, offset, text):
        item = group(index)[offset]
        return item["prompt_id"], item["messages"]

    return row


def restore(args):
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "corpus"
    target.mkdir()
    report = dict(
        source=SOURCE,
        revision=REVISION,
        shard=SHARD,
        shard_sha256=SHARD_SHA,
        input_corpus_sha256=sha256(args.corpus / "corpus-manifest.json"),
        compiled_predecessor_sha256=sha256(args.previous / "manifest.json"),
    )
    expected = json.loads((args.previous / "manifest.json").read_text())
    if expected["corpus_sha256"] != report["input_corpus_sha256"]:
        raise ValueError("historical token files and SQLite corpus use different source manifests")
    old = open_corpus(args.corpus)
    source = get_upstream(args.parquet)
    kept = {}
    for split in SPLITS:
        file = args.previous / expected["stages"]["pretrain"][split]["metadata_file"]
        for row in map(json.loads, file.open()):
            if row["source"] == SOURCE:
                identity = row["sample_id"]
                if identity in kept:
                    raise ValueError("duplicate historical dialogue")
                kept[identity] = (split, row["item_id"], row["group_id"])
    new = sqlite3.connect(target / "corpus.sqlite")
    new.execute(
        old.execute(
            "SELECT sql FROM sqlite_master WHERE name='samples' AND type='table'"
        ).fetchone()[0]
    )
    counts: Counter[str] = Counter()
    groups: dict[str, str] = {}
    seen_upstream = set()
    observed: Counter[str] = Counter()
    query = "SELECT id, group_root, split, payload, first_question, text, simhash FROM samples WHERE source=? AND stage='pretrain'"
    for identity, root, split, payload, question, legacy_text, simhash in old.execute(
        query, (SOURCE,)
    ):
        rec = json.loads(payload)
        if kept.pop(identity) != (split, rec["item_id"], rec["group_id"]):
            raise ValueError(f"stored split or source changed for {identity}")
        loc = re.fullmatch(r"data/(.+):rg(\d+):row(\d+)", rec["item_id"])
        if not loc or loc[1] != SHARD or rec["revision"] != REVISION:
            raise ValueError(f"unexpected upstream identity for {identity}")
        key = int(loc[2]), int(loc[3])
        prompt, turns = source(*key, rec["text"])
        if key in seen_upstream or groups.setdefault(root, split) != split:
            raise ValueError(f"duplicate item or split group for {identity}")
        seen_upstream.add(key)
        flat = "\n\n".join(t["role"].capitalize() + ": " + t["content"] for t in turns)
        if not (
            prompt == rec["group_id"]
            and digest(flat) == rec["raw_text_sha256"]
            and normalized(flat) == rec["text"]
            and digest(rec["text"]) == rec["normalized_text_sha256"]
            and fingerprint(rec["text"]) == identity == rec["sample_id"] == rec["content_hash"]
        ):
            raise ValueError(f"upstream source bytes/group/identity differ for {identity}")
        if (
            not turns
            or len(turns) % 2
            or any(t["role"] != ("user", "assistant")[i % 2] for i, t in enumerate(turns))
        ):
            raise ValueError(f"unexpected role sequence at {rec['item_id']}")
        for t in turns:
            observed[t["role"]] += 1
        nul = [
            [i, j] for i, t in enumerate(turns) for j, c in enumerate(t["content"]) if c == chr(0)
        ]
        if nul:
            observed["raw_nul_records"] += 1
            observed["raw_nul_characters"] += len(nul)
            rec["raw_nul_positions"] = nul
        clean = [dict(t, content=t["content"].replace(chr(0), "")) for t in turns]
        rec.update(text_format="chat_turns", turns=clean)
        new.execute(
            "INSERT INTO samples(id, stage, source, task, first_question, text, payload, "
            "simhash, group_root, split) VALUES (?, 'pretrain', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                identity,
                SOURCE,
                rec["task"],
                question,
                legacy_text,
                json.dumps(rec, ensure_ascii=False, separators=(",", ":")),
                simhash,
                root,
                split,
            ),
        )
        counts[split] += 1
    old.close()
    if kept or sum(counts.values()) != 32953:
        raise ValueError(f"retained cohort incomplete: {len(kept)} missing")
    new.commit()
    new.execute("CREATE INDEX partition ON samples(stage,split,id)")
    new.close()
    report.update(
        splits=counts,
        turns=observed,
        source_rows_restored=len(seen_upstream),
        frozen_group_count=len(groups),
        database_sha256=sha256(target / "corpus.sqlite"),
        record_identity="Original sample_id, group, item_id and split are retained; original raw hashes verified; only NUL characters removed from model text, with offsets recorded.",
    )
    (target / "corpus-manifest.json").write_text(json.dumps(report, indent=2))
    print("restored", dict(counts), flush=True)
    toroot = args.output / "encoded"
    tokenizer_file = args.previous / "tokenizer.json"
    encoded = encode_corpus(target, tokenizer_file, toroot, max_length=args.max_length)
    report["encoded_manifest_sha256"] = sha256(toroot / "manifest.json")
    report["tokenizer_sha256"] = sha256(tokenizer_file)
    report["admission"] = audit(toroot, encoded, tokenizer_file, args.lengths)
    (args.output / "rebuild-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["admission"]), flush=True)


def audit(root, manifest, tokenizer_path, lengths):
    tk = Tokenizer.from_file(str(tokenizer_path))
    report = {}
    for split in SPLITS:
        part = manifest["stages"]["pretrain"][split]
        x = np.memmap(root / part["file"], dtype=np.int32, mode="r")
        y = np.memmap(root / part["labels_file"], dtype=np.int32, mode="r")
        ix = np.load(root / part["index_file"], mmap_mode="r")
        counts: Counter[str] = Counter()
        admitted: dict[str, Counter[str]] = {str(length): Counter() for length in lengths}
        for (a, n), line in zip(ix, (root / part["metadata_file"]).open(), strict=True):
            turns = json.loads(line)["turns"]
            ids, labels = chat_tokens(turns, tk)
            if not np.array_equal(ids, x[a : a + n]) or not np.array_equal(labels, y[a : a + n]):
                raise ValueError("stored token/label sequence differs from serving tokenizer")
            # Check each role, the EOS loss and the exact user-to-assistant generation prefix.
            end = 1
            for i, t in enumerate(turns):
                single, _ = chat_tokens([t], tk)
                role_ids = single[1:]
                if ids[end : end + len(role_ids)] != role_ids:
                    raise ValueError("turn boundaries differ")
                if t["role"] == "assistant":
                    prompt, _ = chat_tokens([turns[i - 1]], tk, generation_prompt=True)
                    if not (
                        ids[end] == prompt[-1]
                        and labels[end] == -100
                        and labels[end + 1 : end + len(role_ids)] == role_ids[1:]
                        and labels[end + len(role_ids) - 1] == 2
                    ):
                        raise ValueError("assistant prompt, target or EOS differs")
                    counts["assistant_turns"] += 1
                elif any(t != -100 for t in labels[end : end + len(role_ids)]):
                    raise ValueError("loss leaks into user input")
                end += len(role_ids)
            if end != len(ids) or labels[0] != -100:
                raise ValueError("unexpected position outside turn boundaries")
            if counts["examples"] % 256 == 0:
                for turn in range(1, len(turns), 2):
                    prefix, _ = chat_tokens(turns[:turn], tk, generation_prompt=True)
                    if ids[: len(prefix)] != prefix:
                        raise ValueError(
                            "multi-turn serving prompt differs before an assistant answer"
                        )
                    counts["reencoded_full_history_prefixes"] += 1
            counts["examples"] += 1
            supervised = sum(t != -100 for t in labels)
            counts["supervised_tokens"] += supervised
            for length in lengths:
                admitted[str(length)]["admitted" if n <= length else "too_long"] += 1
                if n > length:
                    admitted[str(length)]["withheld_supervised_tokens"] += supervised
        if counts["supervised_tokens"] != part["supervised_tokens"]:
            raise ValueError("loss token count differs from manifest")
        for length in lengths:
            runtime = DocumentDataset(root, part, "pretrain", length)
            c = admitted[str(length)]
            if (
                len(runtime) != c["admitted"]
                or runtime.overlength_dialogues["records"] != c["too_long"]
            ):
                raise ValueError("runtime admitted a partial or excluded conversation")
        report[split] = dict(counts=counts, by_context=admitted)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[384, 512, 1024, 2048, 4096, 8192]
    )
    restore(parser.parse_args())
