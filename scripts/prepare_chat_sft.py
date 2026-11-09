"""Build a bilingual direct-answer corpus and verify its encoded conversation boundaries."""

import argparse
import hashlib
import heapq
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from minifrontier.data import StageDataset, chat_tokens, sha256
from minifrontier.data.corpus import CorpusBuilder, canonical_question, encode_corpus


def direct_turns(record):
    turns = record.get("conversations")
    if not isinstance(turns, list) or not turns or len(turns) % 2:
        return None
    result = []
    for i, turn in enumerate(turns):
        role = ("user", "assistant")[i % 2]
        if turn.get("role") != role or any(
            turn.get(key) for key in ("tools", "tool_calls", "reasoning", "reasoning_content")
        ):
            return None
        text = turn.get("content")
        if not isinstance(text, str):
            return None
        text = text.strip()
        if role == "assistant":
            text = re.sub(r"^<think>\s*</think>\s*", "", text)
        if not text or any(
            marker in text for marker in ("<|", "<think>", "</think>", "\ufffd", "\x00")
        ):
            return None
        result.append({"role": role, "content": text})
    return result


def language(text):
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall("[A-Za-z]", text))
    if han >= 2 and han * 4 >= latin:
        return "zh"
    if latin >= 3 and latin >= 0.75 * sum(c.isalpha() for c in text):
        return "en"
    return None


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    blocked = set()
    if args.exclude_prompts:
        blocked = {canonical_question(s) for s in json.loads(args.exclude_prompts.read_text())}
    rng = random.Random(args.seed)
    pools: dict[tuple[str, str], list[tuple[float, int, list[dict[str, str]]]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    # Reservoirs prevent the first rows of a source ordered by task from dominating.
    with args.source.open() as f:
        for line_no, line in enumerate(f, 1):
            counts["source_rows"] += 1
            turns = direct_turns(json.loads(line))
            if turns is None:
                counts["outside_direct_chat"] += 1
                continue
            users = [t["content"] for t in turns[::2]]
            if any(canonical_question(q) in blocked for q in users):
                counts["reserved_prompt"] += 1
                continue
            lang = language(" ".join(users))
            if not lang or sum(len(t["content"]) for t in turns) > args.length * 5:
                counts["outside_language_or_length"] += 1
                continue
            kind = (
                "multi"
                if len(turns) > 2
                else "short"
                if len(turns[-1]["content"]) <= 80
                else "other"
            )
            pool = pools[lang, kind]
            entry = (rng.random(), line_no, turns)
            if len(pool) < args.per_bucket:
                heapq.heappush(pool, entry)
            elif entry[0] > pool[0][0]:
                heapq.heapreplace(pool, entry)

    builder = CorpusBuilder(root / "corpus", seed=args.seed, val_buckets=100, test_buckets=100)
    used: set[str] = set()
    for (lang, kind), pool in sorted(pools.items()):
        for _, line_no, turns in sorted(pool, reverse=True):
            ids, _ = chat_tokens(turns, tokenizer, mode="direct", effort="low")
            if len(ids) > args.length:
                counts["over_context"] += 1
                continue
            prompts = {canonical_question(t["content"]) for t in turns[::2]}
            if prompts & used:
                counts["duplicate_user_prompt"] += 1
                continue
            group = hashlib.sha256(canonical_question(turns[0]["content"]).encode()).hexdigest()
            if builder.add(
                dict(
                    source="jingyaogong/minimind_dataset",
                    revision=args.revision,
                    item_id=f"{args.source.name}:{line_no}",
                    group_id=group,
                    license="source dataset terms",
                    lang=lang,
                    task="dialogue",
                    stage="sft",
                    mode="direct",
                    effort="low",
                    turns=turns,
                )
            ):
                used.update(prompts)
                counts[f"admitted_{lang}_{kind}"] += 1
    builder.finalize()
    encoded = root / "encoded"
    manifest = encode_corpus(root / "corpus", args.tokenizer, encoded, max_length=args.length)
    checks = audit(encoded, manifest, tokenizer, args.length, blocked)
    report = dict(
        source_sha256=sha256(args.source),
        revision=args.revision,
        seed=args.seed,
        tokenizer_sha256=sha256(args.tokenizer),
        source_counts=dict(counts),
        builder_counts=dict(builder.counts),
        splits=checks,
        excluded_prompts=len(blocked),
        template=manifest["chat_template"],
        manifest_sha256=sha256(encoded / "manifest.json"),
    )
    (root / "chat-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(checks), flush=True)


def audit(encoded, manifest, tokenizer, length_limit, blocked):
    checks = {}
    split_prompts: set[str] = set()
    for split in ("train", "val", "test"):
        record = manifest["stages"]["sft"][split]
        dataset = StageDataset(encoded, "sft", split, length_limit).documents
        rows: Counter[str] = Counter()
        prompts = set()
        with (encoded / record["metadata_file"]).open() as f:
            for index, line in enumerate(f):
                row = json.loads(line)
                turns = row["turns"]
                ids, labels = chat_tokens(turns, tokenizer, mode="direct", effort="low")
                start, length = map(int, dataset.documents[index])
                if not np.array_equal(
                    dataset.tokens[start : start + length], ids
                ) or not np.array_equal(dataset.labels[start : start + length], labels):
                    raise ValueError(f"encoded conversation differs at {split}/{index}")
                expected = [-100] * len(ids)
                for i in range(1, len(turns), 2):
                    prefix, _ = chat_tokens(
                        turns[:i], tokenizer, generation_prompt=True, mode="direct", effort="low"
                    )
                    answer = [
                        tokenizer.token_to_id("<|final|>"),
                        *tokenizer.encode(turns[i]["content"], add_special_tokens=False).ids,
                        2,
                    ]
                    end = len(prefix) + len(answer)
                    if ids[: len(prefix)] != prefix or labels[len(prefix) : end] != answer:
                        raise ValueError(
                            f"assistant prefix or answer differs at {split}/{index}/{i}"
                        )
                    expected[len(prefix) : end] = answer
                    rows["assistant_turns"] += 1
                    prompts.add(canonical_question(turns[i - 1]["content"]))
                if labels != expected:
                    raise ValueError(f"supervision outside assistant answer at {split}/{index}")
                rows[row["lang"]] += 1
                rows["examples"] += 1
                rows["assistant_tokens"] += sum(t != -100 for t in labels[1:])
        if (
            prompts & (split_prompts | blocked)
            or rows["examples"] != len(dataset)
            or not rows["examples"]
        ):
            raise ValueError(f"invalid split membership or size in {split}")
        if rows["assistant_tokens"] != record["supervised_tokens"]:
            raise ValueError(f"supervision count differs in {split}")
        split_prompts.update(prompts)
        checks[split] = dict(rows)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="MiniMind conversation JSONL")
    parser.add_argument("--revision", required=True, help="Revision of the input dataset")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-prompts", type=Path, help="JSON array of prior evaluation prompts"
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=8000,
        help="Candidate rows per language and answer-length group",
    )
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261109)
    prepare(parser.parse_args())
