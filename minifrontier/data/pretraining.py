"""Construct a bounded candidate text slice; admission is a separate, evidence-bound step.

Reuse the corpus deduplicator and pinned public Parquet reader. This command
never trains models, downloads whole datasets, or promotes its output to formal
data. Token counts use an explicitly named reference tokenizer, not a byte guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

from tokenizers import Tokenizer

from minifrontier.data import normalized, sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.public_sources import SOURCES, normalized_source, source_rows
from minifrontier.storage import GIB, require_space

TEXT_SOURCES = {k: dict(SOURCES[k]) for k in ("zh_edu", "en_edu", "ultrachat")}
TEXT_SOURCES["openwebmath"] = dict(
    repo="open-web-math/open-web-math",
    revision="fde8ef8de2300f5e778f56261843dab89f230815",
    prefix="data",
    license="ODC-BY-1.0; Common Crawl terms; underlying page rights retained",
    task="verified_math_science",
    lang="en",
)
DEFAULT_TARGETS = dict(
    zh_edu=225_000_000, en_edu=150_000_000, openwebmath=50_000_000, ultrachat=40_000_000
)


def clean_record(name, row, identity):
    """Return an auditable normalized record, or a concrete rejection reason."""
    spec = TEXT_SOURCES[name]
    if name == "openwebmath":
        if not row.get("url") or not row.get("text"):
            return None, "missing_origin_or_text"
        record = dict(
            source=spec["repo"],
            revision=spec["revision"],
            item_id=identity,
            group_id=row["url"],
            license=spec["license"],
            lang="en",
            task=spec["task"],
            stage="pretrain",
            text=row["text"],
            source_metadata={k: row[k] for k in ("url", "date", "metadata") if k in row},
        )
    else:
        if name == "zh_edu":
            score = row.get("score")
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                return None, "score_outside_declared_0_1_range"
        record = normalized_source(name, row, identity)
        if record is None:
            return None, "missing_payload"
        if name == "ultrachat":
            if any(
                t.get("role") not in {"system", "user", "assistant"}
                or not isinstance(t.get("content"), str)
                for t in record["turns"]
            ):
                return None, "invalid_dialogue"
            record["text"] = "\n\n".join(
                t["role"].capitalize() + ": " + t["content"] for t in record.pop("turns")
            )
            record["stage"] = "pretrain"
    raw = record["text"]
    text = normalized(raw)
    if not 100 <= len(text) <= 100_000:
        return None, "length_outside_100_100000_characters"
    if "\ufffd" in text or any(ord(c) < 32 and c not in "\n\t\r" for c in text):
        return None, "invalid_encoding_or_controls"
    if len(re.findall(r"<\s*(?:script|style|div|span|html|body|nav)\b", text, re.I)) > 3:
        return None, "unremoved_html"
    letters = sum(c.isalpha() for c in text)
    if letters < 30:
        return None, "insufficient_language_content"
    chinese = sum("\u3400" <= c <= "\u9fff" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if (spec["lang"] == "zh" and chinese / letters < 0.4) or (
        spec["lang"] == "en" and latin / letters < 0.7
    ):
        return None, "language_script_mismatch"
    if name == "zh_edu" and not row.get("source"):
        return None, "missing_chinese_subsource"
    url = record.get("source_metadata", {}).get("url")
    if url:
        record["document_id"] = urldefrag(url)[0]
    record.update(
        text=text,
        raw_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        quality_flags=["candidate-not-formally-admitted"],
    )
    return record, None


def build_text_slice(output, reference_tokenizer, *, targets=None, seed=20260910, max_gib=12):
    root, reference = Path(output).resolve(), Path(reference_tokenizer).resolve()
    targets = DEFAULT_TARGETS if targets is None else targets
    if not targets or any(k not in TEXT_SOURCES or n <= 0 for k, n in targets.items()):
        raise ValueError(
            "choose positive budgets for reviewed text sources; code stays quarantined"
        )
    if root.exists():
        raise FileExistsError("choose a new immutable candidate slice")
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib, val_buckets=50, test_buckets=100)
    tokenizer = Tokenizer.from_file(str(reference))
    shutil.copyfile(reference, root / "reference-tokenizer.json")
    audit = dict(
        schema_version=1,
        status="building",
        formal_admission=False,
        seed=seed,
        reference_tokenizer_sha256=sha256(reference),
        reference_is_final_training_tokenizer=False,
        target_candidate_tokens=targets,
        max_gib=max_gib,
        minimum_free_gib_for_new_allocation=80,
        processor_files={
            str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("corpus.py"),
                Path(__file__).with_name("public_sources.py"),
            )
        },
        sources={},
        unresolved=[
            "code repository/blob/license evidence",
            "benchmark identity exclusion and sealed evaluation inventory",
            "stratified manual review of at least 100 records per major source",
            "final tokenizer quality and per-model stage mixture admission",
            "image/OCR/chart/video sources and independent held-out media",
        ],
    )
    path = root / "source-audit.json"

    def progress():
        builder.db.commit()
        audit["updated_unix"] = time.time()
        audit["free_gib"] = shutil.disk_usage(root).free / GIB
        audit["database_bytes"] = (root / "corpus.sqlite").stat().st_size
        audit["dedup_and_quality_counts"] = dict(builder.counts)
        write_json(path, audit)

    write_json(
        root / "source_allowlist.json",
        dict(
            status="candidate_extraction_only",
            formal_admission=False,
            sources={k: TEXT_SOURCES[k] for k in targets},
            quarantined={
                "python_edu": "original repository/license mapping unresolved; no replacement"
            },
        ),
    )
    try:
        progress()
        for source_index, (name, target) in enumerate(targets.items()):
            entry: dict[str, Any] = dict(
                specification=TEXT_SOURCES[name],
                accepted_records=0,
                accepted_reference_tokens=0,
                rejected=Counter(),
                rejection_examples={},
                status="reading",
            )
            audit["sources"][name] = entry
            progress()
            for row, identity in source_rows(
                name,
                seed=seed + source_index * 104729,
                audit=entry,
                specification=TEXT_SOURCES[name],
            ):
                record, reason = clean_record(name, row, identity)
                if record is None:
                    entry["rejected"][reason] += 1
                    examples = entry["rejection_examples"].setdefault(reason, [])
                    if len(examples) < 20:
                        examples.append(
                            dict(
                                identity=identity,
                                score=str(row.get("score")),
                                subsource=row.get("source"),
                            )
                        )
                    continue
                ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
                record["reference_tokens"] = len(ids)
                if not builder.add(record):
                    continue
                entry["accepted_records"] += 1
                entry["accepted_reference_tokens"] += len(ids)
                if entry["accepted_records"] % 1000 == 0:
                    # Stop ingestion before the reserved checkpoint headroom is spent.
                    require_space(root, 64 * 1024**2, reserve_bytes=80 * GIB)
                    progress()
                    print(
                        json.dumps(
                            dict(
                                source=name,
                                records=entry["accepted_records"],
                                tokens=entry["accepted_reference_tokens"],
                            )
                        ),
                        flush=True,
                    )
                if entry["accepted_reference_tokens"] >= target:
                    break
            entry["status"] = (
                "candidate_target_reached"
                if entry["accepted_reference_tokens"] >= target
                else "source_exhausted_below_target"
            )
            progress()
        audit["corpus"] = builder.finalize()
        counts: dict[str, Counter] = {}
        review = root / "review-samples.jsonl"
        selected: Counter[str] = Counter()
        with review.open("w") as handle:
            for split, source, payload in builder.db.execute(
                "SELECT split,source,payload FROM samples ORDER BY id"
            ):
                row = json.loads(payload)
                counts.setdefault(split, Counter())[source] += row["reference_tokens"]
                # Hash order provides a stable content-based sample; sealed test stays sealed.
                if split == "train" and selected[source] < 100:
                    handle.write(
                        json.dumps(dict(split=split, record=row), ensure_ascii=False) + "\n"
                    )
                    selected[source] += 1
        audit["split_reference_tokens"] = counts
        audit["review"] = dict(
            status="awaiting_manual_review", samples=dict(selected), sha256=sha256(review)
        )
        audit["status"] = "candidate_slice_complete_pending_admission"
        progress()
    except BaseException as error:
        audit.update(
            status="interrupted_unadmitted", error=type(error).__name__ + ": " + str(error)
        )
        progress()
        raise
    finally:
        builder.db.close()
    return audit


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-tokenizer", required=True)
    parser.add_argument("--targets", help="JSON object of accepted reference-token targets")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-gib", type=float, default=12)
    args = parser.parse_args(argv)
    build_text_slice(
        args.output,
        args.reference_tokenizer,
        targets=json.loads(args.targets) if args.targets else None,
        seed=args.seed,
        max_gib=args.max_gib,
    )


if __name__ == "__main__":
    main()
