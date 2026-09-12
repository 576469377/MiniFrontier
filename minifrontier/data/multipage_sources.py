"""Bounded Docmatix two-page pretraining records, with original pages and complete QAs.

This is multipage document QA, not webpage text/image interleaving. Source rows
with more than two pages are rejected whole; answer-relevant pages are never
selected by guessing. Group IDs retain the original PDF provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.public_sources import source_rows
from minifrontier.multimodal import pretraining_tokens
from minifrontier.storage import GIB, require_space, reserve_write

SOURCE: dict[str, Any] = dict(
    repo="HuggingFaceM4/Docmatix",
    revision="0725b65616e0e5f6024be10e38ddf8d8c48664fd",
    prefix="data",
    file_prefix="train-",
    bounded_http_ranges=True,
    license="MIT dataset; original PDF content rights retained",
)


def document_question(row, identity, tokenizer, *, max_text_tokens=750):
    """Select one complete QA deterministically; retain all two pages of its PDF."""
    if len(row.get("images", [])) != 2:
        return None, "not_exactly_two_pages"
    questions = row.get("texts", [])
    sources = {q.get("source") for q in questions if isinstance(q, dict)}
    if len(sources) != 1 or not all(isinstance(s, str) and s.strip() for s in sources):
        return None, "missing_or_ambiguous_pdf_origin"
    document = next(iter(sources))
    assert isinstance(document, str)
    order = sorted(
        enumerate(questions),
        key=lambda pair: hashlib.sha256(f"{identity}:{pair[0]}".encode()).digest(),
    )
    for index, qa in order:
        question, answer = qa.get("user"), qa.get("assistant")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question, answer = question.strip(), answer.strip()
        if not question or not answer or "\ufffd" in question + answer:
            continue
        record = dict(
            source=SOURCE["repo"],
            revision=SOURCE["revision"],
            item_id=f"{identity}:qa{index}",
            group_id=document,
            document_id=SOURCE["repo"] + ":" + document,
            license=SOURCE["license"],
            official_split="train",
            lang="en",
            task="multiimage_multipage",
            stage="pretrain",
            text=f"User: {question}\nAssistant: {answer}",
            visual_question=question,
            visual_answer=answer,
            source_task="complete_two_page_document_qa",
            answer_page_localization="not supplied; both original pages retained",
            media=[dict(kind="image"), dict(kind="image")],
        )
        ids, _ = pretraining_tokens(record, tokenizer)
        if len(ids) <= max_text_tokens:
            record["reference_tokens"] = len(ids)
            record["answer_reference_tokens"] = len(tokenizer.encode(answer).ids)
            return record, None
    return None, "no_complete_qa_within_text_budget"


def decode_pages(row):
    pages = []
    for source in row["images"]:
        value = source.get("bytes")
        if not isinstance(value, bytes) or not 0 < len(value) <= 16 * 1024**2:
            raise ValueError("page bytes absent or above 16 MiB")
        with Image.open(io.BytesIO(value)) as image:
            if image.width * image.height > 24_000_000 or min(image.size) < 64:
                raise ValueError("page dimensions outside the decoding bound")
            hashes = decoded_hashes(image)
        pages.append(
            (value, dict(kind="image", sha256=hashlib.sha256(value).hexdigest(), **hashes))
        )
    if pages[0][1]["rgb_sha256"] == pages[1][1]["rgb_sha256"]:
        raise ValueError("two pages have identical decoded content")
    return pages


def build_multipage(
    output,
    tokenizer_path,
    *,
    documents=5000,
    seed=42,
    max_rows=60000,
    media_gib=4.0,
    metadata_gib=0.5,
    network_gib=48.0,
):
    output = Path(output).resolve()
    if min(documents, max_rows, media_gib, metadata_gib, network_gib) <= 0:
        raise ValueError("all production limits must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a new corpus directory; existing records are immutable")
    require_space(output, int((media_gib + metadata_gib) * GIB), reserve_bytes=80 * GIB)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    builder = CorpusBuilder(output, seed=seed, max_gib=metadata_gib, group_image_phash=False)
    images = output / "images"
    images.mkdir()
    audit = dict(
        kind="docmatix_complete_two_page_candidate_corpus",
        source=SOURCE,
        seed=seed,
        requested_documents=documents,
        max_source_rows=max_rows,
        media_budget_bytes=int(media_gib * GIB),
        metadata_budget_bytes=int(metadata_gib * GIB),
        network_budget_bytes=int(network_gib * GIB),
        minimum_free_gib=80,
        tokenizer_sha256=sha256(tokenizer_path),
        max_text_tokens=750,
        source_quality_review="waived_by_maintainer",
        human_source_quality_review_completed=False,
        formal_admission=False,
        producer_finished=False,
        objective="pretraining all text; structural media labels masked by the existing native processor",
        exclusions="all rows other than exactly two complete pages; no response truncation",
    )
    counters: Counter[str] = Counter()
    source_audit: dict = {}
    stored: set[str] = set()
    documents_seen: set[str] = set()
    media_bytes = 0

    def progress(status):
        audit.update(
            status=status,
            counts=dict(counters),
            source_reads=source_audit,
            media_bytes=media_bytes,
            stored_images=len(stored),
        )
        write_json(output / "source-audit.json", audit)

    try:
        progress("producing")
        specification = dict(SOURCE, network_byte_budget=int(network_gib * GIB))
        for row, identity in source_rows(
            "docmatix", seed=seed, audit=source_audit, specification=specification
        ):
            counters["source_rows"] += 1
            record, reason = document_question(row, identity, tokenizer)
            if record is None:
                counters[reason] += 1
            elif record["document_id"] in documents_seen:
                counters["repeated_pdf_origin"] += 1
            else:
                try:
                    pages = decode_pages(row)
                except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                    counters["invalid_page"] += 1
                    pages = []
                if pages:
                    incoming = sum(len(value) for value, m in pages if m["sha256"] not in stored)
                    if media_bytes + incoming > media_gib * GIB:
                        counters["media_budget_stop"] += 1
                        break
                    written = []
                    for value, m in pages:
                        relative = "images/" + m["sha256"] + ".image"
                        m.update(path=relative, document_id=record["document_id"])
                        if m["sha256"] not in stored:
                            with reserve_write(output, len(value), reserve_bytes=80 * GIB):
                                (output / relative).write_bytes(value)
                            stored.add(m["sha256"])
                            media_bytes += len(value)
                            written.append(m)
                    record["media"] = [m for _, m in pages]
                    if builder.add(record):
                        counters["accepted_documents"] += 1
                        documents_seen.add(record["document_id"])
                    else:
                        for m in written:
                            path = output / m["path"]
                            media_bytes -= path.stat().st_size
                            path.unlink()
                            stored.remove(m["sha256"])
                        counters["corpus_rejection"] += 1
                    builder.db.commit()
            if counters["source_rows"] % 100 == 0:
                progress("producing")
            if counters["accepted_documents"] >= documents or counters["source_rows"] >= max_rows:
                break
        builder.db.commit()
        db_bytes = (output / "corpus.sqlite").stat().st_size
        if 2 * db_bytes + 16 * 1024**2 > metadata_gib * GIB:
            raise ValueError("metadata finalization would exceed its storage budget")
        require_space(output, db_bytes + 16 * 1024**2, reserve_bytes=80 * GIB)
        audit["corpus"] = builder.finalize()
        audit["producer_finished"] = True
        audit["corpus_manifest_sha256"] = sha256(output / "corpus-manifest.json")
        audit["dedup_counts"] = dict(builder.counts)
        audit["split_records"] = dict(
            builder.db.execute("SELECT split,COUNT(*) FROM samples GROUP BY split")
        )
        audit["split_groups"] = dict(
            builder.db.execute(
                "SELECT split,COUNT(DISTINCT group_root) FROM samples GROUP BY split"
            )
        )
        progress("candidate_corpus_complete_pending_incremental_identity_check")
        return audit
    except BaseException as error:
        audit["error"] = type(error).__name__ + ": " + str(error)
        progress("failed_requires_attention")
        raise
    finally:
        builder.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("tokenizer")
    parser.add_argument("--documents", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--media-gib", type=float, default=4.0)
    parser.add_argument("--metadata-gib", type=float, default=0.5)
    parser.add_argument("--network-gib", type=float, default=48.0)
    args = parser.parse_args()
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    result = build_multipage(
        args.output,
        args.tokenizer,
        documents=args.documents,
        seed=args.seed,
        max_rows=args.max_rows,
        media_gib=args.media_gib,
        metadata_gib=args.metadata_gib,
        network_gib=args.network_gib,
    )
    print(json.dumps({k: result[k] for k in ("status", "counts", "split_records", "split_groups")}))


if __name__ == "__main__":
    main()
