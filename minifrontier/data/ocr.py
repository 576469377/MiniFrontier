"""Source-grounded, legible P0 OCR images; no model-generated transcription labels."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

from minifrontier.data import fingerprint, sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.partitions import open_corpus
from minifrontier.storage import GIB, require_space, reserve_write

SOURCE = "MiniFrontier/source-grounded-ocr"
SOURCE_TASKS = ("zh_edu", "en_edu", "verified_math_science", "code", "dialogue")
LAYOUTS = {"train": ("plain", "ruled", "note"), "val": ("border",), "test": ("side_rule",)}
QUESTIONS = {
    "zh": "请逐行转写图片中的全部文字，保留标点和换行。",  # noqa: RUF001
    "en": "Transcribe all printed text in reading order, preserving punctuation and line breaks.",
}


class Renderer:
    """Draw only complete, supported glyphs in a 224px image, with exact line targets."""

    def __init__(self, font_manifest):
        self.path = Path(font_manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        used: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            used[split] = set()
            for group in self.manifest["split_fonts"][split]:
                if not group:
                    raise ValueError("every font group needs at least one font")
                for key in group:
                    used[split].add(self.manifest["assets"][key]["sha256"])
                    license_key = self.manifest["license_map"][key]
                    for name in (key, license_key):
                        asset = self.manifest["assets"][name]
                        path = (self.path.parent / asset["path"]).resolve()
                        if (
                            not path.is_relative_to(self.path.parent)
                            or sha256(path) != asset["sha256"]
                        ):
                            raise ValueError("font or license resource differs from its manifest")
            if not used[split]:
                raise ValueError("each split needs its own font family")
        if any(
            used[a] & used[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError("OCR held-out fonts must be disjoint from training and one another")
        self.font = lru_cache(maxsize=64)(self._font)
        self.glyph = lru_cache(maxsize=131072)(self._glyph)

    def _font(self, key, size):
        return ImageFont.truetype(
            str(self.path.parent / self.manifest["assets"][key]["path"]), size
        )

    def _glyph(self, group, size, char):
        if unicodedata.category(char)[0] in {"C", "M"}:
            raise ValueError("unsupported_glyph")
        for key in group:
            font = self.font(key, size)
            mask = bytes(font.getmask(char))
            if char == " " or (any(mask) and mask != bytes(font.getmask("\U0010ffff"))):
                return key, font.getlength(char), font.getbbox(char, anchor="ls")
        raise ValueError("unsupported_glyph")

    def render(self, text, identity, split, lang, seed):
        if split not in LAYOUTS or lang not in QUESTIONS:
            raise ValueError("unsupported source split/language")
        clean = " ".join(text.split())
        if len(clean) < 80:
            raise ValueError("short_source")
        rng = random.Random(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest())
        group = tuple(rng.choice(self.manifest["split_fonts"][split]))
        size = rng.choice((18, 20, 22))
        family = rng.choice(LAYOUTS[split])
        margin, gap = rng.randint(10, 17), rng.randint(2, 5)
        start = rng.randrange(max(1, len(clean) - 500))
        if lang == "en" and start:
            boundary = clean.find(" ", start)
            if boundary < 0:
                raise ValueError("no_word_boundary")
            start = boundary + 1
        passage = clean[start : start + 1000]
        units = re.findall(r"[A-Za-z0-9_]+(?:['’.-][A-Za-z0-9]+)*| +|.", passage)  # noqa: RUF001
        ascent = max(self.font(key, size).getmetrics()[0] for key in group)
        descent = max(self.font(key, size).getmetrics()[1] for key in group)
        line_height = ascent + descent + gap
        limit = (224 - 2 * margin) // line_height
        lines, line, width = [], "", 0.0
        for unit in units:
            if unit.isspace() and not line:
                continue
            advance = sum(self.glyph(group, size, char)[1] for char in unit)
            if advance > 224 - 2 * margin:
                raise ValueError("word_exceeds_line")
            if width + advance > 224 - 2 * margin:
                lines.append(line.rstrip())
                line, width = "", 0.0
                if len(lines) >= limit:
                    break
                if unit.isspace():
                    continue
            line += unit
            width += advance
        else:
            if line:
                lines.append(line.rstrip())
        answer = "\n".join(lines)
        if len(answer) < 30 or len(lines) < 2:
            raise ValueError("insufficient_rendered_text")
        paper = rng.choice(((255, 255, 255), (250, 248, 240), (237, 246, 252)))
        ink = rng.choice(((20, 22, 27), (25, 40, 62), (55, 31, 28)))
        image = Image.new("RGB", (224, 224), paper)
        draw = ImageDraw.Draw(image)
        if family == "border":
            draw.rectangle((3, 3, 220, 220), outline=(130, 130, 130), width=1)
        elif family == "side_rule":
            draw.line((4, 6, 4, 218), fill=(55, 90, 130), width=3)
        elif family == "note":
            draw.rectangle((0, 0, 223, 4), fill=(200, 182, 125))
        boxes = []
        for number, line in enumerate(lines):
            y, x = margin + number * line_height + ascent, float(margin)
            bounds = []
            if family == "ruled":
                draw.line(
                    (margin, y + descent + 1, 224 - margin, y + descent + 1), fill=(212, 216, 225)
                )
            for char in line:
                key, advance, box = self.glyph(group, size, char)
                actual = (x + box[0], y + box[1], x + box[2], y + box[3])
                if not (0 <= actual[0] <= actual[2] < 224 and 0 <= actual[1] <= actual[3] < 224):
                    raise ValueError("glyph_would_be_clipped")
                draw.text((x, y), char, font=self.font(key, size), fill=ink, anchor="ls")
                if char != " ":
                    bounds.append(actual)
                x += advance
            boxes.append(
                [
                    min(b[0] for b in bounds),
                    min(b[1] for b in bounds),
                    max(b[2] for b in bounds),
                    max(b[3] for b in bounds),
                ]
            )
        return (
            image,
            answer,
            dict(
                family=family,
                fonts=list(group),
                font_size=size,
                line_height=line_height,
                line_boxes=boxes,
                source_start=start,
                normalized_source_sha256=hashlib.sha256(clean.encode()).hexdigest(),
                text_selection="contiguous normalized source; only complete rendered lines become targets",
                model_input_size=[224, 224],
                vision_features=49,
            ),
        )


def generate_ocr(
    corpus,
    font_manifest,
    reference_tokenizer,
    output,
    *,
    id_start="0",
    id_stop="1",
    max_gib=4,
    metadata_gib=2,
    seed=20260912,
    source_tasks=("zh_edu", "en_edu"),
    max_records=None,
):
    """Build one disjoint source-ID slice; never assign a text-training source to OCR holdout."""
    source, root = Path(corpus).resolve(), Path(output).resolve()
    if (
        not isinstance(source_tasks, (tuple, list))
        or not source_tasks
        or any(task not in SOURCE_TASKS for task in source_tasks)
        or len(set(source_tasks)) != len(source_tasks)
    ):
        raise ValueError("OCR source tasks must be an explicit unique selection of supported tasks")
    if max_records is not None and (type(max_records) is not int or max_records <= 0):
        raise ValueError("OCR max_records must be a positive integer or None")
    if (
        root.exists()
        or not (0 < metadata_gib < max_gib)
        or not re.fullmatch(r"[0-9a-f]{1,64}", id_start)
        or not re.fullmatch(r"[0-9a-f]{1,64}|g", id_stop)
        or id_start >= id_stop
    ):
        raise ValueError(
            "choose a new OCR shard, disjoint ordered hex-ID range and bounded storage"
        )
    parent = json.loads((source / "source-audit.json").read_text())
    if parent["status"] != "candidate_slice_complete_pending_admission":
        raise ValueError("OCR text source must be a completed canonical candidate")
    renderer = Renderer(font_manifest)
    tokenizer = Tokenizer.from_file(str(reference_tokenizer))
    header_bytes = Path(reference_tokenizer).stat().st_size + 2 * 1024**2
    if header_bytes >= metadata_gib * GIB:
        raise ValueError("OCR metadata cap cannot hold the tokenizer and audit records")
    require_space(root, int(max_gib * GIB), reserve_bytes=80 * GIB)
    builder = CorpusBuilder(
        root,
        seed=seed,
        max_gib=metadata_gib - header_bytes / GIB,
        val_buckets=50,
        test_buckets=100,
        group_image_phash=False,
    )
    (root / "images").mkdir()
    font_binding = dict(
        path=os.path.relpath(Path(font_manifest).resolve(), root), sha256=sha256(font_manifest)
    )
    write_json(root / "font-assets.json", font_binding)
    shutil.copyfile(reference_tokenizer, root / "reference-tokenizer.json")
    audit = dict(
        kind="source_grounded_ocr_candidate",
        status="building",
        formal_admission=False,
        main_budget_eligible=False,
        source=SOURCE,
        revision=sha256(__file__),
        seed=seed,
        corpus_manifest_sha256=sha256(source / "corpus-manifest.json"),
        parent_audit_sha256=sha256(source / "source-audit.json"),
        font_manifest_sha256=sha256(font_manifest),
        font_assets=font_binding,
        reference_tokenizer_sha256=sha256(reference_tokenizer),
        source_id_range=[id_start, id_stop],
        source_tasks=list(source_tasks),
        max_records=max_records,
        max_gib=max_gib,
        metadata_gib=metadata_gib,
        media_bytes=0,
        counts={},
        splits={},
        sources={},
        started_unix=time.time(),
        remaining=[
            "bound cross-corpus known-transcription and external-image candidate audit",
            "source and rendered-image human quality review",
            "external visual benchmark exclusion",
            "full encoding and per-model CE/length/exposure admission",
        ],
    )
    counts: Counter[str] = Counter()
    splits: dict[str, Counter[str]] = {}
    locks, transcriptions = {}, set()
    review: list[dict[str, Any]] = []

    def progress():
        builder.db.commit()
        audit.update(
            counts=dict(counts),
            splits=splits,
            updated_unix=time.time(),
            free_gib=shutil.disk_usage(root).free / GIB,
        )
        write_json(root / "source-audit.json", audit)

    progress()
    try:
        with contextlib.closing(open_corpus(source)) as db:
            # Seal test first. Similar page layouts are not printed-text identities;
            # the cross-corpus text-aware audit resolves visual candidates later.
            for split in ("test", "val", "train"):
                if max_records is not None and counts["accepted"] >= max_records:
                    break
                placeholders = ",".join("?" for _ in source_tasks)
                query = f"SELECT payload,group_root FROM samples WHERE stage='pretrain' AND task IN ({placeholders}) AND split=? AND id>=? AND id<? ORDER BY id"
                for payload, group in db.execute(query, (*source_tasks, split, id_start, id_stop)):
                    original = json.loads(payload)
                    # Code sources use lang="code"; use the English transcription
                    # instruction while retaining their original language below.
                    lang = {"zh_edu": "zh", "en_edu": "en", "code": "en"}.get(
                        original["task"], original.get("lang")
                    )
                    counts["examined"] += 1
                    if lang not in QUESTIONS:
                        counts["unsupported_source_language"] += 1
                        continue
                    try:
                        image, answer, layout = renderer.render(
                            original["text"], original["sample_id"], split, lang, seed
                        )
                    except ValueError as error:
                        counts[str(error)] += 1
                        continue
                    quote = fingerprint(re.sub(r"\s+", "", answer))
                    if quote in transcriptions:
                        counts["duplicate_transcription"] += 1
                        continue
                    hashes = decoded_hashes(image)
                    question = QUESTIONS[lang]
                    text = question + " " + answer
                    identity = fingerprint(text + "|media:" + hashes["rgb_sha256"])
                    bucket = (
                        int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16], 16)
                        % 10000
                    )
                    if split == "train" and bucket < 150:
                        counts["would_create_text_training_ocr_holdout"] += 1
                        continue
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    raw = buffer.getvalue()
                    if audit["media_bytes"] + len(raw) > (max_gib - metadata_gib) * GIB:
                        raise ValueError("OCR raw image byte cap reached")
                    path = root / "images" / (hashes["rgb_sha256"] + ".png")
                    origin = {
                        k: original[k]
                        for k in (
                            "source",
                            "revision",
                            "item_id",
                            "license",
                            "sample_id",
                            "content_hash",
                        )
                    }
                    origin.update(
                        split=split, group_root=group, task=original["task"], lang=original["lang"]
                    )
                    record = dict(
                        source=SOURCE,
                        revision=sha256(__file__),
                        item_id=original["sample_id"],
                        group_id=group,
                        document_id=group,
                        license=original["license"]
                        + "; shared font licenses bound by font-assets.json",
                        lang=lang,
                        task="ocr_document",
                        stage="pretrain",
                        text=text,
                        visual_question=question,
                        visual_answer=answer,
                        reference_tokens=len(tokenizer.encode(text).ids),
                        answer_reference_tokens=len(tokenizer.encode(answer).ids),
                        text_origin=origin,
                        rendering=layout,
                        media=[
                            dict(
                                kind="image",
                                path=str(path.relative_to(root)),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                **hashes,
                            )
                        ],
                    )
                    with reserve_write(path, len(raw), reserve_bytes=80 * GIB):
                        path.write_bytes(raw)
                    if not builder.add(record):
                        path.unlink()
                        counts["builder_rejected"] += 1
                        continue
                    if split != "train":
                        locks["source-group:" + SOURCE + ":" + group] = split
                    transcriptions.add(quote)
                    audit["media_bytes"] += len(raw)
                    counts["accepted"] += 1
                    splits.setdefault(split + "." + lang, Counter()).update(
                        records=1, answer_reference_tokens=record["answer_reference_tokens"]
                    )
                    audit["sources"][original["source"]] = {
                        k: original[k] for k in ("revision", "license")
                    }
                    if split == "train" and sum(r["record"]["lang"] == lang for r in review) < 100:
                        review.append(
                            dict(
                                split=split, record=dict(record, sample_id=builder.last_retained_id)
                            )
                        )
                    if counts["accepted"] % 250 == 0:
                        progress()
                    if max_records is not None and counts["accepted"] >= max_records:
                        break
        manifest = builder.finalize(split_locks=locks)
        mismatches = builder.db.execute(
            "SELECT count(*) FROM samples WHERE split!=json_extract(payload,'$.text_origin.split')"
        ).fetchone()[0]
        if mismatches:
            raise ValueError("OCR partition differs from its immutable text origins")
        audit.update(
            status="candidate_slice_complete_pending_admission",
            corpus=manifest,
            split_origin_mismatches=0,
            original_text_reused_across_modalities=True,
            independent_text_supply_added=0,
            stop_reason=(
                "accepted_record_limit"
                if max_records is not None and counts["accepted"] >= max_records
                else "source_selection_exhausted"
            ),
            completed_unix=time.time(),
        )
        (root / "review-samples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review)
        )
        write_json(
            root / "source_allowlist.json",
            dict(
                status="candidate_derivatives_only",
                formal_admission=False,
                parents=audit["sources"],
                font_manifest_sha256=audit["font_manifest_sha256"],
            ),
        )
        progress()
        metadata_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
        if metadata_bytes + audit["media_bytes"] > max_gib * GIB:
            raise ValueError("OCR data and audit files exceed the total byte budget")
        return audit
    except BaseException as error:
        audit.update(status="failed", error=repr(error))
        progress()
        (root / "corpus-manifest.json").unlink(missing_ok=True)
        raise
    finally:
        builder.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "font_manifest", "reference_tokenizer", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--id-start", default="0")
    parser.add_argument("--id-stop", default="1")
    parser.add_argument("--max-gib", type=float, default=4)
    parser.add_argument("--metadata-gib", type=float, default=2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--source-tasks", nargs="+", choices=SOURCE_TASKS, default=["zh_edu", "en_edu"]
    )
    parser.add_argument("--max-records", type=int)
    print(json.dumps(generate_ocr(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
