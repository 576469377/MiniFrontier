"""Small immutable media-identity exports for grouping corpora held on different hosts."""

import contextlib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minifrontier.data import sha256
from minifrontier.data.media_hash import PHASH_BANDS
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.storage import GIB, reserve_write

FORMAT = "media-group-identities-v1"
TEXT_FORMAT = "media-group-identities-v2"
OCR_SOURCE = "MiniFrontier/source-grounded-ocr"


def _rendered_signature(row):
    from minifrontier.data.corpus import text_shingles

    if row.get("source") != OCR_SOURCE:
        return None
    if not row.get("rendering") or not row.get("text_origin") or not row.get("visual_answer"):
        raise ValueError("rendered OCR requires its printed text and source provenance")
    text = row["visual_answer"]
    return dict(
        sha256=hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest(),
        shingles=sorted(
            hashlib.blake2b(s.encode(), digest_size=8).hexdigest() for s in text_shingles(text)
        ),
    )


def _text_neighbors(signatures, *, max_comparisons):
    """Exact >=0.85 Jaccard join of hashed 5-gram sets using shared prefix candidates.

    A common global token order and n-ceil(0.85*n)+1 prefixes preserve recall
    at this threshold, independently of the font, layout or image pHash.
    """
    frequency = Counter(token for values in signatures.values() for token in values)
    index: dict[Any, set[Any]] = defaultdict(set)
    comparisons = 0
    for key, values in signatures.items():
        ordered = sorted(values, key=lambda token: (frequency[token], token))
        prefix = ordered[: len(values) - (85 * len(values) + 99) // 100 + 1]
        candidates = set().union(*(index[token] for token in prefix))
        for other in sorted(candidates):
            previous = signatures[other]
            if 100 * min(len(values), len(previous)) < 85 * max(len(values), len(previous)):
                continue
            comparisons += 1
            if comparisons > max_comparisons:
                raise ValueError("rendered-text candidate comparison budget exceeded")
            intersection = len(values & previous)
            if 100 * intersection >= 85 * (len(values) + len(previous) - intersection):
                yield key, other
        for token in prefix:
            index[token].add(key)


def _write_new(path, record, max_bytes):
    path = Path(path)
    if path.exists():
        raise FileExistsError("media identity records are immutable; choose a new output")
    content = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    if len(content.encode()) > max_bytes:
        raise ValueError("media identity metadata exceeds the declared byte budget")
    with (
        reserve_write(path, len(content.encode()), reserve_bytes=80 * GIB),
        path.open("x") as stream,
    ):
        stream.write(content)


def export_media_identities(corpus, output, *, max_images=100_000, max_bytes=64 * 1024**2):
    """Export actual effective groups and origin aliases without images or QA content."""
    corpus = Path(corpus).resolve()
    if Path(output).exists() or max_images < 1 or max_bytes < 1:
        raise ValueError("choose a new output and positive identity inventory bounds")
    manifest = json.loads((corpus / "corpus-manifest.json").read_text())
    groups: dict[str, Any] = {}
    images: dict[str, dict[str, Any]] = {}
    with contextlib.closing(open_corpus(corpus)) as db:
        storage = corpus_storage_root(db)
        if sha256(storage / "corpus.sqlite") != manifest["database_sha256"]:
            raise ValueError("media identity source database changed")
        for split, group, payload in db.execute(
            "SELECT split,group_root,payload FROM samples ORDER BY id"
        ):
            row = json.loads(payload)
            if not row.get("media"):
                continue
            if split not in {"train", "val", "test"} or not group:
                raise ValueError("media record has no effective group/split")
            node = groups.setdefault(group, dict(split=split, records=0, origin_keys=[]))
            if node["split"] != split:
                raise ValueError("one source media group crosses effective splits")
            node["records"] += 1
            for media in row["media"]:
                if (
                    media.get("kind") != "image"
                    or media.get("phash_version") != "dct32-low8-median-v1"
                ):
                    raise ValueError("this inventory requires still images with the declared pHash")
                identity = media["rgb_sha256"]
                value: dict[str, Any] = dict(group=group, phash=media["phash"])
                signature = _rendered_signature(row)
                if signature is not None:
                    value["rendered_text"] = signature
                if identity in images and images[identity] != value:
                    raise ValueError("one RGB identity has inconsistent source groups or hashes")
                images[identity] = value
                if len(images) > max_images or len(groups) > max_images:
                    raise ValueError("media identity inventory exceeds its image/group bound")
        # Retain aliases of rejected duplicates, including groups with another origin.
        for key, group in db.execute(
            "SELECT DISTINCT l.key,s.group_root FROM main.links l JOIN samples s ON s.id=l.id "
            "WHERE l.key GLOB 'source-group:*' OR l.key GLOB 'document:*' "
            "OR l.key GLOB 'repo:*' OR l.key GLOB 'document_id:*' "
            "OR l.key GLOB 'video_id:*' OR l.key GLOB 'origin_id:*' ORDER BY l.key,s.group_root"
        ):
            if group in groups:
                groups[group]["origin_keys"].append(key)
    record = dict(
        format=TEXT_FORMAT,
        corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        database_sha256=manifest["database_sha256"],
        processor_sha256=sha256(__file__),
        groups=groups,
        images=images,
        image_count=len(images),
        group_count=len(groups),
        split_images=dict(Counter(groups[v["group"]]["split"] for v in images.values())),
        split_groups=dict(Counter(v["split"] for v in groups.values())),
        formal_admission=False,
        raw_media_copied=False,
        raw_pixels_redecoded=False,
        scope="all declared still-image RGB/pHash values, effective split groups and stored origin aliases; canonical raw-pixel integrity is a separate bound audit",
    )
    _write_new(output, record, max_bytes)
    return record


def audit_media_identities(
    inventories,
    output,
    *,
    max_edges=100_000,
    max_bytes=64 * 1024**2,
    max_text_comparisons=5_000_000,
):
    """Group source/pixel identities, natural pHash and known printed OCR content.

    OCR-to-external pHash matches remain explicit unresolved candidates: a
    similar page layout alone cannot establish matching printed content.
    """
    if len(inventories) < 2 or min(max_edges, max_text_comparisons) < 1 or Path(output).exists():
        raise ValueError("choose at least two inventories, a positive edge cap and a new output")
    data, bindings, nodes = {}, {}, {}
    ocr_nodes = set()
    for name, path in inventories.items():
        path = Path(path)
        if not isinstance(name, str) or not name or path.stat().st_size > max_bytes:
            raise ValueError("invalid media inventory label or byte bound")
        item = json.loads(path.read_text())
        if (
            item["format"] not in {FORMAT, TEXT_FORMAT}
            or item["image_count"] != len(item["images"])
            or item["group_count"] != len(item["groups"])
        ):
            raise ValueError("media inventory format or counts differ")
        for group, record in item["groups"].items():
            if record["split"] not in {"train", "val", "test"}:
                raise ValueError("media inventory has an unknown split")
            nodes[(name, group)] = record
            if any(
                key.startswith("source-group:" + OCR_SOURCE + ":") for key in record["origin_keys"]
            ):
                if item["format"] == FORMAT:
                    raise ValueError("rendered OCR needs a text-aware identity export")
                ocr_nodes.add((name, group))
        data[name] = item
        bindings[name] = dict(
            sha256=sha256(path),
            corpus_manifest_sha256=item["corpus_manifest_sha256"],
            database_sha256=item["database_sha256"],
        )
    parents = {node: node for node in nodes}

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    edges = set()

    def connect(a, b, reason):
        if a == b:
            return
        edges.add((*sorted((a, b)), reason))
        if len(edges) > max_edges:
            raise ValueError("media grouping exceeds the declared edge bound; no pass published")
        left, right = find(a), find(b)
        parents[max(left, right)] = min(left, right)

    origins: dict[str, tuple[str, str]] = {}
    pixels: dict[str, tuple[str, str]] = {}
    bands: dict[tuple[int, int], set[tuple[int, tuple[str, str], str]]] = defaultdict(set)
    rendered = {}
    image_nodes = {}
    unresolved = []
    for name, item in data.items():
        for group, record in item["groups"].items():
            node = (name, group)
            for key in record["origin_keys"]:
                connect(node, origins.setdefault(key, node), "origin_alias")
        for identity, image in item["images"].items():
            node = (name, image["group"])
            if (
                node not in nodes
                or len(identity) != 64
                or any(c not in "0123456789abcdef" for c in identity)
            ):
                raise ValueError("media inventory has an invalid RGB identity/group")
            if len(image["phash"]) != 16 or any(
                c not in "0123456789abcdef" for c in image["phash"]
            ):
                raise ValueError("media inventory has an invalid pHash")
            connect(node, pixels.setdefault(identity, node), "exact_rgb")
            image_key = (name, identity)
            image_nodes[image_key] = node
            signature = image.get("rendered_text")
            if node in ocr_nodes and signature is None:
                raise ValueError("rendered OCR identity export lacks its text signature")
            if signature is not None:
                if (
                    item["format"] != TEXT_FORMAT
                    or not isinstance(signature, dict)
                    or not isinstance(signature.get("shingles"), list)
                    or not signature["shingles"]
                    or len(signature["shingles"]) != len(set(signature["shingles"]))
                    or any(
                        not isinstance(s, str) or re.fullmatch("[0-9a-f]{16}", s) is None
                        for s in signature["shingles"]
                    )
                    or not isinstance(signature.get("sha256"), str)
                    or re.fullmatch("[0-9a-f]{64}", signature["sha256"]) is None
                ):
                    raise ValueError("invalid rendered-text signature")
                rendered[image_key] = set(signature["shingles"])
                continue
            code = int(image["phash"], 16)
            candidates = set()
            for band, (offset, width) in enumerate(PHASH_BANDS):
                candidates.update(bands[(band, (code >> offset) & ((1 << width) - 1))])
            for other_code, other, _rgb in candidates:
                if (code ^ other_code).bit_count() <= 6:
                    connect(node, other, "phash_hamming_le_6")
            for band, (offset, width) in enumerate(PHASH_BANDS):
                bands[(band, (code >> offset) & ((1 << width) - 1))].add((code, node, identity))
    for left, right in _text_neighbors(rendered, max_comparisons=max_text_comparisons):
        connect(image_nodes[left], image_nodes[right], "rendered_text_jaccard_ge_0.85")
    for name, identity in rendered:
        node = image_nodes[(name, identity)]
        code = int(data[name]["images"][identity]["phash"], 16)
        candidates = set()
        for band, (offset, width) in enumerate(PHASH_BANDS):
            candidates.update(bands[(band, (code >> offset) & ((1 << width) - 1))])
        for other_code, other, rgb in sorted(candidates):
            distance = (code ^ other_code).bit_count()
            if distance <= 6 and rgb != identity and node != other:
                unresolved.append(
                    dict(
                        rendered=dict(
                            inventory=name,
                            group=node[1],
                            rgb_sha256=identity,
                            split=nodes[node]["split"],
                        ),
                        external=dict(
                            inventory=other[0],
                            group=other[1],
                            rgb_sha256=rgb,
                            split=nodes[other]["split"],
                        ),
                        phash_distance=distance,
                    )
                )
                if len(unresolved) + len(edges) > max_edges:
                    raise ValueError("media candidate/edge budget exceeded; no pass published")
    components = defaultdict(list)
    for node in nodes:
        components[find(node)].append(node)
    conflicts, linked = [], []
    rank = dict(train=0, val=1, test=2)
    for members in components.values():
        if len(members) < 2:
            continue
        splits = {nodes[node]["split"] for node in members}
        target = max(splits, key=rank.__getitem__)
        record = dict(
            members=[
                dict(
                    inventory=name,
                    group=group,
                    split=nodes[(name, group)]["split"],
                    records=nodes[(name, group)]["records"],
                )
                for name, group in sorted(members)
            ],
            required_split=target,
        )
        linked.append(record)
        if len(splits) > 1:
            conflicts.append(record)
    report = dict(
        kind="cross_corpus_media_group_audit",
        inputs=bindings,
        processor_sha256=sha256(__file__),
        phash_bands=[list(v) for v in PHASH_BANDS],
        phash_hamming_threshold=6,
        grouping_policy="whole connected groups; preserve holds, test precedence over val over train",
        input_images=sum(len(item["images"]) for item in data.values()),
        unique_rgb_images=len(pixels),
        input_groups=len(nodes),
        connected_groups=len(components),
        link_counts=dict(Counter(reason for _a, _b, reason in edges)),
        cross_inventory_link_counts=dict(Counter(reason for a, b, reason in edges if a[0] != b[0])),
        linked_components=linked,
        split_conflicts=conflicts,
        status="split_conflicts_require_partition_update"
        if conflicts
        else "rendered_visual_candidates_require_verification"
        if unresolved
        else "mechanical_group_checks_passed",
        rendered_text_policy="known complete OCR targets; casefolded normalized 5-character shingle Jaccard >=0.85; pHash does not join rendered text layouts",
        rendered_images=len(rendered),
        unresolved_rendered_visual_candidates=unresolved,
        max_text_comparisons=max_text_comparisons,
        formal_admission=False,
        partition_changes_applied=False,
        benchmark_exclusion_performed=False,
        limitations=[
            "pHash neighbors are conservative split groups, not confirmed semantic duplicates",
            "bound canonical raw-pixel audit, source quality and benchmark identity checks remain separate",
        ],
    )
    _write_new(output, report, max_bytes)
    return report


def close_media_candidate_review(audit_path, reviews, output, *, max_bytes=64 * 1024**2):
    """Close a fixed candidate list only with complete, bound nonduplicate decisions.

    This records an explicit model-assisted visual review, never human source
    quality or formal admission. Confirmed duplicates need a partition update.
    """
    audit_path, output = Path(audit_path), Path(output)
    if output.exists() or not reviews or max_bytes < 1 or audit_path.stat().st_size > max_bytes:
        raise ValueError("choose review records, a new output and a positive byte bound")
    parent_hash = sha256(audit_path)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("kind") != "cross_corpus_media_group_audit"
        or audit.get("split_conflicts") != []
        or audit.get("status") != "rendered_visual_candidates_require_verification"
    ):
        raise ValueError("candidate review cannot clear unresolved connected split conflicts")
    candidates = audit["unresolved_rendered_visual_candidates"]
    if not candidates:
        raise ValueError("candidate review has no fixed candidates")
    checked, references = set(), []
    for path in reviews:
        path = Path(path)
        if path.stat().st_size > max_bytes:
            raise ValueError("visual review exceeds the metadata byte bound")
        review = json.loads(path.read_text())
        if (
            review.get("kind") != "model_assisted_visual_candidate_review"
            or review.get("group_audit_sha256") != parent_hash
            or not review.get("reviewer")
        ):
            raise ValueError("visual review is not bound to this audit and reviewer")
        for decision in review["decisions"]:
            index = decision["candidate_index"]
            if type(index) is not int or not 0 <= index < len(candidates) or index in checked:
                raise ValueError("visual candidate review has a duplicate or invalid index")
            candidate = candidates[index]
            expected = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
            if (
                decision.get("candidate_sha256") != expected
                or decision.get("rendered_rgb_sha256") != candidate["rendered"]["rgb_sha256"]
                or decision.get("external_rgb_sha256") != candidate["external"]["rgb_sha256"]
                or decision.get("disposition") != "not_visual_duplicate"
                or not decision.get("reason")
                or any(
                    not isinstance(decision.get(k), str)
                    or re.fullmatch("[0-9a-f]{64}", decision[k]) is None
                    for k in ("rendered_file_sha256", "external_raw_sha256")
                )
            ):
                raise ValueError(
                    "candidate identity/decision is unresolved or needs partition repair"
                )
            checked.add(index)
        references.append(
            dict(
                sha256=sha256(path), reviewer=review["reviewer"], decisions=len(review["decisions"])
            )
        )
    if len(checked) != len(candidates):
        raise ValueError("visual candidate review is incomplete")
    result = dict(
        audit,
        status="mechanical_group_checks_passed_with_model_assisted_review",
        parent_group_audit_sha256=parent_hash,
        visual_candidate_reviews=references,
        reviewed_visual_candidates=len(checked),
        unresolved_rendered_visual_candidates=[],
        original_visual_candidates_retained_in_parent=True,
        review_processor_sha256=sha256(__file__),
        human_source_quality_review_completed=False,
        formal_admission=False,
    )
    _write_new(output, result, max_bytes)
    return result
