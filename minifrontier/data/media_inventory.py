"""Small immutable media-identity exports for grouping corpora held on different hosts."""

import contextlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minifrontier.data import sha256
from minifrontier.data.media_hash import PHASH_BANDS
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.storage import GIB, reserve_write

FORMAT = "media-group-identities-v1"


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
    images: dict[str, dict[str, str]] = {}
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
                value = dict(group=group, phash=media["phash"])
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
        format=FORMAT,
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


def audit_media_identities(inventories, output, *, max_edges=100_000, max_bytes=64 * 1024**2):
    """Group exact pixels, origin aliases and <=6-bit pHash neighbors across shards."""
    if len(inventories) < 2 or max_edges < 1 or Path(output).exists():
        raise ValueError("choose at least two inventories, a positive edge cap and a new output")
    data, bindings, nodes = {}, {}, {}
    for name, path in inventories.items():
        path = Path(path)
        if not isinstance(name, str) or not name or path.stat().st_size > max_bytes:
            raise ValueError("invalid media inventory label or byte bound")
        item = json.loads(path.read_text())
        if (
            item["format"] != FORMAT
            or item["image_count"] != len(item["images"])
            or item["group_count"] != len(item["groups"])
        ):
            raise ValueError("media inventory format or counts differ")
        for group, record in item["groups"].items():
            if record["split"] not in {"train", "val", "test"}:
                raise ValueError("media inventory has an unknown split")
            nodes[(name, group)] = record
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
    bands: dict[tuple[int, int], set[tuple[int, tuple[str, str]]]] = defaultdict(set)
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
            code = int(image["phash"], 16)
            candidates = set()
            for band, (offset, width) in enumerate(PHASH_BANDS):
                candidates.update(bands[(band, (code >> offset) & ((1 << width) - 1))])
            for other_code, other in candidates:
                if (code ^ other_code).bit_count() <= 6:
                    connect(node, other, "phash_hamming_le_6")
            for band, (offset, width) in enumerate(PHASH_BANDS):
                bands[(band, (code >> offset) & ((1 << width) - 1))].add((code, node))
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
        else "mechanical_group_checks_passed",
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
