"""Compose audited native media shards around one unchanged shared text encoding."""

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from minifrontier.chat_controls import update_manifest
from minifrontier.data import sha256
from minifrontier.data.encoding_audit import _checked
from minifrontier.data.media_cache import validate_policy
from minifrontier.data.minifrontier1 import write_json
from minifrontier.data.native import _check_text_origin, _shared_text_partitions, processor_identity
from minifrontier.storage import require_space

PASSED = "mechanical_checks_passed_pending_quality_admission"
POLICY_KEYS = (
    "family",
    "max_features",
    "min_pixels",
    "model_vocab_size",
    "sequence_length",
    "native_processor_sha256",
    "native_pretrain_encoding",
    "native_pretrain_objective",
)
COUNT_KEYS = (
    "examples",
    "supervised_tokens",
    "image_occurrences",
    "video_examples",
    "frames",
    "image_features",
)
MAP_KEYS = ("domain_ce", "rejected", "chat_template_counts")


def _encoding_proof(root):
    parent = json.loads((root / "source-audit.json").read_text())
    proof = json.loads(
        _checked(root, parent["integrity_report"], parent["integrity_report_sha256"]).read_text()
    )
    checksum = sha256(root / "manifest.json")
    if (
        parent.get("producer_finished") is not True
        or parent.get("status") != PASSED
        or proof.get("status") != PASSED
        or proof.get("errors")
        or parent.get("manifest_sha256") != checksum
        or proof.get("encoded_manifest_sha256", proof.get("manifest_sha256")) != checksum
    ):
        raise ValueError("rebind requires completed immutable encoding evidence")
    return parent


def rebind_native_components(components, text_encoding, output, *, max_gib=1, media_access=None):
    """Rebind unchanged media token sequences to a new shared-text inventory.

    Original train groups whose OCR origins are no longer train are excluded,
    never relabelled as validation. Conflicting old val/test origins stop before
    creating output. This operation preserves processor/resolution policies and
    does not qualify the result for a higher-resolution training phase.
    """
    roots = [Path(p).resolve() for p in components]
    shared, output = Path(text_encoding).resolve(), Path(output).resolve()
    if not roots or len(set(roots)) != len(roots) or output.exists() or max_gib <= 0:
        raise ValueError("choose distinct components, a new output and a positive write budget")
    maximum = int(max_gib * 1024**3)
    text = json.loads((shared / "manifest.json").read_text())
    if text.get("format") != "document-ragged-v2":
        raise ValueError("new native text must retain the document encoding format")
    _encoding_proof(shared)
    text_hash = sha256(shared / "manifest.json")
    partitions = _shared_text_partitions(shared, text_hash)
    manifests: list[dict[str, Any]] = []
    excluded, conflicts = set(), []
    access = {Path(p).resolve(): policy for p, policy in (media_access or {}).items()}
    if access.keys() - set(roots):
        raise ValueError("media policy names a component outside the rebind")

    def rows(root, child, stage, split):
        media = child["stages"][stage][split]["media"]
        path = _checked(root, media["file"], media["sha256"])
        index = np.load(_checked(root, media["index_file"], media["index_sha256"]))
        if index.dtype != np.int64 or index.shape != (media["examples"], 3):
            raise ValueError("native parent media index shape differs")
        with path.open("rb") as stream:
            for offset, size, length in index:
                if stream.tell() != int(offset):
                    raise ValueError("native parent media index has gaps")
                raw = stream.read(int(size))
                row = json.loads(raw)
                if (
                    len(raw) != size
                    or len(row["expected_ids"]) != length
                    or len(row["expected_labels"]) != length
                ):
                    raise ValueError("native parent media/index length differs")
                yield raw, row, int(length)
            if stream.read(1):
                raise ValueError("native parent has unindexed media records")

    for root in roots:
        child, _ = _component(root, manifests[0] if manifests else None)
        _encoding_proof(root)
        if (
            child["tokenizer"] != text["tokenizer"]
            or sha256(shared / "tokenizer.json") != child["tokenizer"]["sha256"]
        ):
            raise ValueError("shared-text rebind cannot change the frozen tokenizer")
        for stage in ("pretrain", "sft"):
            for split in ("train", "val", "test"):
                for _raw, row, _length in rows(root, child, stage, split):
                    origin = row["record"].get("text_origin")
                    if origin and partitions.get(origin.get("sample_id")) != ("pretrain", split):
                        if split != "train":
                            conflicts.append(
                                dict(
                                    component=str(root),
                                    sample_id=row["record"]["sample_id"],
                                    origin=origin,
                                )
                            )
                        else:
                            excluded.add(row["split_group"])
        manifests.append(child)
    if conflicts:
        raise ValueError(
            "old validation/test text origins conflict with new text inventory; explicit handling required: "
            + json.dumps(conflicts[:10])
        )
    require_space(output, min(maximum, 16 * 1024**2))
    output.mkdir(parents=True)
    selection = dict(
        new_shared_text_manifest_sha256=text_hash,
        excluded_train_groups=sorted(excluded),
        policy="exclude complete old train groups; preserve all old val/test records",
    )
    if len(json.dumps(selection).encode()) + 1024**2 > maximum:
        raise ValueError("text-origin selection exceeds the rebind metadata budget")
    write_json(output / "selection.json", selection)
    selection_hash = sha256(output / "selection.json")
    children, child_access, used, reports = [], {}, 0, []
    for number, (root, old) in enumerate(zip(roots, manifests, strict=True)):
        target = output / "components" / f"{number:03d}"
        target.mkdir(parents=True)
        os.link(root / "tokenizer.json", target / "tokenizer.json")
        child = json.loads(json.dumps(old))
        child.update(
            formal_admission=False,
            main_budget_eligible=False,
            text_source=dict(path=os.path.relpath(shared, target), manifest_sha256=text_hash),
        )
        removed = []
        for stage in ("pretrain", "sft"):
            for split in ("train", "val", "test"):
                node = child["stages"][stage][split]
                media = node["media"]
                kept: Counter[str] = Counter()
                domains: Counter[str] = Counter()
                indexes = []
                drop = split == "train" and any(
                    row["split_group"] in excluded
                    for _raw, row, _n in rows(root, old, stage, split)
                )
                target_file = target / media["file"]
                writer = target_file.open("wb") if drop else None
                try:
                    for raw, row, length in rows(root, old, stage, split):
                        record = row["record"]
                        if split == "train" and row["split_group"] in excluded:
                            removed.append(record["sample_id"])
                            continue
                        if record.get("text_origin"):
                            _check_text_origin(record, split, partitions)
                        if writer is not None:
                            if used + len(raw) + 24 > maximum:
                                raise ValueError("rebound native media exceeded new-file budget")
                            indexes.append((writer.tell(), len(raw), length))
                            writer.write(raw)
                            used += len(raw) + 24
                        resources = record["media"]
                        videos = [m for m in resources if m.get("kind") == "video"]
                        ce = sum(t != -100 for t in row["expected_labels"][1:])
                        kept.update(
                            examples=1,
                            supervised_tokens=ce,
                            image_occurrences=len(resources) - len(videos),
                            video_examples=len(videos),
                            frames=sum(len(v["frames"]) for v in videos),
                            image_features=row["expected_ids"].count(7),
                        )
                        domains[record["task"]] += ce
                finally:
                    if writer is not None:
                        writer.close()
                if drop:
                    np.save(
                        target / media["index_file"],
                        np.asarray(indexes, dtype=np.int64).reshape(-1, 3),
                    )
                    dropped = media["examples"] - kept["examples"]
                    media["rejected"] = dict(media["rejected"], shared_text_origin_conflict=dropped)
                    media.update(
                        {key: kept[key] for key in COUNT_KEYS},
                        domain_ce=dict(domains),
                        sha256=sha256(target_file),
                        index_sha256=sha256(target / media["index_file"]),
                    )
                else:
                    if (
                        any(kept[key] != media[key] for key in COUNT_KEYS)
                        or domains != media["domain_ce"]
                    ):
                        raise ValueError("native media counters differ from encoded parent")
                    os.link(root / media["file"], target_file)
                    os.link(root / media["index_file"], target / media["index_file"])
                node.update(
                    text=text["stages"][stage][split],
                    examples=text["stages"][stage][split]["examples"] + media["examples"],
                    supervised_tokens=text["stages"][stage][split]["supervised_tokens"]
                    + media["supervised_tokens"],
                )
        child["text_binding_derivation"] = dict(
            parent=os.path.relpath(root, target),
            parent_manifest_sha256=sha256(root / "manifest.json"),
            shared_text_manifest_sha256=text_hash,
            exclusion_selection_file="../../selection.json",
            exclusion_selection_sha256=selection_hash,
            removed_media_sample_ids=removed,
            retained_media_bytes_unchanged=True,
            processor_and_resolution_unchanged=True,
        )
        write_json(target / "manifest.json", child)
        proof = dict(
            kind="native_shared_text_binding_derivation",
            status=PASSED,
            encoded_manifest_sha256=sha256(target / "manifest.json"),
            parent_source_audit_sha256=sha256(root / "source-audit.json"),
            new_text_source_audit_sha256=sha256(shared / "source-audit.json"),
            source_corpus_manifest_sha256=old["native_corpus_sha256"],
            **child["text_binding_derivation"],
            errors=[],
            formal_admission=False,
            scope="bound parent file hashes; unchanged retained media bytes and val/test splits; whole train-group exclusions; new shared text identities; no retokenization, pixel processing or training",
            completed_unix=time.time(),
        )
        write_json(target / "encoding-audit.json", proof)
        write_json(
            target / "source-audit.json",
            dict(
                kind="encoded_media_component",
                status=PASSED,
                formal_admission=False,
                main_budget_eligible=False,
                producer_finished=True,
                manifest_sha256=proof["encoded_manifest_sha256"],
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(target / "encoding-audit.json"),
            ),
        )
        children.append(target)
        if root in access:
            child_access[target] = access[root]
        reports.append(dict(parent=str(root), output=str(target), removed_records=len(removed)))
    result = assemble_native_components(children, output / "dataset", media_access=child_access)
    report = dict(
        kind="native_media_shared_text_rebind",
        dataset=str(output / "dataset"),
        manifest_sha256=sha256(output / "dataset/manifest.json"),
        components=reports,
        excluded_train_groups=sorted(excluded),
        payload_bytes_written=used,
        full_next_phase_ready=False,
        processor_and_resolution_unchanged=True,
        formal_admission=False,
        status=PASSED,
    )
    write_json(output / "rebind-report.json", report)
    written_bytes = sum(
        p.stat().st_size for p in output.rglob("*") if p.is_file() and p.stat().st_nlink == 1
    )
    if written_bytes > maximum:
        raise ValueError("rebind payload and metadata exceeded the declared new-file budget")
    return dict(report, manifest=result)


def _component(root, baseline=None):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("format") != "hybrid-native-v2"
        or manifest.get("kind") == "canonical_native_composition"
        or not manifest.get("text_source")
    ):
        raise ValueError("native composition requires standalone shared-text media encodings")
    if manifest["native_processor_sha256"] != processor_identity(manifest["family"]):
        raise ValueError("native component processor changed")
    if sha256(root / "tokenizer.json") != manifest["tokenizer"]["sha256"]:
        raise ValueError("native component tokenizer changed")
    if baseline is not None and (
        any(manifest.get(key) != baseline.get(key) for key in POLICY_KEYS)
        or manifest["tokenizer"] != baseline["tokenizer"]
        or manifest["text_source"]["manifest_sha256"] != baseline["text_source"]["manifest_sha256"]
    ):
        raise ValueError("native component model, tokenizer or shared text differs")
    shared = (root / manifest["text_source"]["path"]).resolve()
    if sha256(shared / "manifest.json") != manifest["text_source"]["manifest_sha256"]:
        raise ValueError("native component shared text manifest changed")
    source = json.loads((shared / "manifest.json").read_text())
    if source["format"] != "document-ragged-v2":
        raise ValueError("native composition needs standalone document text")
    for stage, splits in source["stages"].items():
        for split, text in splits.items():
            if manifest["stages"][stage][split]["text"] != text:
                raise ValueError("native component text split differs from its shared source")
    return manifest, shared


def read_components(root, manifest):
    """Resolve immutable component bindings for the existing native training reader."""
    components = []
    seen = set()
    shared = (root / manifest["text_source"]["path"]).resolve()
    for ref in manifest["components"]:
        path = (root / ref["path"]).resolve()
        if path in seen or sha256(path / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("native component manifest changed or is repeated")
        seen.add(path)
        child, child_text = _component(path, manifest)
        if child_text != shared:
            raise ValueError("native components must reuse one shared text directory")
        if "media_access" in ref:
            policy = validate_policy(ref["media_access"])
            child = dict(
                child,
                media_access=dict(policy, cache_dir=str((root / policy["cache_dir"]).resolve())),
            )
        components.append((path, child))
    if not components:
        raise ValueError("native composition has no media components")
    return components


def assemble_native_components(components, output, *, max_bytes=64 * 1024**2, media_access=None):
    """Check source identities and counts; write only a tokenizer and bound manifest.

    Child pixel/token audits remain bound. Cross-corpus near-duplicate reviews,
    source quality and phase supply admission remain separate requirements.
    """
    roots, output = [Path(p).resolve() for p in components], Path(output).resolve()
    if not roots or len(set(roots)) != len(roots) or output.exists() or max_bytes <= 0:
        raise ValueError("choose distinct native components and a bounded new output")
    access = {Path(k).resolve(): validate_policy(v) for k, v in (media_access or {}).items()}
    if access.keys() - set(roots):
        raise ValueError("media access names a component outside this composition")
    manifests: list[dict[str, Any]] = []
    references = []
    first_shared = None
    for root in roots:
        manifest, shared = _component(root, manifests[0] if manifests else None)
        if first_shared is not None and shared != first_shared:
            raise ValueError("native components must reuse one shared text directory")
        first_shared = shared
        parent = json.loads((root / "source-audit.json").read_text())
        proof = json.loads(
            _checked(
                root, parent["integrity_report"], parent["integrity_report_sha256"]
            ).read_text()
        )
        checksum = sha256(root / "manifest.json")
        if (
            parent.get("producer_finished") is not True
            or parent["status"] != PASSED
            or proof["status"] != PASSED
            or proof.get("errors")
            or parent["manifest_sha256"] != checksum
            or proof.get("encoded_manifest_sha256", proof.get("manifest_sha256")) != checksum
        ):
            raise ValueError("native component needs its completed bound encoding audit")
        references.append(
            dict(
                path=os.path.relpath(root, output),
                manifest_sha256=checksum,
                source_audit_sha256=sha256(root / "source-audit.json"),
                encoding_audit_sha256=parent["integrity_report_sha256"],
                corpus_manifest_sha256=manifest["native_corpus_sha256"],
            )
        )
        if root in access:
            references[-1]["media_access"] = access[root]
        manifests.append(manifest)
    baseline = manifests[0]
    partitions = _shared_text_partitions(shared, baseline["text_source"]["manifest_sha256"])
    samples = set(partitions)
    groups: dict[str, str] = {}
    source_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    pixels: dict[str, str] = {}
    stages: dict[str, dict[str, Any]] = {}
    for stage in ("pretrain", "sft"):
        stages[stage] = {}
        for split in ("train", "val", "test"):
            counts: Counter[str] = Counter()
            maps: dict[str, Counter[str]] = {key: Counter() for key in MAP_KEYS}
            unique = set()
            for root, child in zip(roots, manifests, strict=True):
                media = child["stages"][stage][split]["media"]
                path = _checked(root, media["file"], media["sha256"])
                index = np.load(_checked(root, media["index_file"], media["index_sha256"]))
                if index.dtype != np.int64 or index.shape != (media["examples"], 3):
                    raise ValueError("native component index shape/count differs")
                ce = inputs = 0
                domains: Counter[str] = Counter()
                with path.open("rb") as stream:
                    for offset, size, length in index:
                        if stream.tell() != offset or size <= 0 or length < 2:
                            raise ValueError("native component index has gaps or invalid lengths")
                        raw = stream.read(int(size))
                        row = json.loads(raw)
                        ids, labels = row["expected_ids"], row["expected_labels"]
                        if len(raw) != size or len(ids) != length or len(labels) != length:
                            raise ValueError("native component token/index lengths differ")
                        record, group = row["record"], row["split_group"]
                        source_groups[record["source"]][split].add(group)
                        identity = record["sample_id"]
                        if identity in samples:
                            raise ValueError("duplicate sample in native composition")
                        samples.add(identity)
                        if record["stage"] != stage or groups.setdefault(group, split) != split:
                            raise ValueError("native sample stage or connected split differs")
                        if "text_origin" in record:
                            _check_text_origin(record, split, partitions)
                        for resource in record["media"]:
                            identities = resource.get("frame_rgb_sha256") or [
                                resource.get("rgb_sha256", resource.get("sha256"))
                            ]
                            for key in identities:
                                if not key or pixels.setdefault(key, split) != split:
                                    raise ValueError("native media identity crosses splits")
                                unique.add(key)
                        count = sum(value != -100 for value in labels[1:])
                        ce += count
                        inputs += int(length)
                        domains[record["task"]] += count
                    if stream.read(1):
                        raise ValueError("native component index omits trailing records")
                if ce != media["supervised_tokens"] or domains != media["domain_ce"]:
                    raise ValueError("native component CE/domain counts differ")
                counts.update({key: media[key] for key in COUNT_KEYS})
                counts["input_tokens"] += inputs
                for key in MAP_KEYS:
                    maps[key].update(media[key])
            text = baseline["stages"][stage][split]["text"]
            stages[stage][split] = dict(
                format="hybrid-native-v2",
                text=text,
                media=dict(counts, **{key: dict(value) for key, value in maps.items()}),
                media_sources=[dict(component=i, split=split) for i in range(len(roots))],
                examples=text["examples"] + counts["examples"],
                supervised_tokens=text["supervised_tokens"] + counts["supervised_tokens"],
                unique_rgb_images=len(unique),
            )
    result = dict(
        schema_version=2,
        format="hybrid-native-v2",
        kind="canonical_native_composition",
        composition_version=1,
        **{key: baseline[key] for key in POLICY_KEYS},
        tokenizer=baseline["tokenizer"],
        text_source=dict(
            path=os.path.relpath(shared, output),
            manifest_sha256=baseline["text_source"]["manifest_sha256"],
        ),
        stages=stages,
        components=references,
        formal_admission=False,
        main_budget_eligible=False,
        shared_text_files_copied=0,
        media_shard_files_copied=0,
        raw_media_copied=False,
        sample_order="shared text once, then component argument order and original media order",
        identity_checks=dict(duplicate_samples=0, cross_split_groups=0, cross_split_rgb=0),
        source_group_splits={
            source: {
                "groups": {split: len(members[split]) for split in ("train", "val", "test")},
                "validation_group_fraction": len(members["val"])
                / sum(len(members[split]) for split in ("train", "val", "test")),
            }
            for source, members in sorted(source_groups.items())
        },
        remaining=[
            "bound cross-corpus near-duplicate checks",
            "source quality and phase admission",
        ],
        processor_sha256=sha256(__file__),
    )
    update_manifest(result)
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    size = (roots[0] / "tokenizer.json").stat().st_size + len(encoded)
    if size > max_bytes:
        raise ValueError("native composition metadata exceeds its disk budget")
    require_space(output, size)
    output.mkdir(parents=True)
    shutil.copyfile(roots[0] / "tokenizer.json", output / "tokenizer.json")
    (output / "manifest.json").write_bytes(encoded)
    return result
