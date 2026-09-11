"""Apply audited corpus refinements while preserving existing encoded data formats."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from minifrontier.data import sha256
from minifrontier.data.encoding_audit import _checked, _compact_records
from minifrontier.data.minifrontier1 import digest, write_json
from minifrontier.data.minifrontier1_encoding import FORMAT, _encode_compact_items
from minifrontier.data.native import processor_identity
from minifrontier.data.partitions import corpus_storage_root, open_corpus
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.multimodal import prepare_record
from minifrontier.storage import GIB, reserve_write

PASSED = "mechanical_checks_passed_pending_quality_admission"


def _inputs(corpus, encoded, *, reclassify=False):
    canonical = json.loads((corpus / "corpus-manifest.json").read_text())
    source_audit = json.loads((corpus / "source-audit.json").read_text())
    manifest = json.loads((encoded / "manifest.json").read_text())
    if manifest.get("kind") == "canonical_native_composition":
        raise ValueError("filter native media components individually, then reassemble")
    parent = json.loads((encoded / "source-audit.json").read_text())
    # Earlier canonical image producers bind the same fixed report by this key.
    # Both layouts require a recorded hash; an unbound file is never sufficient.
    if "integrity_report" in parent:
        report_name, report_hash = parent["integrity_report"], parent.get("integrity_report_sha256")
    else:
        report_name, report_hash = "encoding-audit.json", parent.get("encoding_audit_sha256")
    if not report_hash:
        raise ValueError("parent encoding has no bound integrity report")
    proof_path = _checked(encoded, report_name, report_hash)
    proof = json.loads(proof_path.read_text())
    parent_hash = sha256(encoded / "manifest.json")
    if (
        parent.get("producer_finished") is not True
        or parent["status"] != PASSED
        or proof["status"] != PASSED
        or proof.get("errors")
        or parent["manifest_sha256"] != parent_hash
        or proof.get("encoded_manifest_sha256", proof.get("manifest_sha256")) != parent_hash
        or source_audit["operation"]
        not in (
            {"correct_source_task_classification"}
            if reclassify
            else {"exclude_cross_pool_train_groups", "exclude_source_quality_train_groups"}
        )
    ):
        raise ValueError(
            "filtering requires a completed audited parent and the declared refinement"
        )
    corpus_key = (
        "source_corpus_manifest_sha256" if manifest["format"] == FORMAT else "native_corpus_sha256"
    )
    if canonical["previous_effective_manifest_sha256"] != manifest[corpus_key]:
        raise ValueError("encoded parent is not the previous effective corpus")
    groups = {} if reclassify else source_audit["excluded_training_groups"]
    if not groups and not reclassify:
        raise ValueError("no newly excluded training groups")
    task_changes = None
    with contextlib.closing(open_corpus(corpus)) as db:
        if sha256(corpus_storage_root(db) / "corpus.sqlite") != canonical["database_sha256"]:
            raise ValueError("encoded exclusion corpus database changed")
        remaining = dict(
            (identity, (stage, split, group))
            for identity, stage, split, group in db.execute(
                "SELECT id,stage,split,group_root FROM samples"
            )
        )
        if reclassify:
            task_changes = dict(
                (identity, (old_task, new_task))
                for identity, old_task, new_task in db.execute(
                    "SELECT s.id,m.task,s.task FROM samples s JOIN main.samples m ON s.id=m.id"
                )
            )
            if task_changes.keys() != remaining.keys():
                raise ValueError("task correction has incomplete canonical membership")
        removed = {}
        for group, count in groups.items():
            rows = list(
                db.execute(
                    "SELECT id,stage,split,payload FROM main.samples WHERE group_root=?", (group,)
                )
            )
            if len(rows) != count or any(split != "train" for _id, _stage, split, _payload in rows):
                raise ValueError("excluded original group differs from the checked partition")
            for identity, stage, _split, payload in rows:
                row = json.loads(payload)
                if canonical.get("task_policy") is not None:
                    from minifrontier.data.media_tasks import effective_task

                    # Excluded rows are absent from the effective SQL view. Apply its
                    # already-validated task policy to their immutable source payloads.
                    row["task"] = effective_task(
                        row["source"], row.get("revision"), row["task"], row.get("visual_question")
                    )
                removed[identity] = (stage, group, row)
        if set(removed) & remaining.keys():
            raise ValueError("excluded record is still present in the effective corpus")
        if any(stage != "pretrain" for stage, _group, _row in removed.values()):
            raise ValueError("this filter supports pretraining media exclusions only")
    bindings = dict(
        parent_manifest_sha256=parent_hash,
        parent_source_audit_sha256=sha256(encoded / "source-audit.json"),
        parent_integrity_sha256=sha256(proof_path),
        corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        corpus_source_audit_sha256=sha256(corpus / "source-audit.json"),
        processor_sha256=sha256(__file__),
        processor_dependencies={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "encoding_audit.py",
                "minifrontier1_encoding.py",
                "native.py",
                "partitions.py",
                "media_tasks.py",
            )
        },
    )
    if reclassify:
        bindings["task_policy"] = canonical["task_policy"]
    else:
        evidence_key = (
            "quality_review_sha256"
            if source_audit["operation"] == "exclude_source_quality_train_groups"
            else "grouping_audit_sha256"
        )
        bindings[evidence_key] = canonical[evidence_key]
    return manifest, proof, remaining, removed, bindings, task_changes


def _difference(original, removed, actual):
    expected = Counter(original)
    expected.subtract(removed)
    if any(value < 0 for value in expected.values()) or +expected != +Counter(actual):
        raise ValueError("filtered counters do not equal parent minus complete excluded records")


def _record_digest(hasher, domain, ids, mask, metadata):
    raw = json.dumps(
        dict(domain=domain, metadata=metadata), ensure_ascii=False, sort_keys=True
    ).encode()
    hasher.update(len(ids).to_bytes(8, "little") + len(raw).to_bytes(8, "little") + raw)
    hasher.update(np.asarray(ids, dtype="<u4").tobytes())
    hasher.update(np.asarray(mask, dtype=np.uint8).tobytes())


def _filter_mf1(corpus, encoded, output, config, maximum, old, remaining, removed, task_changes):
    if old["kind"] != "canonical_image_component" or old["config_sha256"] != digest(asdict(config)):
        raise ValueError("MF1 filtering requires the original canonical-image model configuration")
    _checked(encoded, "tokenizer.json", old["tokenizer_sha256"])
    base = dict(
        old,
        source_corpus_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        source_audit_sha256=sha256(corpus / "source-audit.json"),
        complete_record_length_buckets={},
    )
    expected_hashes, omitted, domain_adjustments = {}, {}, {}
    removed_seen = set()

    def records(split):
        hasher = hashlib.sha256()
        excluded: Counter[str] = Counter()
        excluded_domains: Counter[str] = Counter()
        redistributed: Counter[str] = Counter()
        buckets: dict[str, dict[str, Counter[str]]] = {}
        for identity, domain, origin, group, ids, mask, metadata in _compact_records(
            encoded, old, split, images=True
        ):
            delta = dict(
                records=1,
                input_tokens=len(ids),
                ce_tokens=int(mask[1:].sum()),
                media_exposures=len(metadata["resources"]),
                vision_tokens=sum(s["feature_count"] for s in metadata["media"]),
            )
            if identity in removed:
                if split != "train" or removed[identity][1] != group or identity in removed_seen:
                    raise ValueError("excluded compact record identity/group differs")
                removed_seen.add(identity)
                excluded.update(delta)
                excluded_domains[domain] += delta["ce_tokens"]
                continue
            if remaining.pop(identity, None) != ("pretrain", split, group):
                raise ValueError("kept compact record is duplicated or in the wrong partition")
            if task_changes is not None:
                before, after = task_changes[identity]
                if domain != before:
                    raise ValueError("parent compact task differs from the canonical original")
                redistributed[before] -= delta["ce_tokens"]
                redistributed[after] += delta["ce_tokens"]
                domain = after
            _record_digest(hasher, domain, ids, mask, metadata)
            bucket = str(
                next((n for n in (512, 1024, 2048, 4096, 8192) if len(ids) <= n), "over_8192")
            )
            buckets.setdefault(domain, {}).setdefault(bucket, Counter()).update(delta)
            x = torch.from_numpy(np.asarray(ids, dtype=np.int64).copy())[None]
            labels = x.clone().masked_fill(~torch.from_numpy(mask)[None], -100)
            yield (
                dict(media=metadata["resources"], origin=origin),
                dict(
                    input_ids=x,
                    labels=labels,
                    media=metadata["media"],
                    sample_id=identity,
                    split_group=group,
                    domain=domain,
                    media_exposures=delta["media_exposures"],
                ),
            )
        expected_hashes[split] = hasher.hexdigest()
        omitted[split] = dict(counts=dict(excluded), domain_ce=dict(excluded_domains))
        domain_adjustments[split] = dict(redistributed)
        base["complete_record_length_buckets"][split] = buckets

    result = _encode_compact_items(
        ((split, records(split)) for split in ("train", "val", "test")),
        output,
        config,
        encoded / "tokenizer.json",
        base,
        source_manifest_sha256=sha256(corpus / "corpus-manifest.json"),
        media_root=old["media_root"],
        max_gib=maximum,
        shard_tokens=old["shard_tokens"],
        domain_order=old["domains"],
    )
    if remaining or removed_seen != removed.keys():
        raise ValueError("compact filtering did not cover every retained and excluded record")
    for split in ("train", "val", "test"):
        _difference(
            old["splits"][split]["counts"],
            omitted[split]["counts"],
            result["splits"][split]["counts"],
        )
        _difference(
            old["splits"][split]["domain_ce"],
            {
                key: omitted[split]["domain_ce"].get(key, 0) - domain_adjustments[split].get(key, 0)
                for key in set(omitted[split]["domain_ce"]) | set(domain_adjustments[split])
            },
            result["splits"][split]["domain_ce"],
        )
        hasher = hashlib.sha256()
        for _id, domain, _origin, _group, ids, mask, metadata in _compact_records(
            output, result, split, images=True
        ):
            _record_digest(hasher, domain, ids, mask, metadata)
        if hasher.hexdigest() != expected_hashes[split]:
            raise ValueError("compact repacking changed kept tokens, labels, order or media spans")
        if (
            task_changes is None
            and split != "train"
            and result["splits"][split] != old["splits"][split]
        ):
            raise ValueError("compact exclusion changed validation/test payload bytes")
    return result, dict(
        splits=result["splits"],
        removed=omitted,
        retained_record_sha256=expected_hashes,
        sealed_holdout_payloads_unchanged=task_changes is None,
        sealed_holdout_contents_and_membership_unchanged=True,
        domain_ce_adjustments=domain_adjustments,
    )


def _filter_native(
    corpus, encoded, output, maximum, old, parent_proof, remaining, removed, task_changes
):
    if old["format"] != "hybrid-native-v2" or not old.get("text_source"):
        raise ValueError("native filtering requires a canonical shared-text encoding")
    if old["native_processor_sha256"] != processor_identity(old["family"]):
        raise ValueError("native processor changed since parent encoding")
    tokenizer = Tokenizer.from_file(
        str(_checked(encoded, "tokenizer.json", old["tokenizer"]["sha256"]))
    )
    shared = (encoded / old["text_source"]["path"]).resolve()
    if sha256(shared / "manifest.json") != old["text_source"]["manifest_sha256"]:
        raise ValueError("shared text manifest changed")

    def prepare(row):
        return prepare_record(
            row,
            tokenizer,
            old["family"],
            root=old["media_root"],
            max_features=old["max_features"],
            min_pixels=old.get("min_pixels"),
            model_vocab_size=old["model_vocab_size"],
        )

    removed_items = {identity: prepare(row) for identity, (_stage, _group, row) in removed.items()}
    output.mkdir()
    shutil.copyfile(encoded / "tokenizer.json", output / "tokenizer.json")
    used = (output / "tokenizer.json").stat().st_size
    result = json.loads(json.dumps(old))
    result["text_source"]["path"] = os.path.relpath(shared, output)
    result["native_corpus_sha256"] = sha256(corpus / "corpus-manifest.json")
    splits, omitted, seen, domain_adjustments = {}, {}, set(), {}
    for stage in ("pretrain", "sft"):
        for split in ("train", "val", "test"):
            node = result["stages"][stage][split]
            media = node["media"]
            path = _checked(encoded, media["file"], media["sha256"])
            index = np.load(_checked(encoded, media["index_file"], media["index_sha256"]))
            indexes = []
            kept: Counter[str] = Counter()
            drop: Counter[str] = Counter()
            domains: Counter[str] = Counter()
            redistributed: Counter[str] = Counter()
            expected = hashlib.sha256()
            with path.open("rb") as source, (output / media["file"]).open("wb") as target:
                for offset, size, length in index:
                    if source.tell() != int(offset):
                        raise ValueError("native parent index has gaps or overlaps")
                    raw = source.read(int(size))
                    row = json.loads(raw)
                    identity = row["record"]["sample_id"]
                    if len(raw) != int(size) or len(row["expected_ids"]) != int(length):
                        raise ValueError("native parent index length differs")
                    if identity in removed:
                        item = removed_items[identity]
                        if (
                            identity in seen
                            or split != "train"
                            or (stage, row["split_group"], row["record"]) != removed[identity]
                        ):
                            raise ValueError(
                                "excluded native record differs from the canonical original"
                            )
                        if (
                            item.input_ids[0].tolist() != row["expected_ids"]
                            or item.labels[0].tolist() != row["expected_labels"]
                        ):
                            raise ValueError(
                                "excluded native item differs from the audited processor"
                            )
                        seen.add(identity)
                        count = int(item.labels[:, 1:].ne(-100).sum())
                        drop.update(
                            examples=1,
                            supervised_tokens=count,
                            input_tokens=int(length),
                            image_occurrences=item.image_count,
                            video_examples=item.video_count,
                            frames=item.frame_count,
                            image_features=item.image_features,
                        )
                        domains[row["record"]["task"]] += count
                        continue
                    if remaining.pop(identity, None) != (stage, split, row["split_group"]):
                        raise ValueError("kept native item is duplicated or in the wrong partition")
                    if task_changes is not None:
                        before, after = task_changes[identity]
                        if row["record"]["task"] != before:
                            raise ValueError(
                                "parent native task differs from the canonical original"
                            )
                        count = sum(t != -100 for t in row["expected_labels"][1:])
                        redistributed[before] -= count
                        redistributed[after] += count
                        if before != after:
                            row["record"]["task"] = after
                            raw = json.dumps(row, ensure_ascii=False).encode() + b"\n"
                    used += len(raw) + 24
                    if used > maximum * GIB:
                        raise ValueError("native filtering exceeds the declared encoded byte cap")
                    indexes.append((target.tell(), len(raw), int(length)))
                    target.write(raw)
                    expected.update(raw)
                    kept.update(
                        records=1,
                        input_tokens=int(length),
                        ce_tokens=sum(t != -100 for t in row["expected_labels"][1:]),
                    )
                if source.read(1):
                    raise ValueError("native parent has unindexed records")
            np.save(
                output / media["index_file"], np.asarray(indexes, dtype=np.int64).reshape(-1, 3)
            )
            for key in (
                "examples",
                "supervised_tokens",
                "image_occurrences",
                "video_examples",
                "frames",
                "image_features",
            ):
                media[key] -= drop[key]
                if media[key] < 0:
                    raise ValueError("native exclusion count exceeds its parent")
            domain_counts = Counter(media["domain_ce"])
            domain_counts.subtract(domains)
            domain_counts.update(redistributed)
            if any(v < 0 for v in domain_counts.values()):
                raise ValueError("native domain adjustment exceeds parent CE")
            media["domain_ce"] = dict(+domain_counts)
            excluded_overflow = sum(
                s == stage and split == "train" and item.input_ids.shape[1] > old["sequence_length"]
                for identity, item in removed_items.items()
                for s, _g, _r in [removed[identity]]
            )
            rejected = (
                media["rejected"].get("complete_media_answer_exceeds_bucket", 0) - excluded_overflow
            )
            if rejected < 0:
                raise ValueError("native excluded overflow exceeds its parent rejection count")
            media["rejected"] = (
                {"complete_media_answer_exceeds_bucket": rejected} if rejected else {}
            )
            if (
                kept["records"] != media["examples"]
                or kept["ce_tokens"] != media["supervised_tokens"]
            ):
                raise ValueError("native retained CE differs from parent minus exclusions")
            media["sha256"] = sha256(output / media["file"])
            media["index_sha256"] = sha256(output / media["index_file"])
            if expected.hexdigest() != media["sha256"]:
                raise ValueError("native filtering changed retained record bytes")
            if (
                task_changes is None
                and split != "train"
                and media != old["stages"][stage][split]["media"]
            ):
                raise ValueError("native filtering changed validation/test payloads")
            node["examples"] = node["text"]["examples"] + media["examples"]
            node["supervised_tokens"] = (
                node["text"]["supervised_tokens"] + media["supervised_tokens"]
            )
            key = f"{stage}.{split}"
            parent_counts = parent_proof["splits"][key]["counts"]
            _difference(
                {k: parent_counts.get(k, 0) for k in ("records", "input_tokens", "ce_tokens")},
                dict(
                    records=drop["examples"],
                    input_tokens=drop["input_tokens"],
                    ce_tokens=drop["supervised_tokens"],
                ),
                kept,
            )
            splits[key] = dict(
                counts={k: kept[k] for k in ("records", "input_tokens", "ce_tokens")},
                domain_ce=media["domain_ce"],
                rejected=media["rejected"],
            )
            omitted[key] = dict(
                counts=dict(drop), domain_ce=dict(domains), excluded_overflow=excluded_overflow
            )
            domain_adjustments[key] = dict(redistributed)
    if seen != {
        key
        for key, item in removed_items.items()
        if item.input_ids.shape[1] <= old["sequence_length"]
    }:
        raise ValueError("eligible excluded native records were not fully accounted")
    if task_changes is not None and remaining:
        raise ValueError("task-only repacking requires a complete parent without omitted records")
    observed_overflow: Counter[tuple[str, str]] = Counter()
    with contextlib.closing(open_corpus(corpus)) as db:
        for identity, (stage, split, _group) in remaining.items():
            payload = db.execute("SELECT payload FROM samples WHERE id=?", (identity,)).fetchone()[
                0
            ]
            if prepare(json.loads(payload)).input_ids.shape[1] <= old["sequence_length"]:
                raise ValueError("native filtering omitted an eligible canonical record")
            observed_overflow[(stage, split)] += 1
    for stage in ("pretrain", "sft"):
        for split in ("train", "val", "test"):
            if observed_overflow[(stage, split)] != result["stages"][stage][split]["media"][
                "rejected"
            ].get("complete_media_answer_exceeds_bucket", 0):
                raise ValueError("remaining native overflow count differs from the canonical view")
    result["max_encoded_gib"] = maximum
    return result, dict(
        splits=splits,
        removed=omitted,
        sealed_holdout_payloads_unchanged=task_changes is None,
        sealed_holdout_contents_and_membership_unchanged=True,
        domain_ce_adjustments=domain_adjustments,
        shared_text_payloads_unchanged=True,
    )


def filter_media_encoding(corpus, encoded, output, *, config=None, max_gib=1, reclassify=False):
    corpus, encoded, output = (
        Path(corpus).resolve(),
        Path(encoded).resolve(),
        Path(output).resolve(),
    )
    if output.exists() or max_gib <= 0:
        raise ValueError("choose a new filtered encoding and a positive byte cap")
    old, proof, remaining, removed, bindings, task_changes = _inputs(
        corpus, encoded, reclassify=reclassify
    )
    maximum = int(max_gib * GIB)
    metadata_reserve = 1024**2
    if (encoded / "tokenizer.json").stat().st_size + metadata_reserve > maximum:
        raise ValueError("filtered encoding byte cap cannot hold tokenizer and audit metadata")
    reserve = int(float(os.environ.get("MINIFRONTIER_MIN_FREE_GIB", "80")) * GIB)
    # No nested reserve_write occurs inside either repacker. Hold one bounded allocation.
    with reserve_write(output.with_suffix(".reservation"), maximum, reserve_bytes=reserve):
        try:
            if old["format"] == FORMAT:
                values = (
                    json.loads(Path(config).read_text())
                    if isinstance(config, (str, Path))
                    else config
                )
                if not isinstance(values, dict):
                    raise ValueError("MF1 filtering requires its original model configuration")
                result, details = _filter_mf1(
                    corpus,
                    encoded,
                    output,
                    MiniFrontier1Config(**values),
                    (maximum - metadata_reserve) / GIB,
                    old,
                    remaining,
                    removed,
                    task_changes,
                )
            else:
                result, details = _filter_native(
                    corpus,
                    encoded,
                    output,
                    (maximum - metadata_reserve) / GIB,
                    old,
                    proof,
                    remaining,
                    removed,
                    task_changes,
                )
                result["max_encoded_gib"] = max_gib
            result.update(formal_admission=False, main_budget_eligible=False)
            result["task_derivation" if reclassify else "exclusion_derivation"] = dict(
                bindings, parent=os.path.relpath(encoded, output)
            )
            write_json(output / "manifest.json", result)
            report = dict(
                kind="encoded_media_task_derivation"
                if reclassify
                else "encoded_media_exclusion_derivation",
                status=PASSED,
                formal_admission=False,
                encoded_manifest_sha256=sha256(output / "manifest.json"),
                **bindings,
                **details,
                raw_media_copied=False,
                retained_encoded_pixels_redecoded=False,
                model_forward_executed=False,
                scope=(
                    "all parent encoded file hashes, identical tokens/masks/media/order and split membership, corrected task labels and domain CE; val/test task metadata deliberately changes; inherited parent audit remains bound"
                    if reclassify
                    else "all parent encoded file hashes, retained token/mask/media/order identity, complete train-group exclusion, actual effective membership and CE, identical val/test payloads; inherited parent audit remains bound"
                ),
                completed_unix=time.time(),
            )
            write_json(output / "encoding-audit.json", report)
            summary = dict(
                kind="encoded_media_component",
                status=PASSED,
                formal_admission=False,
                main_budget_eligible=False,
                producer_finished=True,
                manifest_sha256=report["encoded_manifest_sha256"],
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(output / "encoding-audit.json"),
                updated_unix=time.time(),
            )
            write_json(output / "source-audit.json", summary)
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > maximum:
                raise ValueError("filtered data and audit files exceed the byte budget")
            return result
        except BaseException:
            (output / "manifest.json").unlink(missing_ok=True)
            (output / "source-audit.json").unlink(missing_ok=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "encoded", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--config")
    parser.add_argument("--max-gib", type=float, default=1)
    parser.add_argument("--reclassify", action="store_true")
    result = filter_media_encoding(**vars(parser.parse_args()))
    print(json.dumps(dict(format=result["format"], formal_admission=False)))


if __name__ == "__main__":
    main()
