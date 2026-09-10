"""Canonical image QAs retain pixels, complete answers and held-out identities in MF1."""

import functools
import json
import random
import shutil
import threading
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.encoding_audit import audit_image_encoding
from minifrontier.data.encoding_filters import filter_media_encoding
from minifrontier.data.media_cache import METADATA_BYTES
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, encode_record, safe_text
from minifrontier.data.minifrontier1_components import ComponentDataset, assemble_components
from minifrontier.data.minifrontier1_encoding import (
    CompactDataset,
    canonical_image_record,
    encode_canonical_images,
    encode_canonical_text,
    evaluation_items,
    open_dataset,
)
from minifrontier.data.partitions import create_media_exclusion_view
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.training.minifrontier1 import Sampler, train
from minifrontier.training.minifrontier1_curriculum import collate_records
from minifrontier.training.minifrontier1_strategy import scheduler_factor


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    root = tmp_path / "canonical"
    builder = CorpusBuilder(root)
    for i, task in enumerate(("caption", "vqa", "ocr_document", "chart_table")):
        path = root / f"{i}.png"
        Image.frombytes("RGB", (32, 24), random.Random(i).randbytes(32 * 24 * 3)).save(path)
        checksum = sha256(path)
        assert builder.add(
            dict(
                source="fixture",
                revision="pinned",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                task=task,
                lang="en",
                stage="pretrain",
                text=f"Image {i} question and unique complete answer for {task}.",
                reference_tokens=30,
                answer_reference_tokens=10,
                visual_question=f"Describe image {i}. Is <|image|> printed?",
                visual_answer=f"Complete answer {i}, including literal <|eos|>.",
                media=[
                    dict(
                        kind="image",
                        path=path.name,
                        width=32,
                        height=24,
                        sha256=checksum,
                        rgb_sha256=checksum,
                    )
                ],
            )
        )
    builder.finalize(
        split_locks={"source-group:fixture:2": "val", "source-group:fixture:3": "test"}
    )
    rows = [
        (s, json.loads(p), g)
        for s, p, g in builder.db.execute(
            "SELECT split,payload,group_root FROM samples ORDER BY id"
        )
    ]
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(dict(status="candidate_inventory_below_target", formal_admission=False))
    )
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350, special_tokens=SPECIAL_TOKENS)
    config = MiniFrontier1Config(
        **dict(
            asdict(MiniFrontier1Config.tiny(350)),
            max_position_embeddings=2048,
            protected_media_tokens=1024,
        )
    )
    return root, tokenizer, config, rows


def test_canonical_media_keeps_complete_answer_pixels_and_model_outputs(corpus, tmp_path):
    root, tokenizer, config, rows = corpus
    output = tmp_path / "encoded"
    original_database = sha256(root / "corpus.sqlite")
    manifest = encode_canonical_images(root, tokenizer, output, config, max_features=49)
    audit = audit_image_encoding(root, output, tmp_path / "encoding-audit.json", asdict(config))
    assert audit["status"] == "mechanical_checks_passed_pending_quality_admission"
    assert audit["raw_media_files"] == 4 and not audit["formal_admission"]
    assert sha256(root / "corpus.sqlite") == original_database
    assert not manifest["formal_admission"] and not manifest["raw_media_copied"]
    assert manifest["image_transform"] == dict(max_features=49, document_tiles=False)
    model = MiniFrontier1ForCausalLM(config).eval()
    for split in ("train", "val", "test"):
        dataset = CompactDataset(output, split, config)
        selected = [r for r in rows if r[0] == split]
        assert len(dataset) == len(selected)
        for i, (_, row, group) in enumerate(selected):
            item = dataset[i]
            assert item["sample_id"] == row["sample_id"] and item["split_group"] == group
            assert item["domain"] == row["task"] and item["media_exposures"] == 1
            assert not dataset.windowable_at(i)
            expected_ce = [17, *safe_text(dataset.tokenizer, row["visual_answer"]), 2]
            assert item["labels"][item["labels"] != -100].tolist() == expected_ce
            assert sum(m["feature_count"] for m in item["media"]) <= 49
            for span in item["media"]:
                start, count = span["start"], span["feature_count"]
                assert item["labels"][0, start : start + count].eq(-100).all()
            reference = encode_record(
                canonical_image_record(row, group, max_features=49),
                dataset.tokenizer,
                config,
                root,
            )
            batches = [
                collate_records([sample], config.pad_token_id) for sample in (item, reference)
            ]
            with torch.no_grad():
                results = [
                    model(
                        b["input_ids"],
                        labels=b["labels"],
                        media=b["media"],
                        segment_ids=b.get("segment_ids"),
                    )
                    for b in batches
                ]
            torch.testing.assert_close(results[0].logits, results[1].logits, atol=0, rtol=0)
    assert not list(output.glob("*.png"))


def test_canonical_media_rejects_changed_pixels_and_unfinished_inventory(corpus, tmp_path):
    root, tokenizer, config, _ = corpus
    (root / "source-audit.json").write_text(json.dumps(dict(status="building")))
    with pytest.raises(ValueError, match="must finish"):
        encode_canonical_images(root, tokenizer, tmp_path / "unfinished", config, max_features=49)
    (root / "source-audit.json").write_text(
        json.dumps(dict(status="candidate_inventory_below_target"))
    )
    (root / "0.png").write_bytes(b"changed pixels")
    with pytest.raises(ValueError, match="hash mismatch"):
        encode_canonical_images(root, tokenizer, tmp_path / "changed", config, max_features=49)
    assert not (tmp_path / "changed/manifest.json").exists()


@pytest.mark.parametrize("audit_binding", ["integrity_report", "encoding_audit_sha256"])
def test_compact_exclusion_preserves_tokens_spans_holdout_bytes_and_actual_loader(
    corpus, tmp_path, monkeypatch, audit_binding
):
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3)
    )
    root, tokenizer, config, rows = corpus
    parent = tmp_path / "parent"
    manifest = encode_canonical_images(root, tokenizer, parent, config, max_features=49)
    audit_image_encoding(root, parent, parent / "encoding-audit.json", asdict(config))
    report_hash = sha256(parent / "encoding-audit.json")
    report_binding = (
        dict(integrity_report="encoding-audit.json", integrity_report_sha256=report_hash)
        if audit_binding == "integrity_report"
        else dict(encoding_audit_sha256=report_hash)
    )
    (parent / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="mechanical_checks_passed_pending_quality_admission",
                producer_finished=True,
                manifest_sha256=sha256(parent / "manifest.json"),
                **report_binding,
            )
        )
    )
    (root / "review-samples.jsonl").write_text("")
    rejected = next(row for row in rows if row[0] == "train")
    grouping = tmp_path / "grouping.json"
    grouping.write_text(
        json.dumps(
            dict(
                kind="cross_corpus_media_group_audit",
                inputs={
                    "fixture": dict(
                        corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                        database_sha256=sha256(root / "corpus.sqlite"),
                    )
                },
                split_conflicts=[
                    dict(
                        required_split="test",
                        members=[
                            dict(inventory="fixture", split="train", group=rejected[2], records=1)
                        ],
                    )
                ],
            )
        )
    )
    view = tmp_path / "view"
    create_media_exclusion_view(root, grouping, "fixture", view)
    output = tmp_path / "filtered"
    with monkeypatch.context() as patch:

        def no_pixels(*args, **kwargs):
            raise AssertionError("repacking must not decode image pixels")

        patch.setattr("minifrontier.data.minifrontier1_encoding.prepare_media", no_pixels)
        filtered = filter_media_encoding(view, parent, output, config=asdict(config))
    assert not filtered["formal_admission"]
    audit_image_encoding(view, output, tmp_path / "independent-filter-audit.json", asdict(config))
    for split in ("train", "val", "test"):
        old, new = CompactDataset(parent, split, config), CompactDataset(output, split, config)
        expected = [
            old[i] for i in range(len(old)) if old[i]["sample_id"] != rejected[1]["sample_id"]
        ]
        assert len(new) == len(expected)
        for item, original in zip(new, expected, strict=True):
            assert item["sample_id"] == original["sample_id"]
            for key in ("input_ids", "labels"):
                torch.testing.assert_close(item[key], original[key], atol=0, rtol=0)
            for media, before in zip(item["media"], original["media"], strict=True):
                torch.testing.assert_close(media["patches"], before["patches"], atol=0, rtol=0)
        if split != "train":
            assert filtered["splits"][split] == manifest["splits"][split]
    with pytest.raises(ValueError, match="byte cap"):
        filter_media_encoding(
            view, parent, tmp_path / "tiny-cap", config=asdict(config), max_gib=1e-9
        )
    assert not (tmp_path / "tiny-cap").exists()
    proof = parent / "encoding-audit.json"
    original_proof = proof.read_bytes()
    proof.write_bytes(original_proof + b" ")
    with pytest.raises(ValueError, match="hash/size differs"):
        filter_media_encoding(view, parent, tmp_path / "corrupt-proof", config=asdict(config))
    assert not (tmp_path / "corrupt-proof").exists()
    proof.write_bytes(original_proof)
    # A completed audit cannot justify mutated parent token bytes.
    token_path = parent / manifest["splits"]["train"]["parts"][0]["files"]["tokens.bin"]["name"]
    token_path.write_bytes(b"x" + token_path.read_bytes()[1:])
    with pytest.raises(ValueError, match="hash/size differs"):
        filter_media_encoding(view, parent, tmp_path / "corrupt", config=asdict(config))
    assert not (tmp_path / "corrupt/manifest.json").exists()


def test_canonical_media_never_shortens_answers_to_fit_context(corpus, tmp_path):
    root, tokenizer, config, rows = corpus
    small = MiniFrontier1Config(
        **dict(asdict(config), max_position_embeddings=64, protected_media_tokens=49)
    )
    with pytest.raises(ValueError, match="expanded sample exceeds context"):
        encode_canonical_images(root, tokenizer, tmp_path / "overflow", small, max_features=49)
    for missing in ("visual_question", "visual_answer"):
        row = dict(rows[0][1], **{missing: ""})
        with pytest.raises(ValueError, match="complete grounded QA"):
            canonical_image_record(row, rows[0][2], max_features=49)
    assert not (tmp_path / "overflow/manifest.json").exists()


@pytest.mark.parametrize("fault", ["mask", "geometry"])
def test_image_audit_rejects_rehashed_wrong_supervision_or_span(corpus, tmp_path, fault):
    root, tokenizer, config, _ = corpus
    output = tmp_path / "image-component"
    manifest = encode_canonical_images(root, tokenizer, output, config, max_features=49)
    part = manifest["splits"]["train"]["parts"][0]
    if fault == "mask":
        entry = part["files"]["mask.bin"]
        path = output / entry["name"]
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1  # The BOS must not receive answer CE.
        path.write_bytes(raw)
    else:
        entry = part["files"]["metadata.jsonl"]
        path = output / entry["name"]
        rows = [json.loads(line) for line in path.open()]
        rows[0]["media"][0]["start"] = 4
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
        )
    entry.update(bytes=path.stat().st_size, sha256=sha256(path))
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="labels differ" if fault == "mask" else "geometry"):
        audit_image_encoding(root, output, tmp_path / "audit.json", asdict(config))
    assert json.loads((tmp_path / "audit.json").read_text())["status"] == "failed"


@pytest.fixture
def components(corpus, tmp_path):
    root, tokenizer, config, _ = corpus
    media = tmp_path / "media-component"
    encode_canonical_images(root, tokenizer, media, config, max_features=49)
    text_root = tmp_path / "text-corpus"
    builder = CorpusBuilder(text_root)
    for i in range(3):
        rng = random.Random(i + 70)
        assert builder.add(
            dict(
                source="text-fixture",
                revision="pinned",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                task="en_edu",
                lang="en",
                stage="pretrain",
                text="".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=500)),
            )
        )
    builder.finalize(
        split_locks={"source-group:text-fixture:1": "val", "source-group:text-fixture:2": "test"}
    )
    builder.db.close()
    (text_root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    text = tmp_path / "text-component"
    encode_canonical_text(text_root, tokenizer, text, config)
    return [media, text], config


def test_composition_reuses_shards_with_distinct_domain_indices_and_media_roots(
    components, tmp_path
):
    roots, config = components
    output = tmp_path / "joint"
    manifest = assemble_components(roots, output, config)
    assert manifest["shard_files_copied"] == 0 and not manifest["formal_admission"]
    assert {p.name for p in output.iterdir()} == {"tokenizer.json", "manifest.json"}
    for split in ("train", "val", "test"):
        joint = open_dataset(output, split, config)
        assert isinstance(joint, ComponentDataset)
        offset = 0
        for root in roots:
            child = CompactDataset(root, split, config)
            for i in range(len(child)):
                assert joint.domain_at(offset + i) == child.domain_at(i)
                assert joint.windowable_at(offset + i) == child.windowable_at(i)
                assert joint.length_at(offset + i) == child.length_at(i)
                a, b = joint[offset + i], child[i]
                assert a["sample_id"] == b["sample_id"] and a["split_group"] == b["split_group"]
                torch.testing.assert_close(a["input_ids"], b["input_ids"], atol=0, rtol=0)
                torch.testing.assert_close(a["labels"], b["labels"], atol=0, rtol=0)
            offset += len(child)
        assert offset == len(joint)
        assert joint[-1]["sample_id"] == joint[len(joint) - 1]["sample_id"]
        with pytest.raises(IndexError):
            joint[len(joint)]
        assert (
            sum(
                int(x["labels"][:, 1:].ne(-100).sum())
                for x in evaluation_items(joint, max_length=256)
            )
            == manifest["splits"][split]["counts"]["ce_tokens"]
        )


def test_cached_component_preserves_pixels_tokens_and_sampling_without_local_raw_files(
    components, tmp_path
):
    roots, config = components
    media, _text = roots
    raw = Path(json.loads((media / "manifest.json").read_text())["media_root"])
    original = {
        split: [
            CompactDataset(media, split, config)[i]
            for i in range(len(CompactDataset(media, split, config)))
        ]
        for split in ("train", "val", "test")
    }
    served = tmp_path / "served"
    shutil.copytree(raw, served)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=served)
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    policy = dict(
        base_url=f"http://127.0.0.1:{server.server_port}/",
        uri_prefix="",
        cache_dir="../media-cache",
        max_bytes=METADATA_BYTES + 100_000,
        max_file_bytes=10_000,
        reserve_bytes=0,
    )
    output = tmp_path / "cached-composition"
    try:
        manifest = assemble_components(roots, output, config, media_access={media: policy})
        assert manifest["components"][0]["media_access"] == policy
        for path in raw.glob("*.png"):
            path.rename(path.with_suffix(".saved"))
        for split in ("train", "val", "test"):
            dataset = ComponentDataset(output, split, config)
            for index, expected in enumerate(original[split]):
                actual = dataset[index]
                assert actual["sample_id"] == expected["sample_id"]
                assert dataset.length_at(index) == expected["input_ids"].shape[1]
                for key in ["input_ids", "labels"]:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                for span, before in zip(actual["media"], expected["media"], strict=True):
                    torch.testing.assert_close(span["patches"], before["patches"], atol=0, rtol=0)
                    assert span["source_sha256"] == before["source_sha256"]
            cache = dataset.datasets[0].media_cache
            assert cache.accounting()["actual_bytes"] <= policy["max_bytes"]
        assert cache.accounting()["pinned_files"] == sum(len(original[s]) for s in ["val", "test"])
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_composition_training_resume_retains_window_and_domain_ledgers(components, tmp_path):
    roots, config = components
    output = tmp_path / "joint"
    assemble_components(roots, output, config)
    args = dict(
        data=output,
        config=asdict(config),
        steps=2,
        input_batch_tokens=64,
        weights={"caption": 0.25, "vqa": 0.25, "en_general": 0.5},
        save_every=2,
        eval_every=2,
    )
    train(**args, output=tmp_path / "continuous")
    train(**args, output=tmp_path / "resume", stop_after_updates=1)
    train(**args, output=tmp_path / "resume", resume=tmp_path / "resume/checkpoint.pt")
    a, b = [
        torch.load(tmp_path / x / "checkpoint.pt", weights_only=True)
        for x in ("continuous", "resume")
    ]
    torch.testing.assert_close(a["model"], b["model"], atol=0, rtol=0)
    torch.testing.assert_close(a["optimizer"]["state"], b["optimizer"]["state"], atol=0, rtol=0)
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert a["sampler"] == b["sampler"] and a["ledger"] == b["ledger"]


def test_composition_rejects_duplicate_components_and_changed_child_manifest(components, tmp_path):
    roots, config = components
    with pytest.raises(ValueError, match="distinct components"):
        assemble_components([roots[0], roots[0]], tmp_path / "duplicate", config)
    output = tmp_path / "joint"
    assemble_components(roots, output, config)
    child_manifest = roots[0] / "manifest.json"
    child_manifest.write_text(child_manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        open_dataset(output, "train", config)


@pytest.mark.parametrize("fault", ["sample", "group", "media"])
def test_composition_rejects_cross_split_identities_even_with_rehashed_metadata(
    components, tmp_path, fault
):
    roots, config = components
    child = roots[0]
    manifest = json.loads((child / "manifest.json").read_text())
    first = manifest["splits"]["train"]["parts"][0]["files"]["metadata.jsonl"]
    saved = json.loads(next((child / first["name"]).open()))
    entry = manifest["splits"]["val"]["parts"][0]["files"]["metadata.jsonl"]
    path = child / entry["name"]
    value = json.loads(path.read_text())
    if fault == "media":
        value["resources"][0]["rgb_sha256"] = saved["resources"][0]["rgb_sha256"]
    else:
        key = "sample_id" if fault == "sample" else "split_group"
        value[key] = saved[key]
    path.write_text(json.dumps(value) + "\n")
    entry.update(bytes=path.stat().st_size, sha256=sha256(path))
    (child / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"duplicate|crosses"):
        assemble_components(roots, tmp_path / "leaking", config)
    assert not (tmp_path / "leaking").exists()


def test_performance_window_uses_production_schedule_and_never_exports_weights(
    components, tmp_path
):
    roots, config = components
    data, output = tmp_path / "joint", tmp_path / "performance"
    assemble_components(roots, data, config)
    result = train(
        data=data,
        output=output,
        config=asdict(config),
        phase="p0",
        run_kind="performance",
        input_batch_tokens=64,
        batch_size=2,
        profile_warmup=1,
        profile_updates=2,
        weights={"caption": 0.25, "vqa": 0.25, "en_general": 0.5},
    )
    assert result["state"] == "measurement_complete_unqualified"
    assert result["measured_updates"] == 2
    assert [r["step"] for r in result["updates"]] == [2, 3]
    assert result["all_updates_ledger"]["optimizer_updates"] == 3
    assert result["all_updates_ledger"]["media_exposures"] > 0
    assert not result["exports_model_weights"] and not result["main_budget_eligible"]
    assert not result["formal_admission"] and not result["capability_qualified"]
    assert not list(output.glob("*.pt")) and not (output / "evaluation.json").exists()
    status = json.loads((output / "status.json").read_text())
    assert not status["resumable"]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").open()]
    assert (
        len(metrics) == 3 and result["all_updates_ledger"]["ce_tokens"] == metrics[-1]["ce_tokens"]
    )
    for row in result["updates"]:
        m = metrics[row["step"] - 1]
        factor = scheduler_factor("p0", m["phase_tokens"], m["main_ce_tokens"])
        assert row["context_length"] in {512, 1024}
        assert row["input_tokens"] == m["input_batch_actual"]
        assert row["ce_tokens"] == m["ce_tokens"] - metrics[row["step"] - 2]["ce_tokens"]
        for group in result["recipe"]["optimizer_groups"]:
            assert row["learning_rates"][group["name"]] == pytest.approx(group["base_lr"] * factor)
    assert result["ce_tokens_per_second"] == pytest.approx(
        sum(r["ce_tokens"] for r in result["updates"])
        / sum(r["seconds"] for r in result["updates"])
    )


@pytest.mark.parametrize(
    "override",
    [
        {"init": "old.pt"},
        {"resume": "old.pt"},
        {"steps": 250},
        {"token_budget": 4_000_000},
        {"phase": "p1"},
        {"profile_updates": 201},
    ],
)
def test_performance_entry_rejects_inheritance_and_unbounded_controls(tmp_path, override):
    args = dict(
        data=tmp_path / "absent", output=tmp_path / "output", run_kind="performance", phase="p0"
    )
    with pytest.raises(ValueError, match="fresh bounded"):
        train(**dict(args, **override))
    assert not (tmp_path / "output").exists()


def test_acceptance_actual_token_limit_is_checked_before_any_optimizer_update(
    components, tmp_path, monkeypatch
):
    roots, config = components
    data, output = tmp_path / "joint", tmp_path / "acceptance"
    assemble_components(roots, data, config)

    def oversized(self, dataset, capacity):
        ids = torch.ones(1, 2_000_002, dtype=torch.long)
        labels = ids.clone()
        labels[:, 0] = -100
        return dict(input_ids=ids, labels=labels), "all"

    monkeypatch.setattr(Sampler, "next_item", oversized)
    with pytest.raises(ValueError, match="2M actual-token limit"):
        train(
            data=data,
            output=output,
            config=asdict(config),
            phase="p0",
            steps=1,
            input_batch_tokens=64,
        )
    status = json.loads((output / "status.json").read_text())
    assert status["step"] == 0 and status["ledger"]["ce_tokens"] == 0
    assert status["state"] == "acceptance_token_limit"
    assert not (output / "checkpoint.pt").exists()


def test_visual_partition_reaches_encoding_and_audit_without_copying_pixels(
    corpus, tmp_path, monkeypatch
):
    import shutil
    from types import SimpleNamespace

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    from minifrontier.data.partitions import create_partition_view

    root, tokenizer, config, rows = corpus
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(
                kind="visual_candidate_inventory",
                status="candidate_slice_complete_pending_admission",
                formal_admission=False,
                split_independent_images={"train": 2, "val": 1, "test": 1},
            )
        )
    )
    (root / "integrity-and-split-audit.json").write_text(json.dumps(dict(cross_split_groups=0)))
    selected = next(group for split, _, group in rows if split == "train")
    proposal = tmp_path / "reservation.json"
    proposal.write_text(
        json.dumps(
            dict(
                corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                integrity_audit_sha256=sha256(root / "integrity-and-split-audit.json"),
                minimum_validation_group_fraction=0.005,
                groups={selected: "val"},
            )
        )
    )
    view, output = tmp_path / "view", tmp_path / "encoded-view"
    create_partition_view(root, proposal, view)
    assert not (view / "corpus.sqlite").exists() and not list(view.glob("*.png"))
    partition_audit = json.loads((view / "source-audit.json").read_text())
    assert partition_audit["split_independent_images"] == {"train": 1, "val": 2, "test": 1}
    assert partition_audit["split_independent_groups"] == {"train": 1, "val": 2, "test": 1}
    encode_canonical_images(view, tokenizer, output, config, max_features=49)
    report = audit_image_encoding(view, output, tmp_path / "view-audit.json", asdict(config))
    assert report["status"] == "mechanical_checks_passed_pending_quality_admission"
    assert report["splits"]["train"]["counts"]["records"] == 1
    assert report["splits"]["val"]["counts"]["records"] == 2
    assert selected not in {
        CompactDataset(output, "train", config)[i]["split_group"] for i in range(1)
    }
    assert selected in {CompactDataset(output, "val", config)[i]["split_group"] for i in range(2)}
