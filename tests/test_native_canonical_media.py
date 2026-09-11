"""Canonical visual PT preserves literal text and shares audited text storage."""

import copy
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_kimi
from tokenizers import Tokenizer

from minifrontier.data import StageDataset, sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.encoding_filters import filter_media_encoding
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.native import audit_native_encoding, encode_native
from minifrontier.data.native_components import assemble_native_components
from minifrontier.data.partitions import create_media_exclusion_view, create_partition_view
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.multimodal import collate, prepare_record, pretraining_tokens
from minifrontier.training.media_mixture import MediaMixtureCursor


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    root = tmp_path / "canonical"
    builder = CorpusBuilder(root)
    for i, task in enumerate(("caption", "vqa", "ocr_document", "chart_table")):
        image = Image.frombytes("RGB", (32, 24), random.Random(i).randbytes(32 * 24 * 3))
        path = root / f"{i}.png"
        image.save(path)
        question = f"Describe image {i}. Is <|image|> printed?"
        answer = f"Complete answer {i}, including literal <|eos|>."
        assert builder.add(
            dict(
                source="fixture",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                stage="pretrain",
                task=task,
                text="<|image|>\nUser: " + question + "\nAssistant: " + answer,
                visual_question=question,
                visual_answer=answer,
                reference_tokens=30,
                answer_reference_tokens=20,
                media=[
                    dict(kind="image", path=path.name, sha256=sha256(path), **decoded_hashes(image))
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
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350)
    return root, tokenizer, None, rows


@pytest.fixture
def inputs(corpus, tmp_path):
    root, tokenizer, _, rows = corpus
    text = CorpusBuilder(tmp_path / "canonical-text")
    sentences = (
        "The museum preserves rare books and maps for future generations.",
        "A spacecraft measures the magnetic field around a distant planet.",
        "Insects communicate through chemical signals produced by their glands.",
    )
    for i, sentence in enumerate(sentences):
        assert text.add(
            dict(
                source="text",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                stage="pretrain",
                task="en_edu",
                text=sentence,
            )
        )
    text.finalize(split_locks={"source-group:text:1": "val", "source-group:text:2": "test"})
    text.db.close()
    shared = tmp_path / "shared-text"
    encode_corpus(text.root, tokenizer, shared, max_length=512)
    return root, tokenizer, rows, shared


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_canonical_literals_stay_text_and_full_pt_targets_reach_vision(corpus, family):
    root, tokenizer_path, _, rows = corpus
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    row = rows[0][1]
    expected = "\nUser: " + row["visual_question"] + "\nAssistant: " + row["visual_answer"]
    assert "<|image|>" in expected and "<|eos|>" in expected
    before = tokenizer.encode_special_tokens
    batch = prepare_record(
        row,
        tokenizer,
        family,
        root=root,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
    )
    assert tokenizer.encode_special_tokens == before
    target = batch.labels[0][batch.labels[0].ne(-100)].tolist()
    assert tokenizer.decode(target, skip_special_tokens=True) == expected
    assert set(target).intersection(range(20)) == {2}
    assert batch.labels[0, -1] == 2 and batch.input_ids.eq(2).sum() == 1
    assert batch.input_ids.eq(7).sum() == batch.image_features
    if family == "minikimik3":
        vision = KimiVisionConfig(
            depth=1,
            hidden_size=32,
            qkv_hidden_size=48,
            num_heads=2,
            intermediate_size=64,
            output_size=32,
        )
        model = MiniKimiK3ForCausalLM(
            tiny_kimi(vocab_size=512, max_position_embeddings=512, vision_config=vision)
        )
    else:
        vision = QwenVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        )
        model = MiniQwen4ForCausalLM(
            tiny_config(
                vocab_size=512, hidden_size=32, max_position_embeddings=512, vision_config=vision
            )
        )
    result = model(batch.input_ids, labels=batch.labels, return_logits=False, **batch.extras)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision.parameters())
    with pytest.raises(ValueError, match="complete question/answer"):
        pretraining_tokens(dict(row, visual_answer=""), tokenizer)
    assert tokenizer.encode_special_tokens == before
    with pytest.raises(ValueError, match="raw media hash"):
        prepare_record(
            dict(row, media=[dict(row["media"][0], sha256="bad")]),
            tokenizer,
            family,
            root=root,
            max_features=4,
            min_pixels=1024 if family == "miniqwen4" else None,
        )


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_native_reuses_text_files_and_resolves_pixels_through_partition(
    inputs, tmp_path, family, monkeypatch
):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root, tokenizer, rows, shared = inputs
    original = sha256(root / "corpus.sqlite")
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    integrity = root / "integrity-and-split-audit.json"
    integrity.write_text(json.dumps(dict(cross_split_groups=0)))
    group = next(g for split, _, g in rows if split == "train")
    proposal = tmp_path / "reservation.json"
    proposal.write_text(
        json.dumps(
            dict(
                corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                integrity_audit_sha256=sha256(integrity),
                minimum_validation_group_fraction=0.005,
                groups={group: "val"},
            )
        )
    )
    view = tmp_path / "partition"
    create_partition_view(root, proposal, view)
    output = tmp_path / family
    manifest = encode_native(
        view,
        tokenizer,
        output,
        family,
        text_encoding=shared,
        max_length=512,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
        max_gib=0.01,
    )
    assert manifest["shared_text_files_copied"] == 0 and not manifest["formal_admission"]
    assert manifest["media_root"] == str(root)
    assert sha256(root / "corpus.sqlite") == original
    assert not list(output.glob("*.bin")) and not list(output.glob("pretrain.*.labels*"))
    assert all(p.suffix != ".png" for p in output.rglob("*"))
    assert (output / manifest["text_source"]["path"]).resolve() == shared
    audit = audit_native_encoding(view, output, output / "encoding-audit.json")
    assert audit["status"] == "mechanical_checks_passed_pending_quality_admission"
    assert sum(v["counts"]["records"] for v in audit["splits"].values()) == 4
    assert not audit["errors"] and not audit["formal_admission"]
    for split, media_examples in (("train", 1), ("val", 2), ("test", 1)):
        data = StageDataset(output, "pretrain", split, 512)
        text = StageDataset(shared, "pretrain", split, 512)
        assert len(data) == len(text) + media_examples
        batch = collate([data[i] for i in range(len(data))], "cpu")
        assert batch.image_count == media_examples
        assert int(batch.labels[:, 1:].ne(-100).sum()) == sum(data.documents.ce_counts)
        assert torch.equal(data[0].input_ids[0], text[0][0])
        serialized = [
            json.loads(line)
            for line in (output / f"pretrain.{split}.media.jsonl").read_text().splitlines()
        ]
        assert (group in {v["split_group"] for v in serialized}) == (split == "val")
    # A different shared manifest cannot be accepted merely because the files still exist.
    source_manifest = shared / "manifest.json"
    source_manifest.write_text(source_manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="shared native text manifest"):
        StageDataset(output, "pretrain", "train", 512)


def test_native_disk_limit_never_publishes_partial_manifest_or_changes_shared_data(
    inputs, tmp_path
):
    root, tokenizer, _, shared = inputs
    checksum = sha256(shared / "manifest.json")
    output = tmp_path / "bounded"
    budget = (Path(tokenizer).stat().st_size + 128 * 1024 + 10) / 1024**3
    with pytest.raises(ValueError, match="disk budget"):
        encode_native(
            root,
            tokenizer,
            output,
            "minikimik3",
            text_encoding=shared,
            max_features=4,
            max_gib=budget,
        )
    assert not (output / "manifest.json").exists()
    assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= budget * 1024**3
    assert sha256(shared / "manifest.json") == checksum


@pytest.mark.parametrize("size", [(32, 24), (60, 20), (100, 200)])
def test_qwen_small_image_minimum_rounding_cannot_exceed_phase_pixel_cap(size):
    from minifrontier.models.miniqwen4.processing import process_frames
    from minifrontier.models.miniqwen4.upstream_processing import smart_resize

    h, w = smart_resize(size[1], size[0], factor=32, min_pixels=50176, max_pixels=50176)
    assert h * w // 1024 > 49  # The upstream ceil alone exceeds this mini budget.
    result = process_frames([Image.new("RGB", size)], max_features=49, min_pixels=50176)
    assert result["feature_count"] <= 49
    assert result["patches"].shape[0] == 4 * result["feature_count"]


def test_native_full_audit_rejects_rehashed_partial_text_supervision(inputs, tmp_path):
    root, tokenizer, _, shared = inputs
    output = tmp_path / "encoded"
    manifest = encode_native(
        root, tokenizer, output, "minikimik3", text_encoding=shared, max_features=4
    )
    meta = manifest["stages"]["pretrain"]["val"]["media"]
    path = output / meta["file"]
    row = json.loads(path.read_text())
    index = next(i for i, value in enumerate(row["expected_labels"]) if value > 23)
    row["expected_labels"][index] = -100
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    index_path = output / meta["index_file"]
    offsets = np.load(index_path)
    offsets[0, 1] = path.stat().st_size
    np.save(index_path, offsets)
    meta.update(sha256=sha256(path), index_sha256=sha256(index_path))
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ids/labels"):
        audit_native_encoding(root, output, output / "encoding-audit.json")
    report = json.loads((output / "encoding-audit.json").read_text())
    assert report["status"] == "failed" and not report["formal_admission"]


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
@pytest.mark.parametrize("overflow", [False, True])
def test_native_exclusions_preserve_kept_bytes_shared_text_and_overflow_accounting(
    inputs, tmp_path, family, overflow, monkeypatch
):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root, tokenizer, rows, shared = inputs
    parent = tmp_path / "parent"
    manifest = encode_native(
        root,
        tokenizer,
        parent,
        family,
        text_encoding=shared,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
        max_length=2 if overflow else 1024,
    )
    audit_native_encoding(root, parent, parent / "encoding-audit.json")
    (parent / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="mechanical_checks_passed_pending_quality_admission",
                producer_finished=True,
                manifest_sha256=sha256(parent / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(parent / "encoding-audit.json"),
            )
        )
    )
    (root / "source-audit.json").write_text(
        json.dumps(dict(status="candidate_inventory_below_target", formal_admission=False))
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
                        required_split="val",
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
    filtered = filter_media_encoding(view, parent, output)
    audit_native_encoding(view, output, tmp_path / "independent-filter-audit.json")
    assert (
        not filtered["formal_admission"]
        and filtered["text_source"]["manifest_sha256"] == manifest["text_source"]["manifest_sha256"]
    )
    for split in ("train", "val", "test"):
        original = manifest["stages"]["pretrain"][split]["media"]
        current = filtered["stages"]["pretrain"][split]["media"]
        expected = [
            line
            for line in (parent / original["file"]).read_bytes().splitlines(keepends=True)
            if json.loads(line)["record"]["sample_id"] != rejected[1]["sample_id"]
        ]
        assert (output / current["file"]).read_bytes() == b"".join(expected)
        if split != "train":
            assert current == original
        # The actual training reader includes the unchanged shared text component.
        dataset = StageDataset(output, "pretrain", split, 1024)
        before = StageDataset(parent, "pretrain", split, 1024)
        assert len(dataset) == len(before) - int(split == "train" and not overflow)
        for item in dataset:
            assert item.input_ids.shape == item.labels.shape
    assert not list(output.glob("*.tokens.bin"))
    if overflow:
        counts = filtered["stages"]["pretrain"]["train"]["media"]["rejected"]
        assert counts["complete_media_answer_exceeds_bucket"] == 1
    else:
        other = next(row for row in rows if row[0] == "train" and row[2] != rejected[2])
        evidence = json.loads(grouping.read_text())
        evidence["inputs"]["fixture"]["corpus_manifest_sha256"] = sha256(
            view / "corpus-manifest.json"
        )
        evidence["split_conflicts"][0]["members"][0]["group"] = other[2]
        grouping.write_text(json.dumps(evidence))
        next_view = tmp_path / "next-view"
        create_media_exclusion_view(view, grouping, "fixture", next_view)
        twice = tmp_path / "filtered-again"
        final = filter_media_encoding(next_view, output, twice)
        assert final["stages"]["pretrain"]["train"]["media"]["examples"] == 0
        audit_native_encoding(next_view, twice, tmp_path / "second-independent-audit.json")


def _audited_native(root, tokenizer, shared, output, family):
    encode_native(
        root,
        tokenizer,
        output,
        family,
        text_encoding=shared,
        max_length=512,
        max_features=4,
        min_pixels=1024 if family == "miniqwen4" else None,
    )
    audit_native_encoding(root, output, output / "encoding-audit.json")
    (output / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="mechanical_checks_passed_pending_quality_admission",
                producer_finished=True,
                manifest_sha256=sha256(output / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(output / "encoding-audit.json"),
            )
        )
    )
    return output


def _second_native_corpus(tmp_path, *, conflicting_image=None):
    builder = CorpusBuilder(tmp_path / "second-corpus")
    for i, task in enumerate(("caption", "ocr_document", "chart_table")):
        image = (
            Image.open(conflicting_image).copy()
            if conflicting_image is not None and i == 1
            else Image.frombytes("RGB", (32, 24), random.Random(100 + i).randbytes(32 * 24 * 3))
        )
        path = builder.root / f"{i}.png"
        image.save(path)
        question = f"Read the number {i} shown on the diagram."
        answer = f"The displayed figure contains marker {i} and a rectangular outline."
        assert builder.add(
            dict(
                source="second-fixture",
                revision="fixed",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                lang="en",
                stage="pretrain",
                task=task,
                text=question + " " + answer,
                visual_question=question,
                visual_answer=answer,
                media=[
                    dict(kind="image", path=path.name, sha256=sha256(path), **decoded_hashes(image))
                ],
            )
        )
    builder.finalize(
        split_locks={
            "source-group:second-fixture:1": "val",
            "source-group:second-fixture:2": "test",
        }
    )
    builder.db.close()
    return builder.root


def _assert_native_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_native_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_native_tree_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4"])
def test_native_composition_reads_each_media_root_and_shared_text_once(
    inputs, tmp_path, family, monkeypatch
):
    from minifrontier.data import native

    root, tokenizer, _, shared = inputs
    second = _second_native_corpus(tmp_path)
    children = [
        _audited_native(c, tokenizer, shared, tmp_path / f"encoded-{i}", family)
        for i, c in enumerate((root, second))
    ]
    output = tmp_path / "composed"
    manifest = assemble_native_components(children, output)
    assert not manifest["formal_admission"] and not manifest["main_budget_eligible"]
    assert sorted(p.name for p in output.iterdir()) == ["manifest.json", "tokenizer.json"]
    original = native.DocumentDataset
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "DocumentDataset", counted)
    for split in ("train", "val", "test"):
        before = len(calls)
        data = StageDataset(output, "pretrain", split, 512)
        assert len(calls) == before + 1 and calls[-1] == shared
        text = StageDataset(shared, "pretrain", split, 512)
        expected = []
        for child in children:
            source = StageDataset(child, "pretrain", split, 512)
            expected.extend(source[i] for i in range(len(text), len(source)))
        assert len(data) == len(text) + len(expected)
        assert torch.equal(data[0].input_ids[0], text[0][0])
        for i, item in enumerate(expected, len(text)):
            current = data[i]
            torch.testing.assert_close(current.input_ids, item.input_ids)
            torch.testing.assert_close(current.labels, item.labels)
            _assert_native_tree_equal(current.extras, item.extras)
        batch = collate([data[i] for i in range(len(data))], "cpu")
        assert int(batch.labels[:, 1:].ne(-100).sum()) == sum(data.documents.ce_counts)
        assert (
            sum(data.documents.ce_counts)
            == manifest["stages"]["pretrain"][split]["supervised_tokens"]
        )
        assert batch.image_count == sum(data.documents.image_counts) == len(expected)
        torch.testing.assert_close(data[-1].input_ids, data[len(data) - 1].input_ids)
        with pytest.raises(IndexError):
            data[len(data)]
    # Existing sampler resumes over the same combined index and separate media counters.
    data = StageDataset(output, "pretrain", "train", 512)
    domains = set(
        d for d, n in zip(data.documents.domains, data.documents.image_counts, strict=True) if n
    )
    recipe = dict(
        schema_version=1,
        ce_token_budget=1000,
        image_occurrences=50,
        text_mixture_tokens={"en_edu": 1.0},
        image_mixture_samples={d: 1 / len(domains) for d in domains},
    )
    cursor = MediaMixtureCursor(data, recipe, batch_size=2)
    for _ in range(5):
        cursor.next()
    saved = copy.deepcopy(cursor.state_dict())
    expected = [cursor.next() for _ in range(8)]
    restored = MediaMixtureCursor(
        StageDataset(output, "pretrain", "train", 512), recipe, batch_size=2
    )
    restored.load_state_dict(saved)
    assert [restored.next() for _ in range(8)] == expected
    assert restored.state_dict() == cursor.state_dict()
    # The composition cannot silently accept changed component bytes on reload.
    p = children[1] / "pretrain.train.media.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        StageDataset(output, "pretrain", "train", 512)


def test_native_composition_rejects_duplicate_samples_and_metadata_overflow(inputs, tmp_path):
    root, tokenizer, _, shared = inputs
    child = _audited_native(root, tokenizer, shared, tmp_path / "encoded", "minikimik3")
    copied = tmp_path / "copied"
    shutil.copytree(child, copied)
    with pytest.raises(ValueError, match="duplicate sample"):
        assemble_native_components([child, copied], tmp_path / "duplicates")
    with pytest.raises(ValueError, match="disk budget"):
        assemble_native_components([child], tmp_path / "overflow", max_bytes=1)
    assert not (tmp_path / "duplicates").exists() and not (tmp_path / "overflow").exists()


def test_native_composition_rejects_individually_valid_cross_split_pixels(inputs, tmp_path):
    root, tokenizer, rows, shared = inputs
    training = next(row for split, row, _ in rows if split == "train")
    second = _second_native_corpus(tmp_path, conflicting_image=root / training["media"][0]["path"])
    children = [
        _audited_native(c, tokenizer, shared, tmp_path / f"encoded-{i}", "minikimik3")
        for i, c in enumerate((root, second))
    ]
    with pytest.raises(ValueError, match="media identity crosses splits"):
        assemble_native_components(children, tmp_path / "cross-split")


@pytest.mark.parametrize("mutation", ["family", "shared-text"])
def test_native_composition_rejects_mismatched_model_or_shared_text(inputs, tmp_path, mutation):
    root, tokenizer, _, shared = inputs
    child = _audited_native(root, tokenizer, shared, tmp_path / "encoded", "minikimik3")
    copied = tmp_path / "copied"
    shutil.copytree(child, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    if mutation == "family":
        manifest["max_features"] += 1
        match = "component model"
    else:
        alternate = tmp_path / "alternate-text"
        shutil.copytree(shared, alternate)
        manifest["text_source"]["path"] = "../alternate-text"
        match = "one shared text directory"
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        assemble_native_components([child, copied], tmp_path / "invalid")
