"""Canonical image QAs retain pixels, complete answers and held-out identities in MF1."""

import json
import random
from dataclasses import asdict

import pytest
import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
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
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.training.minifrontier1 import train
from minifrontier.training.minifrontier1_curriculum import collate_records


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
