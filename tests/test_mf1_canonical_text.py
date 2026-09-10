"""Canonical MF1 documents: one storage copy, complete CE coverage and resumable windows."""

import copy
import json
import random
import string
from dataclasses import asdict

import pytest
import torch

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS
from minifrontier.data.minifrontier1_encoding import (
    CompactDataset,
    encode_canonical_text,
    evaluation_items,
)
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.training.minifrontier1 import Sampler, train


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_state_equal(left, right)
    else:
        assert a == b


@pytest.fixture
def encoded(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    root = tmp_path / "canonical"
    builder = CorpusBuilder(root, val_buckets=1, test_buckets=1)
    rng = random.Random(45)
    for i in range(4):
        text = " ".join("".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(100))
        text += "\nLiteral control spelling: " + SPECIAL_TOKENS[2]
        assert builder.add(
            dict(
                source="fixture",
                revision="pinned",
                item_id=str(i),
                group_id=str(i),
                license="CC0-1.0",
                task="en_edu",
                lang="en",
                stage="pretrain",
                text=text,
                reference_tokens=len(text),
            )
        )
    builder.finalize(
        split_locks={"source-group:fixture:2": "val", "source-group:fixture:3": "test"}
    )
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    tokenizer = tmp_path / "tokenizer.json"
    train_tokenizer(root, tokenizer, 350, special_tokens=SPECIAL_TOKENS)
    config = MiniFrontier1Config(
        **dict(asdict(MiniFrontier1Config.tiny(350)), max_position_embeddings=128)
    )
    output = tmp_path / "compact"
    original_hash = sha256(root / "corpus.sqlite")
    manifest = encode_canonical_text(root, tokenizer, output, config, shard_tokens=256)
    assert sha256(root / "corpus.sqlite") == original_hash
    return output, config, manifest


def test_all_targets_survive_variable_length_windows_and_sampler_resume(encoded):
    output, config, manifest = encoded
    data = CompactDataset(output, "train", config)
    assert len(data) == 2 and len(manifest["splits"]["train"]["parts"]) == 2
    assert not manifest["formal_admission"] and not manifest["raw_text_copied"]
    assert all(data.length_at(i) > config.max_position_embeddings for i in range(len(data)))
    sampler = Sampler(data, 14, {"en_general": 1.0}, length_filter=True)
    first, domain = sampler.next_item(data, 32)
    assert first["document_offset"] == 0 and first["input_ids"].shape[1] == 32
    sampler.ce[domain] += 31
    saved = copy.deepcopy(sampler.state_dict())
    restored = Sampler(data, 14, {"en_general": 1.0}, length_filter=True)
    restored.load_state_dict(copy.deepcopy(saved))
    targets = [first["labels"][0, 1:]]
    index = sampler.document_offsets[domain][0]
    full = data[index]["input_ids"][0]
    assert full.eq(2).sum() == 1  # A literal EOS spelling is escaped, not a boundary.
    for i in range(100):
        capacity = (17, 64, 33)[i % 3]
        a, da = sampler.next_item(data, capacity)
        b, db = restored.next_item(data, capacity)
        assert da == db and a["sample_id"] == b["sample_id"]
        torch.testing.assert_close(a["input_ids"], b["input_ids"], atol=0, rtol=0)
        torch.testing.assert_close(a["labels"], b["labels"], atol=0, rtol=0)
        assert a["labels"][0, 0] == -100 and a["input_ids"].shape[1] <= capacity
        count = a["labels"][:, 1:].ne(-100).sum().item()
        sampler.ce[da] += count
        restored.ce[db] += count
        targets.append(a["labels"][0, 1:])
        if not sampler.document_offsets:
            break
    else:
        pytest.fail("the long document did not finish")
    torch.testing.assert_close(torch.cat(targets), full[1:], atol=0, rtol=0)
    assert sampler.state_dict() == restored.state_dict()
    assert sampler.examples[domain] == 1
    assert manifest["splits"]["train"]["counts"]["text_documents"] == 2
    assert manifest["splits"]["train"]["domain_ce"]["en_general"] == sum(
        data.length_at(i) - 1 for i in range(len(data))
    )


def test_validation_windows_cover_heldout_once_and_do_not_split_native_answers(encoded):
    output, config, manifest = encoded
    val = CompactDataset(output, "val", config)
    items = list(evaluation_items(val, max_length=31))
    assert len(val) == 1 and len(items) > 1
    full = val[0]
    torch.testing.assert_close(
        torch.cat([item["labels"][0, 1:] for item in items]), full["labels"][0, 1:], atol=0, rtol=0
    )
    assert (
        sum(int(item["labels"][:, 1:].ne(-100).sum()) for item in items)
        == manifest["splits"]["val"]["counts"]["ce_tokens"]
    )
    assert {item["split_group"] for item in items} == {full["split_group"]}
    assert {full["split_group"]}.isdisjoint(
        {CompactDataset(output, "train", config)[i]["split_group"] for i in range(2)}
    )
    with pytest.raises(ValueError, match="complete validation"):
        list(evaluation_items([dict(input_ids=torch.ones(1, 32, dtype=torch.long))], max_length=16))


def test_checkpoint_resume_retains_the_middle_of_a_canonical_document(encoded, tmp_path):
    output, config, _ = encoded
    args = dict(
        data=output,
        config=asdict(config),
        steps=2,
        input_batch_tokens=64,
        save_every=2,
        eval_every=2,
    )
    train(**args, output=tmp_path / "continuous")
    train(**args, output=tmp_path / "resumed", stop_after_updates=1)
    paused = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert paused["sampler"]["document_offsets"]
    train(**args, output=tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint.pt")
    a, b = [
        torch.load(tmp_path / p / "checkpoint.pt", weights_only=True)
        for p in ("continuous", "resumed")
    ]
    for key in ("model", "optimizer", "sampler", "router_balance", "ledger", "rng"):
        assert_state_equal(a[key], b[key])
