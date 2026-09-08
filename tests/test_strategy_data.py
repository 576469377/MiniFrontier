"""Group split leakage and complete document/answer storage contracts."""

import json
import random
import string

import torch

from minifrontier.data import StageDataset
from minifrontier.data_v2 import CorpusBuilder, encode_corpus, train_tokenizer


def record(i, text, *, stage="pretrain", group=None, media=None):
    row = dict(
        source="local-test",
        revision="fixed-v1",
        item_id=str(i),
        group_id=group or str(i),
        license="CC0-1.0",
        lang="en",
        task="natural",
        stage=stage,
        text=text,
    )
    if stage == "sft":
        row["turns"] = [
            dict(role="user", content=text),
            dict(role="assistant", content="The complete answer ends HERE."),
        ]
    if media:
        row["media"] = media
    return row


def test_split_transitively_binds_same_media_and_first_question(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus")
    a = record(
        1,
        "Describe the distinct contents of this photograph.",
        stage="sft",
        media=[dict(rgb_sha256="a")],
    )
    b = record(
        2,
        "Count the objects and report their locations.",
        stage="sft",
        media=[dict(rgb_sha256="a")],
    )
    c = record(
        3,
        "Count the objects and report their locations.",
        stage="sft",
        media=[dict(rgb_sha256="b")],
    )
    c["turns"][-1]["content"] = "Seven orange objects appear at the top of the photograph."
    for row in (a, b, c):
        assert builder.add(row)
    builder.finalize()
    assert len(set(builder.db.execute("SELECT group_root,split FROM samples"))) == 1


def test_question_cap_and_official_test_split_cannot_enter_training(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus")
    for i in range(4):
        row = record(i, "Explain how this experiment works in detail.", stage="sft")
        row["turns"][-1]["content"] = [
            "A magnet moves through a copper coil to induce current.",
            "Blue light travels through a prism and changes direction.",
            "A warmed sealed gas expands and raises the measured pressure.",
            "A red laser illuminates a rotating disk.",
        ][i]
        assert builder.add(row) == (i < 3)
    heldout = record(9, "This held out test document must never enter the training corpus.")
    heldout["official_split"] = "test_sft"
    assert not builder.add(heldout)
    assert builder.counts["first_question_cap"] == 1


def test_same_byte_tokenizer_budget_and_no_cross_document_packing(tmp_path):
    builder = CorpusBuilder(tmp_path / "corpus")
    rng = random.Random(14)
    for i in range(50):
        text = " ".join("".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(16))
        builder.add(record(i, text))
    builder.add(
        record(100, "Explain this text with a complete short response, please.", stage="sft")
    )
    builder.finalize()
    a = train_tokenizer(builder.root, tmp_path / "small.json", 300, byte_budget=4096)
    b = train_tokenizer(builder.root, tmp_path / "large.json", 400, byte_budget=4096)
    assert a["training_byte_sha256"] == b["training_byte_sha256"] and a["training_bytes"] <= 4096
    manifest = encode_corpus(
        builder.root, tmp_path / "small.json", tmp_path / "encoded", max_length=512
    )
    dataset = StageDataset(tmp_path / "encoded", "pretrain", "train", 32)
    assert len(dataset) > 50
    eos_rows = 0
    for x, y in dataset:
        assert (y[x == 0] == -100).all()
        assert int(x.eq(2).sum()) <= 1
        if x.eq(2).any():
            eos_rows += 1
            position = int(x.eq(2).nonzero()[0])
            assert x[position + 1 :].eq(0).all()
    assert eos_rows == manifest["stages"]["pretrain"]["train"]["examples"]
    sft = StageDataset(tmp_path / "encoded", "sft", "train", 512)
    assert len(sft) == 1
    x, y = sft[0]
    assert y[y != -100][-1] == 2
    too_short = StageDataset(tmp_path / "encoded", "sft", "train", 8)
    assert len(too_short) == 0  # Routing to a smaller bucket never truncates an answer.
    stored = json.loads((tmp_path / "encoded/manifest.json").read_text())
    assert stored["format"] == "document-ragged-v2"


def test_chunked_zero_supervision_has_zero_gradients():
    from minifrontier.training.losses import chunked_linear_ce

    h = torch.randn(1, 5, 4, requires_grad=True)
    w = torch.randn(12, 4, requires_grad=True)
    loss = chunked_linear_ce(h, w, torch.full((1, 5), -100))
    loss.backward()
    assert loss == 0 and h.grad.abs().sum() == 0 and w.grad.abs().sum() == 0
