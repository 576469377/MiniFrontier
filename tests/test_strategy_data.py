"""Group split leakage and complete document/answer storage contracts."""

import json
import random
import string

import torch

from minifrontier.data import StageDataset
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer


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


def test_split_binds_same_media_but_does_not_merge_unrelated_image_prompts(tmp_path):
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
    groups = {
        json.loads(payload)["item_id"]: group
        for payload, group in builder.db.execute("SELECT payload,group_root FROM samples")
    }
    assert groups["1"] == groups["2"]
    assert groups["3"] != groups["2"]


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


def test_phash_aliases_are_unique_and_preserve_transitive_groups(tmp_path):
    from minifrontier.data.corpus import simhash

    builder = CorpusBuilder(tmp_path / "corpus")
    rows = [
        record(
            1,
            "A blue mountain lake reflects the distant forest landscape.",
            media=[dict(rgb_sha256="a", phash="0000000000000000")],
        ),
        record(
            2,
            "Several red garden tools rest beside a wooden fence.",
            media=[dict(rgb_sha256="b", phash="000000000000003f")],
        ),
        record(
            3,
            "A train travels across the metal bridge above the river.",
            media=[dict(rgb_sha256="c", phash="0000000000000fff")],
        ),
        record(
            4,
            "Three oranges sit in the center of a large ceramic bowl.",
            media=[dict(rgb_sha256="d", phash="ffffffffffffffff")],
        ),
    ]
    for row in rows:
        assert builder.add(row)
    alias = dict(rows[0], source="second-source", group_id="alias")
    assert not builder.add(alias)
    before = builder.db.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    assert not builder.add(alias)
    assert builder.db.execute("SELECT COUNT(*) FROM links").fetchone()[0] == before
    assert (
        builder.db.execute(
            "SELECT COUNT(*) FROM (SELECT id,key FROM links GROUP BY id,key HAVING COUNT(*)>1)"
        ).fetchone()[0]
        == 0
    )
    builder.finalize()
    groups = {}
    for payload, code, group in builder.db.execute(
        "SELECT payload,simhash,group_root FROM samples"
    ):
        row = json.loads(payload)
        assert code == f"{simhash(row['text']):016x}"
        groups[row["item_id"]] = group
    assert groups["1"] == groups["2"] == groups["3"] != groups["4"]
    builder.db.close()


def test_metadata_reservation_counts_media_alias_fanout_before_mutation(tmp_path):
    import pytest

    builder = CorpusBuilder(tmp_path / "corpus")
    for i in range(20):
        assert builder.add(
            record(
                i,
                f"This photograph numbered {i} has its own distinct pixel identity.",
                media=[dict(rgb_sha256=str(i), phash="0000000000000000")],
            )
        )
    builder.db.commit()
    tables = ("samples", "bands", "links", "image_bands")
    before = [builder.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables]
    builder.max_bytes = builder.approximate_bytes + 4096
    with pytest.raises(ValueError, match="storage budget"):
        builder.add(
            record(
                21,
                "A final photograph connects to all the similar preceding images.",
                media=[dict(rgb_sha256="last", phash="0000000000000000")],
            )
        )
    assert [
        builder.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables
    ] == before
    builder.db.close()
