import json
import random
from contextlib import closing
from dataclasses import asdict

import pytest
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.encoding_audit import audit_image_encoding
from minifrontier.data.encoding_filters import filter_media_encoding
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.media_tasks import REVISION, SOURCE, allava_task
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS
from minifrontier.data.minifrontier1_encoding import encode_canonical_images
from minifrontier.data.native import audit_native_encoding, encode_native
from minifrontier.data.partitions import (
    TASK_FORMAT,
    create_media_exclusion_view,
    create_partition_view,
    create_task_classification_view,
    open_corpus,
)
from minifrontier.data.visual_sources import candidate_turns
from minifrontier.models.minifrontier1 import MiniFrontier1Config

CAPTION = "Please provide a detailed narrative of the image."
QUESTION = "What do the small label details tell us about the year shown?"


def test_task_is_an_exact_source_template_not_a_keyword_or_turn_number():
    assert allava_task(CAPTION) == "caption"
    assert allava_task("  " + CAPTION.upper().replace(" ", "\n") + "  ") == "caption"
    assert allava_task(QUESTION) == "vqa"
    assert allava_task(CAPTION + " What year was it made?") == "vqa"
    row = dict(
        texts=[
            dict(user=QUESTION, assistant="A complete contextual answer " * 3),
            dict(user=CAPTION, assistant="A complete image description " * 3),
        ]
    )
    for key in [
        "image_correspondence_ratings",
        "visual_dependency_ratings",
        "formatting_ratings",
        "relevance_ratings",
    ]:
        row[key] = [4, 4]
    records, reason = candidate_turns("allava_laion", row)
    assert reason is None and [r["task"] for r in records] == ["vqa", "caption"]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "0")
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": 900 * 1024**3})(),
    )
    root = tmp_path / "corpus"
    builder = CorpusBuilder(root)
    for i in range(6):
        rng = random.Random(i)
        answer = " ".join(rng.randbytes(8).hex() for _ in range(40))
        question = CAPTION if i % 2 == 0 else QUESTION
        image = Image.frombytes("RGB", (32, 24), rng.randbytes(32 * 24 * 3))
        path = root / f"{i}.png"
        image.save(path)
        assert builder.add(
            dict(
                source=SOURCE,
                revision=REVISION,
                item_id=str(i),
                group_id=str(i),
                license="fixture",
                lang="en",
                stage="pretrain",
                task="vqa" if i % 2 == 0 else "caption",
                text=question + " " + answer,
                visual_question=question,
                visual_answer=answer,
                answer_reference_tokens=30,
                media=[
                    dict(kind="image", path=path.name, sha256=sha256(path), **decoded_hashes(image))
                ],
            )
        )
    locks = {"source-group:" + SOURCE + ":" + str(i): "val" if i == 4 else "test" for i in (4, 5)}
    builder.finalize(split_locks=locks)
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    (root / "integrity-and-split-audit.json").write_text("{}")
    (root / "review-samples.jsonl").write_text("")
    with closing(open_corpus(root)) as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM samples ORDER BY id")]
        groups = dict(
            db.execute("SELECT json_extract(payload,'$.item_id'),group_root FROM samples")
        )
    return root, rows, groups


def test_task_view_keeps_contents_partitions_and_subsequent_group_exclusions(corpus, tmp_path):
    root, _rows, groups = corpus
    proposal = tmp_path / "reservation.json"
    proposal.write_text(
        json.dumps(
            dict(
                corpus_manifest_sha256=sha256(root / "corpus-manifest.json"),
                integrity_audit_sha256=sha256(root / "integrity-and-split-audit.json"),
                minimum_validation_group_fraction=0.01,
                groups={groups["3"]: "val"},
            )
        )
    )
    partition = tmp_path / "partition"
    create_partition_view(root, proposal, partition)
    old_db_hash = sha256(root / "corpus.sqlite")
    target = tmp_path / "corrected"
    result = create_task_classification_view(partition, target)
    assert result["format"] == TASK_FORMAT and not (target / "corpus.sqlite").exists()
    assert sha256(root / "corpus.sqlite") == old_db_hash
    with closing(open_corpus(partition)) as before, closing(open_corpus(target)) as after:
        query = "SELECT id,split,group_root,task,payload FROM samples ORDER BY id"
        for old, new in zip(before.execute(query), after.execute(query), strict=True):
            assert old[:3] == new[:3]
            a, b = json.loads(old[4]), json.loads(new[4])
            assert b.pop("task") == new[3] == allava_task(a["visual_question"])
            assert a.pop("task") == old[3] and a == b
    proof = json.loads((target / "source-audit.json").read_text())
    assert sum(sum(v.values()) for v in proof["task_changes"].values()) == 6
    assert proof["review"]["status"] == "awaiting_manual_review"
    grouping = tmp_path / "grouping.json"
    grouping.write_text(
        json.dumps(
            dict(
                kind="cross_corpus_media_group_audit",
                inputs={
                    "fixture": dict(
                        corpus_manifest_sha256=sha256(target / "corpus-manifest.json"),
                        database_sha256=old_db_hash,
                    )
                },
                split_conflicts=[
                    dict(
                        required_split="val",
                        members=[
                            dict(inventory="fixture", split="train", group=groups["1"], records=1)
                        ],
                    )
                ],
            )
        )
    )
    excluded = tmp_path / "corrected-excluded"
    create_media_exclusion_view(target, grouping, "fixture", excluded)
    assert json.loads((excluded / "corpus-manifest.json").read_text())["format"] == TASK_FORMAT
    with closing(open_corpus(excluded)) as db:
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone() == (5,)
        assert all(
            task == allava_task(question)
            for task, question in db.execute(
                "SELECT task,json_extract(payload,'$.visual_question') FROM samples"
            )
        )
    with pytest.raises(ValueError, match="already binds"):
        create_task_classification_view(target, tmp_path / "again")
    overrides = json.loads((target / "split-overrides.json").read_text())
    overrides["task_policy"]["classifier_sha256"] = "0" * 64
    (target / "split-overrides.json").write_text(json.dumps(overrides))
    with pytest.raises(ValueError, match="reservation hash"):
        open_corpus(target)


@pytest.mark.parametrize("family", ["mf1", "minikimik3", "miniqwen4"])
def test_task_repacking_preserves_all_supervision_and_passes_independent_audit(
    corpus, tmp_path, monkeypatch, family
):
    root, _, _ = corpus
    tokenizer = tmp_path / "tokenizer.json"
    if family == "mf1":
        train_tokenizer(root, tokenizer, 400, special_tokens=SPECIAL_TOKENS)
    else:
        train_tokenizer(root, tokenizer, 400)
    parent, output = tmp_path / "parent", tmp_path / "corrected-encoding"
    config = MiniFrontier1Config(
        **dict(asdict(MiniFrontier1Config.tiny(400)), max_position_embeddings=2048)
    )
    if family == "mf1":
        before = encode_canonical_images(root, tokenizer, parent, config, max_features=4)
        proof = audit_image_encoding(root, parent, parent / "encoding-audit.json", asdict(config))
    else:
        text = CorpusBuilder(tmp_path / "text-corpus")
        assert text.add(
            dict(
                source="fixture",
                revision="fixed",
                item_id="1",
                group_id="1",
                license="fixture",
                task="en_edu",
                stage="pretrain",
                lang="en",
                text="Clouds form when water vapour condenses in the cool air above us.",
            )
        )
        text.finalize()
        text.db.close()
        shared = tmp_path / "shared-text"
        encode_corpus(text.root, tokenizer, shared, max_length=2048)
        before = encode_native(
            root,
            tokenizer,
            parent,
            family,
            max_length=2048,
            max_features=4,
            min_pixels=1024 if family == "miniqwen4" else None,
            text_encoding=shared,
            max_gib=0.01,
        )
        proof = audit_native_encoding(root, parent, parent / "encoding-audit.json")
    (parent / "source-audit.json").write_text(
        json.dumps(
            dict(
                status=proof["status"],
                producer_finished=True,
                manifest_sha256=sha256(parent / "manifest.json"),
                integrity_report="encoding-audit.json",
                integrity_report_sha256=sha256(parent / "encoding-audit.json"),
            )
        )
    )
    view = tmp_path / "task-view"
    create_task_classification_view(root, view)

    def no_pixels(*args, **kwargs):
        raise AssertionError("task correction must reuse tokenized records without pixel encoding")

    with monkeypatch.context() as patch:
        patch.setattr("minifrontier.data.minifrontier1_encoding.prepare_media", no_pixels)
        patch.setattr("minifrontier.data.encoding_filters.prepare_record", no_pixels)
        after = filter_media_encoding(view, parent, output, config=asdict(config), reclassify=True)
    audit = json.loads((output / "encoding-audit.json").read_text())
    assert audit["kind"] == "encoded_media_task_derivation"
    assert not audit["formal_admission"] and not audit["sealed_holdout_payloads_unchanged"]
    assert audit["sealed_holdout_contents_and_membership_unchanged"]
    if family == "mf1":
        independent = audit_image_encoding(
            view, output, tmp_path / "independent.json", asdict(config)
        )
        for split in ("train", "val", "test"):
            assert before["splits"][split]["counts"] == after["splits"][split]["counts"]
    else:
        independent = audit_native_encoding(view, output, tmp_path / "independent.json")
        for stage in ("pretrain", "sft"):
            for split in ("train", "val", "test"):
                a, b = before["stages"][stage][split], after["stages"][stage][split]
                assert a["supervised_tokens"] == b["supervised_tokens"]
                assert a["examples"] == b["examples"] and a["text"] == b["text"]
    assert independent["status"] == proof["status"] and not independent.get("errors")
    with pytest.raises(ValueError, match="declared refinement"):
        filter_media_encoding(view, parent, tmp_path / "wrong-operation", config=asdict(config))
