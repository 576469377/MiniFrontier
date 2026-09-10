import json
import random
from contextlib import closing

import pytest

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_tasks import REVISION, SOURCE, allava_task
from minifrontier.data.partitions import (
    TASK_FORMAT,
    create_media_exclusion_view,
    create_partition_view,
    create_task_classification_view,
    open_corpus,
)
from minifrontier.data.visual_sources import candidate_turns

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
                media=[],
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
