"""Real corpus merge/exclusion: preserve sealed aliases and remove whole groups."""

import json
import shutil
import sqlite3
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.evaluation import RULES, BenchmarkMatcher, _download
from minifrontier.data.pretraining import merge_text_slices

PROMPT = "A gardener planted seventeen apple trees beside the river and twice as many pear trees. How many trees were planted altogether?"


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3))
    root = tmp_path / "evaluation"
    root.mkdir()
    (root / "items.jsonl").write_text(
        json.dumps(dict(id="test:1", prompt=PROMPT, answer="51")) + "\n"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                rules=RULES,
                files={"items.jsonl": sha256(root / "items.jsonl")},
                limitations=["fixture overlap only"],
            )
        )
    )
    return root


def candidate(root, rows, *, locks=None):
    builder = CorpusBuilder(root, val_buckets=50, test_buckets=100)
    for item, group, text, repo in rows:
        assert builder.add(
            dict(
                source=root.name,
                revision="pinned",
                item_id=item,
                group_id=group,
                license="CC0",
                lang="en",
                task="natural",
                stage="pretrain",
                text=text,
                repo_id=repo,
                reference_tokens=len(text),
            )
        )
    builder.finalize(split_locks=locks)
    builder.db.close()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.save(str(root / "reference-tokenizer.json"))
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(
                status="candidate_slice_complete_pending_admission",
                formal_admission=False,
                reference_tokenizer_sha256=sha256(root / "reference-tokenizer.json"),
            )
        )
    )
    (root / "source_allowlist.json").write_text(
        json.dumps(dict(sources={root.name: dict(repo=root.name, revision="pinned")}))
    )
    return root


def test_merge_preserves_sealed_aliases_and_excludes_the_whole_contaminated_repo(
    evaluation, tmp_path
):
    text = (
        "A telescope uses curved mirrors to collect faint light from distant astronomical objects."
    )
    a = candidate(
        tmp_path / "a",
        [
            ("1", "dirty1", PROMPT, "example/dirty"),
            (
                "2",
                "dirty2",
                "A completely different file describing the migration of forest birds.",
                "example/dirty",
            ),
            ("3", "clean", text, "example/clean"),
        ],
        locks={"source-group:a:clean": "test"},
    )
    b = candidate(
        tmp_path / "b",
        [
            ("1", "alias", text, "example/alias"),
            (
                "2",
                "alias",
                "The ocean current carries warm surface water toward the northern coastline.",
                "example/alias",
            ),
        ],
    )
    before = [sha256(p / "corpus.sqlite") for p in (a, b)]
    output = tmp_path / "merged"
    audit = merge_text_slices([a, b], output, evaluation)
    assert [sha256(p / "corpus.sqlite") for p in (a, b)] == before
    assert audit["contamination"]["direct_matches"] == 1
    assert audit["contamination"]["excluded_records"] == 2
    assert audit["dedup_and_quality_counts"]["accepted"] == 2
    assert not audit["formal_admission"]
    db = sqlite3.connect(output / "corpus.sqlite")
    assert db.execute("SELECT COUNT(*) FROM samples WHERE split='test'").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(DISTINCT group_root) FROM samples").fetchone()[0] == 1
    assert (
        db.execute(
            "SELECT COUNT(*) FROM links LEFT JOIN samples USING(id) WHERE samples.id IS NULL"
        ).fetchone()[0]
        == 0
    )
    db.close()


def test_benchmark_matcher_detects_normalized_overlap_but_not_short_boilerplate(evaluation):
    matcher = BenchmarkMatcher(evaluation)
    assert matcher.match(
        dict(text="Discussion:\n" + PROMPT.upper().replace(" ", "\n") + " Answer follows.")
    )
    words = PROMPT.split()
    assert matcher.match(
        dict(text="A different prefix " + " ".join(words[:13]) + " an altered ending.")
    )
    assert (
        matcher.match(dict(text="Unrelated text", repo_id="openai/human-eval"))["field"]
        == "repo_id"
    )
    assert (
        matcher.match(
            dict(text="How many trees were planted altogether? The expected answer is 51.")
        )
        is None
    )
    (evaluation / "items.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="path/hash"):
        BenchmarkMatcher(evaluation)


def test_merge_rejects_a_mutated_source_before_creating_output(evaluation, tmp_path):
    source = candidate(
        tmp_path / "source",
        [
            (
                "1",
                "g",
                "A greenhouse traps heat and protects the seedlings during cold winter nights.",
                "example/plants",
            )
        ],
    )
    with (source / "corpus.sqlite").open("ab") as stream:
        stream.write(b"unexpected")
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="database differs"):
        merge_text_slices([source], output, evaluation)
    assert not output.exists()


def test_jsonl_reader_preserves_unicode_separators_inside_a_string(evaluation):
    text = PROMPT.replace("apple", "apple\u2028").replace("pear", "pear\u0085")
    path = evaluation / "items.jsonl"
    path.write_text(
        json.dumps(dict(id="unicode:1", prompt=text, answer="51"), ensure_ascii=False) + "\n"
    )
    manifest = json.loads((evaluation / "manifest.json").read_text())
    manifest["files"][path.name] = sha256(path)
    (evaluation / "manifest.json").write_text(json.dumps(manifest))
    assert BenchmarkMatcher(evaluation).match(dict(text=PROMPT))["item_id"] == "unicode:1"


def test_download_checks_blob_before_persisting(monkeypatch, tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield b"incorrect bytes"

    monkeypatch.setattr("requests.get", lambda *a, **kw: Response())
    path = tmp_path / "raw"
    with pytest.raises(ValueError, match="Git blob hash"):
        _download("https://example.invalid", path, blob="0" * 40)
    assert not path.exists()
