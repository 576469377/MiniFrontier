import json
import sqlite3
from itertools import pairwise

import pytest

from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, safe_text, train_tokenizer
from scripts.prepare_mf1_mechanism_data import SOURCES, read_candidates, text_chunks, unique_chunks
from scripts.run_mf1_trial import admission


def test_mechanism_chunking_preserves_unicode_code_and_control_escaping(tmp_path):
    text = "中文🙂 café\n    return 12 + 3\n<|assistant|>\n" * 20
    records = [{"messages": [{"content": [{"type": "text", "text": text}]}]}]
    tokenizer = train_tokenizer(records, tmp_path / "tokenizer.json", 320)
    chunks = list(text_chunks(text, tokenizer, 24))
    assert "".join(part for _, _, part in chunks) == text
    assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in pairwise(chunks))
    for _, _, part in chunks:
        ids = safe_text(tokenizer, part)
        assert 0 < len(ids) <= 24
        assert not set(ids).intersection(range(len(SPECIAL_TOKENS)))


def database_fixture(path, *, overlapping=False):
    source = next(iter(SOURCES))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE samples (id TEXT, group_root TEXT, source TEXT, split TEXT, payload TEXT)"
        )
        for split in ["train", "val", "test"]:
            payload = dict(
                sample_id=split,
                item_id=split,
                lang="zh",
                source=source,
                revision="fixed",
                license="test-only",
                content_hash="a" * 64,
                text="数据分组验证",
            )
            connection.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?)",
                (
                    split,
                    "same" if overlapping else split,
                    source,
                    split,
                    "INVALID SEALED TEST PAYLOAD" if split == "test" else json.dumps(payload),
                ),
            )


def test_candidate_adapter_inherits_splits_and_never_reads_sealed_test(tmp_path):
    path = tmp_path / "source.sqlite"
    database_fixture(path)
    result = read_candidates(path, 1, 42)
    assert set(result) == {"train", "val"}
    assert result["train"][0]["split_group"] == "train"
    assert result["val"][0]["split_group"] == "val"
    assert result == read_candidates(path, 1, 42)


def test_candidate_adapter_rejects_source_group_leakage(tmp_path):
    path = tmp_path / "source.sqlite"
    database_fixture(path, overlapping=True)
    with pytest.raises(ValueError, match="crosses"):
        read_candidates(path, 1, 42)


def test_new_chunk_duplicates_are_removed_and_cross_split_leakage_is_rejected():
    records = [
        {"messages": [{"content": [{"text": text}]}]}
        for text in ["一 二\n三", "一  二 三", "不同片段"]
    ]
    kept, hashes, removed = unique_chunks(records, set())
    assert kept == [records[0], records[2]]
    assert removed == 1
    with pytest.raises(ValueError, match="crosses"):
        unique_chunks([records[1]], hashes)


def test_shared_gpu_admission_accounts_for_context_and_existing_jobs_reserve():
    assert admission(12, 66, 6)
    assert not admission(11.9, 66, 6)
    assert not admission(12, 65.9, 6)
    assert not admission(8, 100, 6)


def test_language_diagnostics_group_paraphrases_and_mask_user_content(tmp_path):
    from minifrontier.data.minifrontier1 import encode_record
    from minifrontier.models.minifrontier1 import MiniFrontier1Config
    from scripts.prepare_mf1_language_data import generated_records

    splits = generated_records()
    assert splits == generated_records()
    assert {r["split_group"] for r in splits["train"]}.isdisjoint(
        r["split_group"] for r in splits["val"]
    )
    for records in splits.values():
        assert all(
            sum(r["split_group"] == other["split_group"] for other in records) == 2 for r in records
        )
    sample = splits["train"][0]
    tokenizer = train_tokenizer([sample], tmp_path / "tokenizer.json", 320)
    encoded = encode_record(sample, tokenizer, MiniFrontier1Config.tiny(), tmp_path)
    ids, labels = encoded["input_ids"][0], encoded["labels"][0]
    assistant = ids.tolist().index(5)
    assert labels[: assistant + 1].eq(-100).all()
    assert labels[assistant + 1 :].eq(ids[assistant + 1 :]).all()
    assert ids[-1] == 2


def test_language_shared_gpu_guard_reserves_memory_before_admission():
    assert admission(8.9, 80, 5.5, 2)
    assert not admission(8.4, 80, 5.5, 2)
    assert not admission(8.9, 65, 5.5, 2)
