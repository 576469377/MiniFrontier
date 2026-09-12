"""Multipage admission preserves provenance, both pages and complete responses."""

import copy
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from minifrontier.data import multipage_sources as source
from minifrontier.multimodal import pretraining_tokens


@pytest.fixture
def tokenizer(tmp_path):
    value = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    value.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    value.save(str(path))
    return value, path


def document():
    images = []
    for color in ("red", "blue"):
        value = io.BytesIO()
        Image.new("RGB", (80, 100), color).save(value, format="PNG")
        images.append(dict(bytes=value.getvalue(), path=None))
    return dict(
        images=images,
        texts=[
            dict(
                user="What is the title on these pages?",
                assistant="The title is A history of early astronomy.",
                source="PDFA key: 244",
            )
        ],
    )


def test_complete_two_pages_and_original_answer(tokenizer):
    row = document()
    result, reason = source.document_question(row, "file:rg0:row1", tokenizer[0])
    assert reason is None
    assert result["visual_answer"] == row["texts"][0]["assistant"]
    assert result["group_id"] == "PDFA key: 244"
    assert result["task"] == "multiimage_multipage"
    assert len(result["media"]) == 2
    ids, labels = pretraining_tokens(result, tokenizer[0])
    assert ids.count(7) == 2 and labels[0] == -100
    assert source.document_question(row, "file:rg0:row1", tokenizer[0])[0] == result


@pytest.mark.parametrize("pages", [1, 3, 4])
def test_never_drop_pages_to_make_a_shorter_example(tokenizer, pages):
    row = document()
    row["images"] = (row["images"] * 2)[:pages]
    assert source.document_question(row, "row", tokenizer[0])[1] == "not_exactly_two_pages"


def test_no_answer_truncation_or_ambiguous_pdf_group(tokenizer):
    row = document()
    assert source.document_question(row, "row", tokenizer[0], max_text_tokens=5)[0] is None
    row["texts"].append(dict(row["texts"][0], source="another PDF"))
    assert (
        source.document_question(row, "row", tokenizer[0])[1] == "missing_or_ambiguous_pdf_origin"
    )


def test_page_integrity_and_no_duplicate_page_disguised_as_multipage():
    row = document()
    pages = source.decode_pages(row)
    assert all(p[1]["width"] == 80 and p[1]["height"] == 100 for p in pages)
    assert pages[0][1]["rgb_sha256"] != pages[1][1]["rgb_sha256"]
    row["images"][1] = row["images"][0]
    with pytest.raises(ValueError, match="identical"):
        source.decode_pages(row)


def test_bounded_build_pins_source_and_cannot_overwrite(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    def rows(name, **kwargs):
        assert kwargs["specification"]["revision"] == source.SOURCE["revision"]
        assert kwargs["specification"]["network_byte_budget"] > 0
        for index in range(4):
            row = copy.deepcopy(document())
            row["texts"][0]["source"] = f"PDF {index}"
            row["texts"][0]["assistant"] += f" Volume {index}."
            yield row, str(index)

    monkeypatch.setattr(source, "source_rows", rows)
    out = tmp_path / "corpus"
    report = source.build_multipage(out, tokenizer[1], documents=2)
    assert report["producer_finished"] and report["counts"]["source_rows"] == 2
    assert sum(report["split_records"].values()) == 2
    assert report["stored_images"] == 2  # Two QAs reuse two pages, not four unique pages.
    assert len(list((out / "images").iterdir())) == 2
    before = (out / "source-audit.json").read_bytes()
    with pytest.raises(FileExistsError):
        source.build_multipage(out, tokenizer[1], documents=3)
    assert (out / "source-audit.json").read_bytes() == before
    assert json.loads((out / "corpus-manifest.json").read_text())["image_phash_grouping"] is False
