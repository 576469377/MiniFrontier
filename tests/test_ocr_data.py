import contextlib
import hashlib
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.corpus import CorpusBuilder, encode_corpus, train_tokenizer
from minifrontier.data.encoding_audit import audit_image_encoding
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS
from minifrontier.data.minifrontier1_components import assemble_components
from minifrontier.data.minifrontier1_encoding import encode_canonical_images, encode_canonical_text
from minifrontier.data.native import audit_native_encoding, encode_native
from minifrontier.data.ocr import Renderer, generate_ocr
from minifrontier.data.partitions import open_corpus
from minifrontier.models.minifrontier1 import MiniFrontier1Config


@pytest.fixture
def fonts(tmp_path):
    root = tmp_path / "fonts"
    root.mkdir()
    assets = {}
    for key, name in [
        ("sans", "DejaVuSans.ttf"),
        ("serif", "DejaVuSerif.ttf"),
        ("mono", "DejaVuSansMono.ttf"),
    ]:
        source = Path("/usr/share/fonts/truetype/dejavu") / name
        if not source.exists():
            pytest.skip("CPU OCR fixture needs system DejaVu fonts")
        shutil.copyfile(source, root / name)
        assets[key] = dict(path=name, sha256=sha256(root / name))
    (root / "license.txt").write_text("Fixture license record")
    assets["license"] = dict(path="license.txt", sha256=sha256(root / "license.txt"))
    p = root / "manifest.json"
    p.write_text(
        json.dumps(
            dict(
                assets=assets,
                split_fonts={"train": [["sans"]], "val": [["serif"]], "test": [["mono"]]},
                license_map={k: "license" for k in ("sans", "serif", "mono")},
            )
        )
    )
    return p


def test_render_targets_are_exact_visible_complete_lines_and_deterministic(fonts):
    renderer = Renderer(fonts)
    text = (
        "The telescope measures distant stars and galaxies with a carefully calibrated detector. "
        * 12
    )
    for split in ("train", "val", "test"):
        a, answer, layout = renderer.render(text, "fixture", split, "en", 42)
        b, again, second = renderer.render(text, "fixture", split, "en", 42)
        assert a.size == (224, 224) and a.tobytes() == b.tobytes()
        assert (answer, layout) == (again, second)
        assert "".join(answer.split()) in "".join(text.split())
        assert len(answer.splitlines()) == len(layout["line_boxes"])
        for left, top, right, bottom in layout["line_boxes"]:
            assert 0 <= left < right < 224 and 0 <= top < bottom < 224
            patch = a.crop((int(left), int(top), int(right) + 1, int(bottom) + 1))
            assert len(patch.getcolors(224 * 224)) > 1
    with pytest.raises(ValueError, match="unsupported_glyph"):
        renderer.render("\U0010ffff" * 500, "bad", "train", "en", 42)
    spec = json.loads(fonts.read_text())
    spec["assets"]["serif"] = spec["assets"]["sans"]
    fonts.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="disjoint"):
        Renderer(fonts)


def test_source_partition_and_pixels_survive_generation_and_actual_mf1_encoding(
    fonts, tmp_path, monkeypatch
):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=900 * 1024**3)
    )
    root = tmp_path / "text"
    builder = CorpusBuilder(root)
    for i in range(36):
        rng = random.Random(i)
        words = [
            "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(3, 9)))
            for _ in range(160)
        ]
        assert builder.add(
            dict(
                source="fixture",
                revision="pinned",
                item_id=str(i),
                group_id=str(i),
                license="fixture",
                lang="en",
                task="en_edu",
                stage="pretrain",
                text=" ".join(words),
            )
        )
    builder.finalize(
        split_locks={f"source-group:fixture:{i}": "test" if i % 2 else "val" for i in range(12)}
    )
    builder.db.close()
    (root / "source-audit.json").write_text(
        json.dumps(
            dict(status="candidate_slice_complete_pending_admission", formal_admission=False)
        )
    )
    token = tmp_path / "tokenizer.json"
    train_tokenizer(root, token, 400, special_tokens=SPECIAL_TOKENS)
    out = tmp_path / "ocr"
    audit = generate_ocr(root, fonts, token, out, id_stop="g", max_gib=0.04, metadata_gib=0.02)
    assert audit["status"] == "candidate_slice_complete_pending_admission"
    assert not audit["formal_admission"] and not audit["main_budget_eligible"]
    assert audit["split_origin_mismatches"] == 0 and audit["independent_text_supply_added"] == 0
    renderer = Renderer(fonts)
    seen = set()
    with (
        contextlib.closing(open_corpus(root)) as source,
        contextlib.closing(open_corpus(out)) as generated,
    ):
        for payload, split in generated.execute("SELECT payload,split FROM samples"):
            row = json.loads(payload)
            origin = row["text_origin"]
            seen.add(split)
            original, source_split = source.execute(
                "SELECT payload,split FROM samples WHERE id=?", (origin["sample_id"],)
            ).fetchone()
            assert split == source_split == origin["split"]
            image, answer, layout = renderer.render(
                json.loads(original)["text"], origin["sample_id"], split, "en", audit["seed"]
            )
            media = row["media"][0]
            saved = Image.open(out / media["path"]).convert("RGB")
            assert (
                image.tobytes() == saved.tobytes()
                and sha256(out / media["path"]) == media["sha256"]
            )
            assert row["visual_answer"] == answer and row["rendering"] == layout
            assert (
                layout["normalized_source_sha256"]
                == hashlib.sha256(
                    " ".join(json.loads(original)["text"].split()).encode()
                ).hexdigest()
            )
    assert seen == {"train", "val", "test"}
    config = MiniFrontier1Config(
        **dict(asdict(MiniFrontier1Config.tiny(400)), max_position_embeddings=1024)
    )
    encoded = tmp_path / "encoded"
    manifest = encode_canonical_images(out, token, encoded, config, max_features=49)
    proof = audit_image_encoding(out, encoded, tmp_path / "audit.json", asdict(config))
    assert proof["status"] == "mechanical_checks_passed_pending_quality_admission"
    for node in manifest["splits"].values():
        assert node["counts"]["vision_tokens"] == 49 * node["counts"]["records"]
    text = tmp_path / "compact-text"
    encode_canonical_text(root, token, text, config)
    assemble_components([text, encoded], tmp_path / "joint", config)
    shared = tmp_path / "source-text"
    encode_corpus(root, token, shared, max_length=1024)
    for family in ("minikimik3", "miniqwen4"):
        native = tmp_path / family
        encode_native(
            out,
            token,
            native,
            family,
            text_encoding=shared,
            max_features=49,
            max_length=1024,
            min_pixels=224 * 224 if family == "miniqwen4" else None,
        )
        assert (
            audit_native_encoding(out, native, tmp_path / (family + "-audit.json"))["status"]
            == proof["status"]
        )
    # A new text pool cannot promote OCR-source holdouts into text training.
    wrong = tmp_path / "wrong-text"
    shutil.copytree(shared, wrong)
    wrong_manifest = json.loads((wrong / "manifest.json").read_text())
    splits = wrong_manifest["stages"]["pretrain"]
    splits["train"], splits["val"] = splits["val"], splits["train"]
    (wrong / "manifest.json").write_text(json.dumps(wrong_manifest))
    with pytest.raises(ValueError, match="crosses shared text"):
        encode_native(
            out,
            token,
            tmp_path / "leaking-native",
            "minikimik3",
            text_encoding=wrong,
            max_length=1024,
        )
    with contextlib.closing(open_corpus(root)) as db:
        parent_val = db.execute("SELECT id FROM samples WHERE split='val' LIMIT 1").fetchone()[0]
    part = manifest["splits"]["train"]["parts"][0]["files"]["metadata.jsonl"]
    metadata = encoded / part["name"]
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    rows[0]["origin"]["text_origin"]["sample_id"] = parent_val
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    part.update(bytes=metadata.stat().st_size, sha256=sha256(metadata))
    (encoded / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source text identity crosses"):
        assemble_components([text, encoded], tmp_path / "leaking-joint", config)
    with pytest.raises(ValueError, match="new OCR shard"):
        generate_ocr(root, fonts, token, out, id_stop="g")
