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
from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.encoding_audit import audit_image_encoding
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS
from minifrontier.data.minifrontier1_encoding import encode_canonical_images
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
    with pytest.raises(ValueError, match="new OCR shard"):
        generate_ocr(root, fonts, token, out, id_stop="g")
