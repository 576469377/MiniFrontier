"""MF1.1 data uses its own processor while retaining the existing storage contract."""

from collections import Counter
from types import SimpleNamespace

import pytest
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import make_fixture, prepare_media, processing_for
from minifrontier.data.minifrontier1_encoding import CompactDataset, encode_dataset
from minifrontier.models.minifrontier1 import MiniFrontier1Config
from minifrontier.models.minifrontier1 import processing as old_processing
from minifrontier.models.minifrontier11 import MiniFrontier11Config
from minifrontier.models.minifrontier11 import processing as new_processing


def forbid_old_processing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("MF1.1 data called MF1.0 model processing")

    calls = Counter()
    for name in ("process_frames", "process_document", "token_metadata"):
        source = getattr(new_processing, name)

        def tracked(*args, _name=name, _source=source, **kwargs):
            calls[_name] += 1
            return _source(*args, **kwargs)

        monkeypatch.setattr(old_processing, name, forbidden)
        monkeypatch.setattr(new_processing, name, tracked)
    return calls


def test_processor_selection_is_version_specific_and_rejects_unknown_versions():
    assert processing_for(MiniFrontier1Config.tiny()) is old_processing
    assert processing_for(MiniFrontier11Config.tiny()) is new_processing
    with pytest.raises(ValueError, match="unknown MF model version"):
        processing_for(SimpleNamespace(model_version="1.2-unimplemented"))


@pytest.mark.parametrize("compact", [False, True])
def test_mf11_encoding_and_binary_media_loading_do_not_use_mf1_processing(
    tmp_path, monkeypatch, compact
):
    monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
    source, encoded = tmp_path / "source", tmp_path / "encoded"
    make_fixture(source)
    calls = forbid_old_processing(monkeypatch)
    config = MiniFrontier11Config.tiny()
    manifest = encode_dataset(source, encoded, config, compact=compact, shard_tokens=512)
    assert calls["process_frames"] > 0
    if compact:
        assert manifest["processor_version"] == "mf1-native-v1"
        calls.clear()
        dataset = CompactDataset(encoded, "train", config)
        for index in (32, 64):
            assert dataset[index]["media"]
        assert calls["process_frames"] > 0
    else:
        assert calls["token_metadata"] > 0


def test_document_transform_uses_mf11_local_frame_processor(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    Image.new("RGB", (64, 32), "white").save(image)
    resource = dict(uri=image.name, sha256=sha256(image), representation="document")
    calls = forbid_old_processing(monkeypatch)
    samples = prepare_media(resource, MiniFrontier11Config(), tmp_path, 1024)
    assert len(samples) >= 2 and samples[0]["tile_id"] == "global"
    assert calls["process_document"] == 1 and calls["process_frames"] == len(samples)
