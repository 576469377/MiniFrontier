import json

import numpy as np
import pytest

from minifrontier.data import sha256
from scripts.relocate_native_media import relocate


def test_media_relocation_rebuilds_byte_offsets_and_keeps_token_stream(tmp_path):
    rows = [
        dict(
            record=dict(
                media=[dict(path=f"/old/images/{i}.png")], text="相同文本", content_hash="same"
            ),
            expected_ids=[1, 2, 3],
            expected_labels=[-100, 2, 3],
        )
        for i in range(2)
    ]
    lines = [(json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows]
    path, index = tmp_path / "media.jsonl", tmp_path / "media.index.npy"
    path.write_bytes(b"".join(lines))
    np.save(index, np.array([[0, len(lines[0]), 3], [len(lines[0]), len(lines[1]), 3]]))
    media = dict(
        file=path.name, index_file=index.name, sha256=sha256(path), index_sha256=sha256(index)
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(dict(media_root="/old", stages=dict(pretrain=dict(train=dict(media=media)))))
    )
    report = tmp_path / "receipt.json"
    relocate(tmp_path, "/old", "/new/longer", report)
    with path.open("rb") as stream:
        for i, (offset, size, length) in enumerate(np.load(index)):
            stream.seek(int(offset))
            row = json.loads(stream.read(int(size)))
            assert row["record"]["media"][0]["path"] == f"/new/longer/images/{i}.png"
            row["record"]["media"] = rows[i]["record"]["media"]
            assert row == rows[i]
            assert length == 3
    receipt = json.loads(report.read_text())
    assert receipt["original_manifest"]["stages"]["pretrain"]["train"]["media"] == media
    assert receipt["files"][0]["after_sha256"] == sha256(path)
    with pytest.raises(FileExistsError):
        relocate(tmp_path, "/old", "/new/longer", report)
