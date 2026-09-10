"""Relocate encoded native media paths, preserving tokens and recording changed hashes."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from minifrontier.data import sha256


def relocate(root, old_workspace, new_workspace, report_path):
    root, report_path = Path(root), Path(report_path)
    if report_path.exists():
        raise FileExistsError("preserve the existing relocation receipt")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = dict(
        original_manifest=copy.deepcopy(manifest), before_sha256=sha256(manifest_path), files=[]
    )
    updates = []

    def translate(value):
        if isinstance(value, str) and value.startswith(str(old_workspace).rstrip("/") + "/"):
            return str(new_workspace).rstrip("/") + value[len(str(old_workspace).rstrip("/")) :]
        if isinstance(value, list):
            return [translate(item) for item in value]
        if isinstance(value, dict):
            return {k: translate(v) for k, v in value.items()}
        return value

    for splits in manifest["stages"].values():
        for split in splits.values():
            media = split["media"]
            path, index_path = root / media["file"], root / media["index_file"]
            if sha256(path) != media["sha256"] or sha256(index_path) != media["index_sha256"]:
                raise ValueError("source media/index checksum mismatch")
            index = np.load(index_path)
            lines = path.read_bytes().splitlines(keepends=True)
            if len(lines) != len(index):
                raise ValueError("media row count differs from index")
            content = bytearray()
            new_index = index.copy()
            for i, (line, (offset, size, _)) in enumerate(zip(lines, index, strict=True)):
                if int(offset) != sum(len(x) for x in lines[:i]) or int(size) != len(line):
                    raise ValueError("source media index does not describe the original rows")
                row = json.loads(line)
                before = copy.deepcopy(row)
                row["record"]["media"] = translate(row["record"]["media"])
                # Only resource references change. IDs, labels, sample identity,
                # text and precomputed feature counts remain byte-for-byte values.
                serialized = (
                    line if row == before else json.dumps(row, ensure_ascii=False).encode() + b"\n"
                )
                new_index[i, :2] = len(content), len(serialized)
                content.extend(serialized)
            updates.append((path, index_path, bytes(content), new_index, media))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, index_path, payload, new_index, media in updates:
        record = dict(
            file=path.name, before_sha256=media["sha256"], index_before_sha256=media["index_sha256"]
        )
        path.write_bytes(payload)
        np.save(index_path, new_index)
        media.update(sha256=sha256(path), index_sha256=sha256(index_path))
        record.update(after_sha256=media["sha256"], index_after_sha256=media["index_sha256"])
        report["files"].append(record)
    manifest["media_root"] = translate(manifest["media_root"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    report.update(
        after_sha256=sha256(manifest_path),
        old_workspace=str(old_workspace),
        new_workspace=str(new_workspace),
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--old-workspace", required=True)
    parser.add_argument("--new-workspace", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    relocate(args.data, args.old_workspace, args.new_workspace, args.report)


if __name__ == "__main__":
    main()
