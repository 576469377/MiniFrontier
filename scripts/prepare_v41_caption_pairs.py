"""Freeze one caption per unique RGB image from an existing split native component."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from minifrontier.data.text import sha256


def prepare(source, media_root, output):
    source, media_root, output = map(Path, (source, media_root, output))
    source_manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("caption output must be empty")
    if sha256(source / "tokenizer.json") != source_manifest["tokenizer"]["sha256"]:
        raise ValueError("unbound source tokenizer")
    splits, identities = {}, {}
    for split in ("train", "val", "test"):
        media = source_manifest["stages"]["pretrain"][split]["media"]
        refs = source / media["file"]
        if sha256(refs) != media["sha256"]:
            raise ValueError(f"source native {split} changed")
        seen, groups = set(), Counter()
        rows = Counter()
        target = output / f"{split}.jsonl"
        with refs.open() as fp, target.open("x") as out:
            for line in fp:
                wrapper = json.loads(line)
                r = wrapper["record"]
                identity = [str(r["group_id"]), str(wrapper["split_group"])]
                identity += [m["rgb_sha256"] for m in r["media"] if m.get("kind") == "image"]
                for key in identity:
                    if identities.setdefault(key, split) != split:
                        raise ValueError("image/group appears across native splits")
                if r["task"] != "caption" or len(r["media"]) != 1:
                    continue
                item = r["media"][0]
                caption = r["visual_answer"].strip()
                rgb = item["rgb_sha256"]
                if item["kind"] != "image" or not caption or rgb in seen:
                    continue
                seen.add(rgb)
                path = (media_root / item["path"]).resolve()
                if not path.is_relative_to(media_root.resolve()) or not path.is_file():
                    raise ValueError(f"missing local caption image {item['path']}")
                # Verify the bytes referenced by each published image even before first training.
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"caption image checksum mismatch {item['path']}")
                out.write(
                    json.dumps(
                        dict(
                            sample_id=r["sample_id"],
                            group_id=r["group_id"],
                            rgb=rgb,
                            file=item["path"],
                            sha256=item["sha256"],
                            caption=caption,
                            source=r["source"],
                            revision=r["revision"],
                            license=r["license"],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                groups.update([r["source"]])
                rows.update(r.get("quality_flags", []))
        splits[split] = dict(
            file=target.name,
            sha256=sha256(target),
            images=len(seen),
            sources=dict(groups),
            source_quality_flags=dict(rows),
            native_media_sha256=media["sha256"],
        )
    shutil.copy2(source / "tokenizer.json", output / "tokenizer.json")
    manifest = dict(
        schema="v41-caption-pairs-v1",
        media_root=str(media_root.resolve()),
        native_manifest_sha256=sha256(source / "manifest.json"),
        tokenizer_sha256=source_manifest["tokenizer"]["sha256"],
        splits=splits,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "media-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.media_root, args.output), indent=2))


if __name__ == "__main__":
    main()
