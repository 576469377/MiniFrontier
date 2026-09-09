"""Small pinned ALLaVA-in-FineVision pilot with original bytes and decoded hashes."""

import io
import json
import random
from pathlib import Path

from PIL import Image

from minifrontier.data.corpus import CorpusBuilder
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.data.remote import RangeFile
from minifrontier.storage import GIB, require_space, reserve_write

REPO = "HuggingFaceM4/FineVision"
REVISION = "3c380a731a3429c1d04693d6ec16d7e683def84c"


def build_visual_pilot(output, *, images=96, seed=137, max_gib=4):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("visual corpus is immutable; choose a new version")
    if not 1 <= images <= 4096:
        raise ValueError("this bounded pilot admits 1-4096 images")
    require_space(root, max_gib * GIB)
    builder = CorpusBuilder(root, seed=seed, max_gib=max_gib / 2)
    (root / "images").mkdir()
    files = [
        item
        for item in HfApi().list_repo_tree(
            REPO, path_in_repo="allava_laion", revision=REVISION, repo_type="dataset"
        )
        if isinstance(item, RepoFile) and item.path.endswith(".parquet")
    ]
    rng = random.Random(seed)
    rng.shuffle(files)
    audit = dict(
        source=REPO,
        revision=REVISION,
        subset="allava_laion",
        license="CC-BY-NC-4.0",
        upstream="FreedomIntelligence/ALLaVA-4V@0fd42fce5c047d387a4bb5318d588eae9a9797f0",
        main_budget_eligible=False,
        reason="small pilot; formal split/source review incomplete",
        seed=seed,
        unique_images=0,
        media_bytes=0,
        reads=[],
        status="building",
    )
    seen = set()
    try:
        for file in files:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{file.path}"
            with RangeFile(url, file.size, chunk_size=1024**2, network_budget=2 * GIB) as source:
                parquet = pq.ParquetFile(source)
                groups = list(range(parquet.num_row_groups))
                rng.shuffle(groups)
                for group in groups:
                    if parquet.metadata.row_group(group).total_byte_size > 512 * 1024**2:
                        raise ValueError("visual row group exceeds 512 MiB memory cap")
                    rows = parquet.read_row_group(group).to_pylist()
                    order = list(range(len(rows)))
                    rng.shuffle(order)
                    audit["reads"].append(dict(file=file.path, group=group, rows=len(rows)))
                    for index in order:
                        row = rows[index]
                        if (
                            row["source"] != "allava_laion"
                            or len(row["images"]) != 1
                            or not row["texts"]
                        ):
                            continue
                        binary = row["images"][0]["bytes"]
                        if not binary or len(binary) > 16 * 1024**2:
                            continue
                        with Image.open(io.BytesIO(binary)) as image:
                            if image.width * image.height > 20_000_000:
                                continue
                            hashes = decoded_hashes(image)
                        if hashes["rgb_sha256"] in seen:
                            continue
                        caption = row["texts"][0]
                        # Later QA turns are not silently relabeled as captions.
                        if not any(
                            word in caption["user"].lower()
                            for word in (
                                "describ",
                                "description",
                                "descriptive",
                                "elaborate",
                                "details",
                            )
                        ):
                            continue
                        media_path = root / "images" / (hashes["rgb_sha256"] + ".image")
                        if audit["media_bytes"] + len(binary) > max_gib * GIB // 2:
                            raise ValueError("visual media byte budget reached")
                        with reserve_write(media_path, len(binary)):
                            media_path.write_bytes(binary)
                        audit["media_bytes"] += len(binary)
                        resource = dict(
                            path=str(media_path.relative_to(root)), kind="image", **hashes
                        )
                        common = dict(
                            source=REPO + "/allava_laion",
                            revision=REVISION,
                            item_id=f"{file.path}:rg{group}:row{index}",
                            group_id=hashes["rgb_sha256"],
                            license="CC-BY-NC-4.0; ALLaVA underlying LAION-image rights retained",
                            lang="en",
                            media=[resource],
                            original_image_path=row["images"][0].get("path"),
                            quality_flags=["pilot-only", "source-split-review-pending"],
                            upstream_source=audit["upstream"],
                        )
                        accepted = builder.add(
                            dict(
                                common,
                                stage="pretrain",
                                task="caption",
                                text="<|image|> " + caption["assistant"],
                            )
                        )
                        for turn in row["texts"][:3]:
                            builder.add(
                                dict(
                                    common,
                                    stage="sft",
                                    task="caption",
                                    turns=[
                                        dict(
                                            role="user",
                                            content="<|image|>\n" + turn["user"].strip(),
                                        ),
                                        dict(role="assistant", content=turn["assistant"].strip()),
                                    ],
                                )
                            )
                        if accepted:
                            seen.add(hashes["rgb_sha256"])
                            audit["unique_images"] = len(seen)
                        if len(seen) >= images:
                            break
                    (root / "source-audit.json").write_text(json.dumps(audit, indent=2))
                    print(
                        json.dumps(dict(images=len(seen), downloaded_bytes=source.transferred)),
                        flush=True,
                    )
                    if len(seen) >= images:
                        break
            if len(seen) >= images:
                break
        audit["corpus"] = builder.finalize()
        audit["status"] = "complete"
    except BaseException as error:
        builder.db.commit()
        audit.update(status="interrupted", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        (root / "source-audit.json").write_text(json.dumps(audit, indent=2))
    return audit
