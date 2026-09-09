"""Disk-bounded native token/label/position shards and per-record CE accounting."""

import json
from collections import Counter
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.models.minifrontier1.processing import token_metadata
from minifrontier.storage import require_space

from .data import RecordDataset, digest, write_json


def encode_dataset(data, output, config, *, max_gib=16):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or max_gib <= 0:
        raise ValueError("choose a new encoding version and positive disk budget")
    output.mkdir(parents=True)
    require_space(output, 16 * 1024**2)
    token_dtype = "uint16" if config.vocab_size <= 65536 else "uint32"
    splits = {}
    used_bytes = 0
    for split in ("train", "val", "test"):
        dataset = RecordDataset(data, split, config)
        counts: Counter[str] = Counter()
        unique_media = set()
        domains: Counter[str] = Counter()
        paths = {
            key: output / f"{split}.{key}"
            for key in ("ids.bin", "labels.bin", "positions.bin", "index.jsonl")
        }
        with (
            paths["ids.bin"].open("wb") as ids_file,
            paths["labels.bin"].open("wb") as labels_file,
            paths["positions.bin"].open("wb") as positions_file,
            paths["index.jsonl"].open("w") as index_file,
        ):
            for i in range(len(dataset)):
                item = dataset[i]
                metadata = token_metadata(item["input_ids"], config, item["media"])
                ids = item["input_ids"][0].numpy().astype(token_dtype)
                labels = item["labels"][0].numpy().astype("int32")
                positions = metadata["position_ids"][:, 0].T.numpy().astype("int32")
                spans = [
                    {
                        key: value.tolist() if hasattr(value, "tolist") else value
                        for key, value in span.items()
                        if key != "patches"
                    }
                    for span in item["media"]
                ]
                entry = dict(
                    sample_id=item["sample_id"],
                    split_group=item["split_group"],
                    domain=item["domain"],
                    offset=counts["input_tokens"],
                    input_tokens=len(ids),
                    ce_tokens=int((labels[1:] != -100).sum()),
                    vision_tokens=sum(s["feature_count"] for s in spans),
                    media=spans,
                    segment_ids=metadata["segment_ids"][0].tolist(),
                    linear_positions=metadata["linear_positions"][0].tolist(),
                    modality=metadata["modality"][0].tolist(),
                    media_ids=metadata["media_ids"][0].tolist(),
                    encoded_sha256=digest([ids.tolist(), labels.tolist(), positions.tolist()]),
                )
                raw = json.dumps(entry) + "\n"
                incoming = ids.nbytes + labels.nbytes + positions.nbytes + len(raw.encode())
                if used_bytes + incoming > max_gib * 1024**3:
                    raise ValueError(
                        "encoding reached its disk budget; incomplete shards remain unadmitted"
                    )
                require_space(output, incoming)
                ids.tofile(ids_file)
                labels.tofile(labels_file)
                positions.tofile(positions_file)
                index_file.write(raw)
                used_bytes += incoming
                counts.update(
                    records=1,
                    input_tokens=len(ids),
                    ce_tokens=entry["ce_tokens"],
                    vision_tokens=entry["vision_tokens"],
                    media_exposures=item["media_exposures"],
                )
                domains[item["domain"]] += entry["ce_tokens"]
                unique_media.update(item["media_hashes"])
        splits[split] = dict(
            counts=counts,
            domain_ce=dict(domains),
            unique_media=len(unique_media),
            files={k: dict(name=p.name, sha256=sha256(p)) for k, p in paths.items()},
        )
    result = dict(
        format="mf1-native-shards-v1",
        source_manifest_sha256=sha256(data / "manifest.json"),
        tokenizer_sha256=sha256(data / "tokenizer.json"),
        token_dtype=token_dtype,
        label_dtype="int32",
        position_dtype="int32",
        position_axes=["t", "h", "w"],
        bytes=used_bytes,
        splits=splits,
        formal_admission=False,
    )
    write_json(output / "manifest.json", result)
    return result
