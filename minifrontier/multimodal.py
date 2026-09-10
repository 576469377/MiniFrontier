"""One native media/template path shared by corpus encoding, training and inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from minifrontier.data import chat_tokens, sha256
from minifrontier.data.media_hash import decoded_hashes


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


def select_media(media, indices):
    """Remap complete image spans for a subset, including repeated rollout prompts."""
    indices = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
    return [
        dict(span, batch_index=new_index)
        for new_index, old_index in enumerate(indices)
        for span in media
        if span["batch_index"] == old_index
    ]


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    extras: dict = field(default_factory=dict)
    image_count: int = 0
    video_count: int = 0
    frame_count: int = 0
    image_features: int = 0

    def __iter__(self):
        yield self.input_ids
        yield self.labels

    def to(self, device):
        return TrainingBatch(
            self.input_ids.to(device),
            self.labels.to(device),
            move(self.extras, device),
            self.image_count,
            self.video_count,
            self.frame_count,
            self.image_features,
        )


def pretraining_tokens(record, tokenizer):
    """Keep raw text literal; only schema-declared media become structural image IDs."""
    resources = record.get("media", [])
    if "visual_question" in record or "visual_answer" in record:
        if not resources or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in ("visual_question", "visual_answer")
        ):
            raise ValueError(
                "canonical visual pretraining requires complete question/answer and media"
            )
        # Same PT text objective as the canonical serialized document, including
        # the question. This does not apply the assistant-only SFT objective.
        chunks = [""] * len(resources) + [
            "\nUser: " + record["visual_question"] + "\nAssistant: " + record["visual_answer"]
        ]
    elif resources:
        chunks = record["text"].split("<|image|>")
        if len(chunks) == 1:
            chunks = [""] * len(resources) + chunks
        if len(chunks) != len(resources) + 1:
            raise ValueError("placeholder/resource count mismatch")
    else:
        chunks = [record["text"]]
    previous = tokenizer.encode_special_tokens
    tokenizer.encode_special_tokens = True
    try:
        ids = [1]
        for index, chunk in enumerate(chunks):
            if index:
                ids.append(7)
            ids.extend(tokenizer.encode(chunk, add_special_tokens=False).ids)
        ids.append(2)
        return ids, [-100, *ids[1:]]
    finally:
        tokenizer.encode_special_tokens = previous


def prepare_record(
    record,
    tokenizer,
    family,
    *,
    root=None,
    max_features=256,
    generation_prompt=False,
    model_vocab_size=None,
    min_pixels=None,
):
    if tokenizer.token_to_id("<|image|>") != 7:
        raise ValueError(
            "native media requires the frozen strategy tokenizer special-token mapping"
        )
    if record.get("stage") == "pretrain":
        ids, labels = pretraining_tokens(record, tokenizer)
    else:
        ids, labels = chat_tokens(
            record["turns"],
            tokenizer,
            generation_prompt=generation_prompt,
            mode=record.get("mode"),
            effort=record.get("effort"),
        )
    resources = record.get("media", [])
    if ids.count(7) != len(resources):
        raise ValueError("placeholder/resource count mismatch")
    output, targets = [], []
    media: list[dict[str, Any]] = []
    image_count = video_count = frame_count = features = 0
    for token, target in zip(ids, labels, strict=True):
        if token != 7:
            output.append(token)
            targets.append(target)
            continue
        resource = resources[len(media)]
        frame_paths = resource.get("frames", [resource.get("path")])
        if any(path is None for path in frame_paths):
            raise ValueError("media requires locally decoded image or frame paths")
        images = []
        raw_hashes = resource.get("frame_sha256", [resource.get("sha256")] * len(frame_paths))
        for path, expected_hash in zip(frame_paths, raw_hashes, strict=True):
            resolved = Path(root or ".") / path
            if expected_hash and sha256(resolved) != expected_hash:
                raise ValueError("raw media hash differs from immutable corpus record")
            with Image.open(resolved) as image:
                images.append(image.copy())
        digest = decoded_hashes(images[0])
        if resource.get("rgb_sha256") and digest["rgb_sha256"] != resource["rgb_sha256"]:
            raise ValueError("decoded media hash differs from immutable corpus record")
        video = resource.get("kind") == "video"
        if video:
            expected = resource.get("frame_rgb_sha256")
            actual = [decoded_hashes(frame)["rgb_sha256"] for frame in images]
            if expected != actual:
                raise ValueError("every video frame needs an immutable decoded RGB hash")
        if video and (not resource.get("timestamps") or len(images) < 2):
            raise ValueError("video requires at least two genuine timestamped source frames")
        if not video and len(images) != 1:
            raise ValueError("multiple frames must be explicitly identified as video")
        if family == "minideepseekv4":
            if video:
                raise ValueError("Vision-v1 image route does not claim video support")
            from minifrontier.models.minideepseekv4.processing import process_image

            sample = process_image(
                images[0],
                start=len(output),
                max_features=max_features,
                min_pixels=resource.get("min_pixels", 3136) if min_pixels is None else min_pixels,
            )
            if model_vocab_size is None:
                raise ValueError("DeepSeek sentinel layout requires the model vocabulary size")
            tokens = (model_vocab_size + sample["types"]).tolist()
        elif family == "minikimik3":
            from minifrontier.models.minikimik3.processing import process_frames as kimi_frames

            # More than four frames form multiple native temporal groups. Keep
            # every group's spatial metadata and concatenate only its own features.
            groups = [
                kimi_frames(
                    images[i : i + 4],
                    max_features=max_features,
                    timestamps=resource["timestamps"][i : i + 4] if video else None,
                )
                for i in range(0, len(images), 4)
            ]
            sample = dict(
                patches=torch.cat([g["patches"] for g in groups]),
                grid_thw=torch.cat([g["grid_thw"] for g in groups]),
                feature_count=sum(g["feature_count"] for g in groups),
            )
            prefix = [
                9,
                *tokenizer.encode(
                    f"{'video' if video else 'image'} {images[0].width}x{images[0].height}"
                ).ids,
                11,
            ]
            output.extend(prefix)
            targets.extend([-100] * len(prefix))
            tokens = [7] * sample["feature_count"]
            sample["start"] = len(output)
        elif family == "miniqwen4":
            from minifrontier.models.miniqwen4.processing import process_frames as qwen_frames

            sample = qwen_frames(
                images,
                max_features=max_features,
                min_pixels=resource.get("min_pixels", 65536) if min_pixels is None else min_pixels,
                timestamps=resource.get("timestamps") if video else None,
            )
            output.append(9)
            targets.append(-100)
            sample["start"] = len(output)
            tokens = [7] * sample["feature_count"]
        else:
            raise ValueError("unknown native media family")
        sample["batch_index"] = 0
        sample["resource_kind"] = "video" if video else "image"
        sample["source_frame_count"] = len(images) if video else 0
        media.append(sample)
        output.extend(tokens)
        targets.extend([-100] * len(tokens))
        if family != "minideepseekv4":
            output.append(10)
            targets.append(-100)
        features += sample["feature_count"]
        image_count += 0 if video else 1
        video_count += int(video)
        frame_count += len(images) if video else 0
    x, y = torch.tensor([output]), torch.tensor([targets])
    extras: dict[str, Any] = dict(media=media) if media else {}
    if media and family == "miniqwen4":
        from minifrontier.models.miniqwen4.processing import position_ids

        extras["position_ids"] = position_ids(x, media)
        extras["ple_input_ids"] = x.clone()
    return TrainingBatch(x, y, extras, image_count, video_count, frame_count, features)


def collate(rows, device):
    if not isinstance(rows[0], TrainingBatch):
        values = [torch.stack([r[j] for r in rows]).to(device) for j in (0, 1)]
        return TrainingBatch(values[0], values[1])
    length = max(row.input_ids.shape[1] for row in rows)
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    media: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        n = row.input_ids.shape[1]
        x[index, :n], y[index, :n] = row.input_ids[0], row.labels[0]
        media.extend(dict(span, batch_index=index) for span in row.extras.get("media", []))
    extras: dict[str, Any] = dict(media=media) if media else {}
    if any("position_ids" in r.extras for r in rows):
        from minifrontier.models.miniqwen4.processing import position_ids

        extras.update(position_ids=position_ids(x, media), ple_input_ids=x.clone())
    return TrainingBatch(
        x,
        y,
        move(extras, device),
        sum(r.image_count for r in rows),
        sum(r.video_count for r in rows),
        sum(r.frame_count for r in rows),
        sum(r.image_features for r in rows),
    ).to(device)
