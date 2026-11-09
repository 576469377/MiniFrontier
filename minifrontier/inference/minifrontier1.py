"""Target-only native MF1 generation with the same tokenizer, media and control contracts."""

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import safe_text, transform_media
from minifrontier.inference.runtime import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1 import MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier11 import MiniFrontier11ForCausalLM
from minifrontier.multimodal import move


def append_media(
    ids,
    spans,
    model,
    frames,
    hashes,
    *,
    video,
    timestamps=None,
    representation="standard",
    max_features=None,
):
    remaining = model.config.protected_media_tokens - sum(s["feature_count"] for s in spans)
    kind = "video" if video else "image"
    if max_features is None:
        max_features = getattr(model, "inference_media_profile", {}).get(kind + "_max_features")
    samples = transform_media(
        frames,
        model.config,
        remaining,
        representation=representation,
        max_features=max_features,
        timestamps=timestamps,
    )
    for sample in samples:
        ids.append(20 if video else 9)
        sample.update(batch_index=0, start=len(ids), resource_kind=kind, source_hashes=hashes)
        ids.extend([7] * sample["feature_count"] + [21 if video else 10])
        spans.append(sample)
    return samples


def prepare_prompt(
    model,
    tokenizer,
    prompt,
    *,
    images=None,
    video_frames=None,
    timestamps=None,
    mode="direct",
    max_image_features=None,
    max_video_features=None,
    image_representation="standard",
):
    if mode not in {"direct", "thinking"}:
        raise ValueError("MF1 supports direct or thinking control")
    ids = [1, 4]
    media: list[dict[str, Any]] = []
    resources = [([path], None) for path in images or []]
    if video_frames:
        if len(video_frames) < 2 or timestamps is None:
            raise ValueError("video frame input requires at least two frames and source timestamps")
        resources.append((video_frames, timestamps))
    for paths, times in resources:
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        append_media(
            ids,
            media,
            model,
            frames,
            [sha256(p) for p in paths],
            video=times is not None,
            timestamps=times,
            representation=image_representation if times is None else "standard",
            max_features=max_image_features if times is None else max_video_features,
        )
    ids += [*safe_text(tokenizer, prompt), 2, 5, 15 if mode == "thinking" else 17]
    return torch.tensor([ids]), media


def respond(
    checkpoint,
    prompt,
    image=None,
    video_frame=None,
    timestamps=None,
    mode="direct",
    device="cpu",
    max_new_tokens=64,
    temperature=0,
    draft_checkpoint=None,
    draft_steps=4,
    max_image_features=None,
    max_video_features=None,
    image_representation="standard",
):
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    if type(model) not in {MiniFrontier1ForCausalLM, MiniFrontier11ForCausalLM}:
        raise ValueError("mf1 generate requires a MiniFrontier1 checkpoint")
    times = json.loads(timestamps) if isinstance(timestamps, str) else timestamps
    ids, media = prepare_prompt(
        model,
        tokenizer,
        prompt,
        images=image,
        video_frames=video_frame,
        timestamps=times,
        mode=mode,
        max_image_features=max_image_features,
        max_video_features=max_video_features,
        image_representation=image_representation,
    )
    ids, media = ids.to(device), move(media, device)
    start = time.perf_counter()
    stats = None
    if draft_checkpoint:
        if model.mtp is None:
            raise ValueError(
                "this backbone has no MTP; a separately trained MF1.1 drafter is not implemented"
            )
        from minifrontier.models.minifrontier1.draft import MF1Draft
        from minifrontier.speculative import generate_speculative

        saved = torch.load(draft_checkpoint, map_location="cpu", weights_only=True)
        if (
            saved.get("target_sha256") != sha256(checkpoint)
            or saved.get("draft_rule") != "fixed-anchor-v1"
        ):
            raise ValueError("draft must be bound to this exact target checkpoint and rule")
        draft = MF1Draft(model).to(device).eval()
        draft.load_state_dict(saved["draft"])
        if temperature == 0:
            result, stats = draft.generate_greedy(
                ids,
                max_new_tokens=max_new_tokens,
                steps=draft_steps,
                media=media,
                vocab_size=tokenizer.get_vocab_size(),
            )
        elif temperature == 1:
            result, stats = generate_speculative(
                model,
                draft,
                ids,
                max_new_tokens=max_new_tokens,
                draft_steps=draft_steps,
                media=media,
                vocab_size=tokenizer.get_vocab_size(),
            )
        else:
            raise ValueError("MF1 speculative reference supports only temperature 0 or 1")
    else:
        result = generate_ids(
            model,
            ids,
            media=media,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1,
            vocab_size=tokenizer.get_vocab_size(),
        )
    elapsed = time.perf_counter() - start
    tokens = result[0, ids.shape[1] :].tolist()
    return dict(
        text=tokenizer.decode(tokens, skip_special_tokens=True),
        output_ids=tokens,
        raw_output=tokenizer.decode(tokens, skip_special_tokens=False),
        checkpoint_sha256=sha256(checkpoint),
        tokenizer_sha256=sha256(Path(checkpoint).parent / "tokenizer.json"),
        mode=mode,
        input_tokens=ids.numel(),
        vision_tokens=sum(m["feature_count"] for m in media),
        media_plan=[
            {
                k: m[k]
                for k in (
                    "resource_kind",
                    "source_size",
                    "resized_size",
                    "original_frames",
                    "feature_count",
                )
            }
            for m in media
        ],
        generated_tokens=len(tokens),
        elapsed_seconds=elapsed,
        termination="eos" if tokens and tokens[-1] == model.config.eos_token_id else "length",
        capability_qualified=False,
        speculative=stats,
    )
