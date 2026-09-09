"""Target-only native MF1 generation with the same tokenizer, media and control contracts."""

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import safe_text
from minifrontier.inference.runtime import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1.processing import process_frames
from minifrontier.multimodal import move


def prepare_prompt(
    model, tokenizer, prompt, *, images=None, video_frames=None, timestamps=None, mode="direct"
):
    if mode not in {"direct", "thinking"}:
        raise ValueError("MF1 supports direct or thinking control")
    c = model.config
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
        remaining = c.protected_media_tokens - sum(m["feature_count"] for m in media)
        sample = process_frames(
            frames, max_features=remaining, patch_size=c.vision_config.patch_size, timestamps=times
        )
        ids.append(20 if times is not None else 9)
        sample.update(
            batch_index=0,
            start=len(ids),
            resource_kind="video" if times is not None else "image",
            source_hashes=[sha256(p) for p in paths],
        )
        media.append(sample)
        ids += [7] * sample["feature_count"] + [21 if times is not None else 10]
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
):
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    if model.__class__.__name__ != "MiniFrontier1ForCausalLM":
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
    )
    ids, media = ids.to(device), move(media, device)
    start = time.perf_counter()
    stats = None
    if draft_checkpoint:
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
