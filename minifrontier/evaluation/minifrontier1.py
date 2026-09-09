"""Checkpoint-bound fixed generation suite and image/video dependency controls."""

from collections import defaultdict

import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import RecordDataset, encode_record
from minifrontier.inference.runtime import generate_ids, load_checkpoint
from minifrontier.multimodal import move
from minifrontier.provenance import source_identity
from minifrontier.training.minifrontier1_posttrain import verify_answer


def media_controls(spans, generator):
    variants = {"matched": spans}
    if not spans:
        return variants
    variants["black"] = [dict(s, patches=torch.zeros_like(s["patches"])) for s in spans]
    variants["patch_shuffle"] = [
        dict(s, patches=s["patches"][torch.randperm(len(s["patches"]), generator=generator)])
        for s in spans
    ]
    if any(s["resource_kind"] == "video" for s in spans):
        for control in ("reversed_time_groups", "first_time_group_repeated"):
            changed = []
            for span in spans:
                t = int(span["grid_thw"][:, 0].sum())
                patches = span["patches"]
                if span["resource_kind"] == "video":
                    groups = patches.unflatten(0, (t, -1))
                    patches = (
                        groups.flip(0)
                        if control == "reversed_time_groups"
                        else groups[:1].expand_as(groups)
                    ).flatten(0, 1)
                changed.append(dict(span, patches=patches))
            variants[control] = changed
    return variants


def generation_suite(
    checkpoint,
    data,
    *,
    device="cpu",
    split="val",
    limit=0,
    max_new_tokens=32,
    seed=42,
    controls=True,
    domain_filter=None,
    effort_filter=None,
):
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    dataset = RecordDataset(data, split, model.config)
    generator = torch.Generator().manual_seed(seed)
    samples = []
    summaries: dict[str, dict[str, int]] = defaultdict(
        lambda: dict(correct=0, total=0, generated_tokens=0)
    )
    with torch.random.fork_rng(
        devices=[torch.device(device).index or 0] if str(device).startswith("cuda") else []
    ):
        torch.manual_seed(seed)
        selected = 0
        for i in range(len(dataset)):
            record = dataset.record(i)
            if domain_filter and record["domain"] not in domain_filter:
                continue
            effort = (
                "thinking"
                if record.get("mode") == "thinking"
                or any(m.get("channel") == "thinking" for m in record["messages"])
                else "direct"
            )
            if effort_filter and effort != effort_filter:
                continue
            if limit and selected >= limit:
                break
            selected += 1
            prepared = encode_record(
                record, tokenizer, model.config, dataset.media_root, generation_prompt=True
            )
            for control, spans in (
                media_controls(prepared["media"], generator)
                if controls
                else {"matched": prepared["media"]}
            ).items():
                inputs, media = prepared["input_ids"].to(device), move(spans, device)
                generated = generate_ids(
                    model,
                    inputs,
                    media=media,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    top_p=1,
                    vocab_size=tokenizer.get_vocab_size(),
                )
                ids = generated[0, inputs.shape[1] :].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                reward, verifier = verify_answer(record, text)
                key = f"{record['domain']}/{control}"
                summaries[key]["correct"] += int(reward == 1)
                summaries[key]["total"] += 1
                summaries[key]["generated_tokens"] += len(ids)
                samples.append(
                    dict(
                        sample_id=record["sample_id"],
                        domain=record["domain"],
                        effort=effort,
                        control=control,
                        input_ids=inputs.tolist(),
                        output_ids=ids,
                        text=text,
                        reward=reward,
                        verifier=verifier,
                        termination="eos"
                        if ids and ids[-1] == model.config.eos_token_id
                        else "length",
                    )
                )
    return dict(
        checkpoint_sha256=sha256(checkpoint),
        tokenizer_sha256=dataset.manifest["tokenizer_sha256"],
        dataset_manifest_sha256=sha256(dataset.root / "manifest.json"),
        source=source_identity(),
        split=split,
        seed=seed,
        temperature=0,
        top_p=1,
        max_new_tokens=max_new_tokens,
        summaries=dict(summaries),
        samples=samples,
        capability_qualified=False,
        formal_dataset_admitted=dataset.manifest.get("formal_admission") is True,
        control_note="video controls reorder or repeat temporal patch groups, preserving input length; they are not a full video benchmark",
    )
