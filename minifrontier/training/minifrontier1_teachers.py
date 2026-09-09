"""Eight domain/effort slots, held-out qualification and sequential independent training."""

from pathlib import Path

import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import RecordDataset, write_json
from minifrontier.evaluation.minifrontier1 import generation_suite
from minifrontier.inference.runtime import load_checkpoint
from minifrontier.training.minifrontier1_strategy import TEACHER_SLOTS

DOMAINS = dict(
    general_tools={"general", "tools", "structured", "zh_general", "en_general"},
    code={"code"},
    math={"math", "arithmetic"},
    vision={"vision", "video", "caption", "ocr_document", "vqa", "chart_table", "multiimage_video"},
)


def slot_for(record):
    domain = next((key for key, domains in DOMAINS.items() if record["domain"] in domains), None)
    if domain is None:
        raise ValueError("no teacher domain mapping for this prompt bucket")
    thinking = record.get("mode") == "thinking" or any(
        m.get("channel") == "thinking" for m in record["messages"]
    )
    return f"{domain}:{'thinking' if thinking else 'direct'}"


def qualify_teacher(checkpoint, baseline, data, slot, output, *, device="cpu", limit=64):
    if slot not in TEACHER_SLOTS:
        raise ValueError("unknown teacher slot")
    domain, effort = slot.split(":")
    domains = DOMAINS[domain]
    candidate = generation_suite(
        checkpoint,
        data,
        device=device,
        limit=limit,
        controls=False,
        domain_filter=domains,
        effort_filter=effort,
    )
    common = generation_suite(
        baseline,
        data,
        device=device,
        limit=limit,
        controls=False,
        domain_filter=domains,
        effort_filter=effort,
    )
    a = {s["sample_id"]: s for s in candidate["samples"]}
    b = {s["sample_id"]: s for s in common["samples"]}
    if a.keys() != b.keys() or not a:
        raise ValueError(
            "teacher and baseline qualification need the same nonempty held-out prompts"
        )
    differences = torch.tensor([float(a[k]["reward"] == 1) - float(b[k]["reward"] == 1) for k in a])
    rng = torch.Generator().manual_seed(42)
    indices = torch.randint(len(a), (2000, len(a)), generator=rng)
    lower = float(differences[indices].mean(-1).quantile(0.025))
    target_passed = len(a) >= 20 and lower > 0
    retention = {}
    visual_passed = False
    for bucket, bucket_domains in DOMAINS.items():
        if bucket == domain and bucket != "vision":
            continue
        evaluated = generation_suite(
            checkpoint,
            data,
            device=device,
            limit=limit,
            controls=bucket == "vision",
            domain_filter=bucket_domains,
        )
        previous = generation_suite(
            baseline,
            data,
            device=device,
            limit=limit,
            controls=False,
            domain_filter=bucket_domains,
        )
        matched = {s["sample_id"]: s for s in evaluated["samples"] if s["control"] == "matched"}
        prior = {s["sample_id"]: s for s in previous["samples"]}
        if matched.keys() != prior.keys():
            raise ValueError("non-target regression prompts differ")
        n = len(matched)
        delta = sum(
            float(matched[k]["reward"] == 1) - float(prior[k]["reward"] == 1) for k in matched
        ) / max(1, n)
        retention[bucket] = dict(
            samples=n,
            accuracy_change=delta,
            passed=n >= 20 and delta >= -0.02,
            candidate=evaluated,
            baseline=previous,
        )
        if bucket == "vision":
            black = {s["sample_id"]: s for s in evaluated["samples"] if s["control"] == "black"}
            dependency = sum(
                float(matched[k]["reward"] == 1) - float(s["reward"] == 1) for k, s in black.items()
            ) / max(1, len(black))
            visual_passed = len(black) >= 20 and dependency >= 0.10
            retention[bucket]["matched_minus_black_accuracy"] = dependency
    eos_rate = sum(s["termination"] == "eos" for s in a.values()) / len(a)
    retention_passed = bool(retention) and all(r["passed"] for r in retention.values())
    qualified = (
        target_passed
        and retention_passed
        and visual_passed
        and eos_rate >= 0.90
        and candidate["formal_dataset_admitted"]
    )
    result = dict(
        checkpoint_sha256=sha256(checkpoint),
        baseline_sha256=sha256(baseline),
        slot=slot,
        tokenizer_sha256=candidate["tokenizer_sha256"],
        dataset_manifest_sha256=candidate["dataset_manifest_sha256"],
        metrics=dict(
            qualified=qualified,
            paired_accuracy_gain=float(differences.mean()),
            bootstrap_95_lower=lower,
            samples=len(a),
            target_domain_passed=target_passed,
            retention_passed=retention_passed,
            visual_dependency_passed=visual_passed,
            eos_rate=eos_rate,
        ),
        retention=retention,
        candidate=candidate,
        baseline=common,
        thresholds=dict(
            minimum_samples_per_bucket=20,
            maximum_accuracy_drop=0.02,
            minimum_visual_dependency=0.10,
            minimum_eos_rate=0.90,
        ),
        status="qualified" if qualified else "unqualified",
        limitations=["qualification applies to the bound held-out suite and exact checkpoint only"],
    )
    write_json(output, result)
    return result


def train_teachers(checkpoint, data, output, *, device="cpu", steps=2, max_tokens=32):
    from minifrontier.training.minifrontier1_posttrain import train_post

    model, _, _ = load_checkpoint(checkpoint, "cpu")
    dataset = RecordDataset(data, "train", model.config)
    del model
    available = {slot_for(dataset.record(i)) for i in range(len(dataset))}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = dict(
        format="mf1-teacher-registry-v1", baseline_sha256=sha256(checkpoint), teachers={}
    )
    for slot in TEACHER_SLOTS:
        if slot not in available:
            registry["teachers"][slot] = dict(
                status="unqualified", reason="no matching domain/effort training records"
            )
            continue
        root = output / slot.replace(":", "-")
        # Every teacher starts from exactly the same baseline, never a previous teacher.
        report = train_post(
            phase="teacher",
            checkpoint=checkpoint,
            data=data,
            output=root,
            device=device,
            steps=steps,
            max_tokens=max_tokens,
            teacher_slot=slot,
        )
        path = Path(report["checkpoint"])
        registry["teachers"][slot] = dict(
            status="unqualified",
            checkpoint=str(path.relative_to(output)),
            checkpoint_sha256=sha256(path),
            reason="training execution is not qualification",
        )
        write_json(output / "registry.json", registry)
    write_json(output / "registry.json", registry)
    return registry
