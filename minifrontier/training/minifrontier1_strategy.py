"""Executable phase budgets and artifact-bound stage admission for MF1, separate from v2."""

import copy
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import digest
from minifrontier.models.minifrontier1.configuration import MF1_VERSION
from minifrontier.models.minifrontier1.processing import CONTROL_VERSION, PROCESSOR_VERSION
from minifrontier.models.minifrontier11.configuration import MF11_VERSION
from minifrontier.provenance import checkout_root, require_source_checkout, source_identity
from minifrontier.training.strategy_gate import initial_start_authorized

PLAN_PATH = (
    "docs/training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md"
)
PLAN_SHA256 = "e0840c97bc5c23823e723af79b27ddb69c6067e74af1fa06fd294b0cc4791ef7"
MF11_PLAN_PATH = "docs/training-strategies/2026-09-14/07-v41-mf11-implementation-and-training.md"

PHASES: dict[str, dict[str, Any]] = {
    "pilot": dict(budget=20_000_000, unit="ce_tokens", attention="dense_pretrain", predecessor=[]),
    "p0": dict(
        budget=200_000_000,
        unit="ce_tokens",
        attention="dense_pretrain",
        predecessor=[],
        visual_ce=0.20,
        lengths={512: 0.9, 1024: 0.1},
    ),
    "p1": dict(
        budget=800_000_000,
        unit="ce_tokens",
        attention="dense_pretrain",
        predecessor=["p0"],
        visual_ce=0.20,
        lengths={1024: 0.7, 2048: 0.3},
    ),
    "indexer": dict(
        budget=40_000_000,
        unit="input_tokens",
        attention="dense_distill",
        predecessor=["p1"],
        lengths={1024: 0.2, 2048: 0.5, 4096: 0.3},
    ),
    "p2": dict(
        budget=1_400_000_000,
        unit="ce_tokens",
        attention="sparse_cpt",
        predecessor=["indexer"],
        visual_ce=0.20,
        lengths={2048: 0.7, 4096: 0.3},
    ),
    "p3": dict(
        budget=600_000_000,
        unit="ce_tokens",
        attention="sparse_cpt",
        predecessor=["p2"],
        visual_ce=0.25,
        lengths={2048: 0.25, 4096: 0.45, 8192: 0.30},
    ),
    "sft": dict(budget=120_000_000, unit="ce_tokens", attention="sparse_cpt", predecessor=["p3"]),
    "rl": dict(
        budget=20_000_000, unit="generated_tokens", attention="sparse_cpt", predecessor=["sft"]
    ),
    "teacher": dict(
        budget=4_000_000, unit="generated_tokens", attention="sparse_cpt", predecessor=["sft", "rl"]
    ),
    "opd": dict(
        budget=20_000_000,
        unit="generated_tokens",
        attention="sparse_cpt",
        predecessor=["sft", "rl"],
    ),
    "dpo": dict(
        budget=4_000_000,
        unit="response_positions",
        attention="sparse_cpt",
        predecessor=["opd", "rl", "sft"],
        optional=True,
    ),
    "draft": dict(
        budget=10_000_000,
        unit="generated_tokens",
        attention="sparse_cpt",
        predecessor=["opd", "dpo", "rl", "sft"],
    ),
    "qat": dict(
        budget=10_000_000,
        unit="ce_tokens",
        attention="sparse_cpt",
        predecessor=["opd", "dpo", "rl", "sft"],
        optional=True,
    ),
}
# A separate versioned plan: changing the next model must not rewrite MF1.0's recipe.
# The retained QSA/CSA indexers still require dense teacher distillation before top-k use.
MF11_PHASES = copy.deepcopy(PHASES)


def phases_for(model_version=MF1_VERSION):
    if model_version == MF1_VERSION:
        return PHASES
    if model_version == MF11_VERSION:
        return MF11_PHASES
    raise ValueError("unknown MF1 training plan version")


def model_name_for(model_version=MF1_VERSION):
    phases_for(model_version)
    return "minifrontier11" if model_version == MF11_VERSION else "minifrontier1"


def plan_path_for(model_version=MF1_VERSION):
    phases_for(model_version)
    return MF11_PLAN_PATH if model_version == MF11_VERSION else PLAN_PATH


TEACHER_SLOTS = [
    f"{domain}:{effort}"
    for domain in ("general_tools", "code", "math", "vision")
    for effort in ("direct", "thinking")
]
TEXT_MIX = {"zh_general": 0.42, "en_general": 0.28, "code": 0.12, "math": 0.10, "structured": 0.08}
VISION_MIX = {
    "caption": 0.30,
    "ocr_document": 0.35,
    "vqa": 0.20,
    "chart_table": 0.10,
    "video": 0.05,
}
SFT_MIX = {
    "general": 0.30,
    "tools": 0.10,
    "code": 0.15,
    "math": 0.15,
    "ocr_document": 0.15,
    "vqa": 0.10,
    "multiimage_video": 0.05,
}


def budget_report(model_version=MF1_VERSION):
    phases = phases_for(model_version)
    main = [phases[p] for p in ("p0", "p1", "p2", "p3")]
    total = sum(p["budget"] for p in main)
    visual = round(sum(p["budget"] * p["visual_ce"] for p in main))
    if (
        total != 3_000_000_000
        or visual != 630_000_000
        or any(not math.isclose(sum(m.values()), 1) for m in (TEXT_MIX, VISION_MIX, SFT_MIX))
    ):
        raise ValueError("MF1 budget arithmetic is inconsistent")
    return dict(
        main_ce_tokens=total,
        visual_ce_tokens=visual,
        text_ce_tokens=total - visual,
        indexer_input_tokens=40_000_000,
        sft_assistant_ce_tokens=120_000_000,
        p0_visual_mix=dict(VISION_MIX, caption=0.35, video=0.0),
        teacher_slots=TEACHER_SLOTS,
        separate_ledgers=[
            "input_tokens",
            "ce_tokens",
            "generated_tokens",
            "vision_tokens",
            "mtp_target_tokens",
            "index_query_tokens",
            "replay_ce_tokens",
            "media_exposures",
        ],
    )


def scheduler_factor(phase, phase_tokens, main_ce_tokens, *, model_version=MF1_VERSION):
    if phase in {"pilot", "p0", "p1", "p2", "p3"}:
        if main_ce_tokens < 2_000_000:
            return max(1, main_ce_tokens) / 2_000_000
        progress = min(1.0, max(0.0, (main_ce_tokens - 2_400_000_000) / 600_000_000))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    total = phases_for(model_version)[phase]["budget"]
    warmup = 1_000_000 if phase == "indexer" else max(1, round(total * 0.02))
    if phase_tokens < warmup:
        return max(1, phase_tokens) / warmup
    progress = min(1.0, (phase_tokens - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def bindings(config, data_dir, init=None):
    root = checkout_root()
    plan = root / plan_path_for(config.model_version) if root else None
    if config.model_version == MF11_VERSION and (plan is None or not plan.is_file()):
        raise ValueError("MF1.1 training requires a Git checkout containing its versioned plan")
    result = dict(
        strategy_sha256=sha256(plan) if plan and plan.is_file() else PLAN_SHA256,
        model_config_sha256=digest(asdict(config)),
        tokenizer_sha256=sha256(Path(data_dir) / "tokenizer.json"),
        processor_sha256=digest(
            dict(
                version=PROCESSOR_VERSION,
                config=asdict(config.vision_config),
                mrope=config.mrope_sections,
                media_budget=config.protected_media_tokens,
            )
        ),
        template_sha256=digest(CONTROL_VERSION),
        dataset_manifest_sha256=sha256(Path(data_dir) / "manifest.json"),
        actual_init_checkpoint_sha256=sha256(init) if init else None,
        source=source_identity(),
    )
    if config.model_version == MF11_VERSION:
        result["model_version"] = config.model_version
    return result


def validate_gate(phase, evidence, actual, manifest, saved, *, resume=False):
    root = require_source_checkout()
    version = actual.get("model_version", MF1_VERSION)
    phases = phases_for(version)
    plan = root / plan_path_for(version)
    if not plan.is_file() or sha256(plan) != actual.get("strategy_sha256"):
        raise ValueError("formal MF1 admission requires the bound strategy source document")
    direct_start = initial_start_authorized(
        evidence,
        model=model_name_for(version),
        phase=phase,
        data_sha256=actual["dataset_manifest_sha256"],
        config_sha256=actual["model_config_sha256"],
        source_commit=actual["source"]["commit"],
    )
    expected_status = "maintainer_authorized_start" if direct_start else "qualified"
    if evidence.get("stage") != phase or evidence.get("status") != expected_status:
        raise ValueError("stage gate must qualify this exact MF1 phase")
    for key, value in actual.items():
        if evidence.get(key) != value:
            raise ValueError(f"stage gate mismatch: {key}")
    if manifest.get("formal_admission") is not True or manifest.get("kind") == "mechanism_fixture":
        raise ValueError("formal MF1 stages require an admitted real dataset manifest")
    if saved is not None and saved.get("mf1_phase") not in (
        [phase] if resume else phases[phase]["predecessor"]
    ):
        raise ValueError("actual initialization checkpoint has the wrong predecessor phase")
    if (
        saved is not None
        and version == MF11_VERSION
        and (
            saved.get("model_name") != "minifrontier11"
            or saved.get("config", {}).get("model_version") != MF11_VERSION
        )
    ):
        raise ValueError("MF1.1 admission requires its own versioned predecessor checkpoint")
    if phases[phase]["predecessor"] and saved is None:
        raise ValueError("this formal stage requires an actual predecessor checkpoint")
    if direct_start:
        if actual["source"]["dirty"] or actual["actual_init_checkpoint_sha256"] is not None:
            raise ValueError("direct P0 starts require clean source and random initialization")
        return evidence
    evaluations = evidence.get("evaluations", [])
    if not evaluations:
        raise ValueError("stage admission requires hashed evaluation artifacts")
    import json

    metrics = {}
    for entry in evaluations:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("predecessor evaluation artifact changed")
        artifact = json.loads(path.read_text())
        if (
            artifact.get("checkpoint_sha256") != actual["actual_init_checkpoint_sha256"]
            and saved is not None
        ):
            raise ValueError("evaluation belongs to a different initialization checkpoint")
        metrics.update(artifact.get("metrics", {}))
    required = {
        "pilot": ["numerical_correctness", "media_gradient", "exact_resume"],
        "p0": ["numerical_correctness", "media_gradient", "exact_resume", "single_3090_qualified"],
        "p1": ["language_improving", "visual_dependency", "router_stable"],
        "indexer": ["base_continuation", "dense_attention_learned"],
        "p2": ["indexer_mass_qualified", "dense_sparse_gap_qualified"],
        "p3": ["context_4k_stable", "visual_dependency"],
        "sft": ["base_continuation", "visual_dependency"],
        "rl": ["readability_qualified", "format_eos_qualified", "nonzero_task_success"],
        "teacher": ["readability_qualified", "nonzero_task_success"],
        "opd": ["qualified_teachers", "vocabulary_processor_equal"],
        "draft": ["target_frozen", "capability_qualified"],
        "qat": ["ptq_quality_insufficient", "capability_qualified"],
        "dpo": ["preference_defect_demonstrated", "matched_verified_pairs"],
    }[phase]
    for key in required:
        if metrics.get(key) is not True:
            raise ValueError(f"phase admission lacks evaluated condition: {key}")
    if phase == "p2":
        for domain in ("text", "code", "ocr", "video"):
            if metrics.get(f"captured_attention_mass/{domain}", 0) < 0.90:
                raise ValueError(f"indexer recall below threshold for {domain}")
        if (
            metrics.get("relative_nll_increase", float("inf")) > 0.01
            or metrics.get("visual_accuracy_drop", float("inf")) > 0.02
        ):
            raise ValueError("sparse-switch validation gap exceeds the planned threshold")
    if phase == "p0" and metrics.get("peak_reserved_gib", float("inf")) > 22:
        raise ValueError("single-card memory admission exceeds 22 GiB")
    return evidence
