"""Materialize the budgets and dependencies from the user's 2026-09-08 documents."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4.vision import QwenVisionConfig

ROOT = Path(__file__).resolve().parents[1]


def phase(
    id,
    objective,
    minimum,
    maximum=None,
    *,
    images=0,
    videos=0,
    lengths=(1024,),
    attention="dense_pretrain",
    depends=(),
    scope="main",
    requires=(),
):
    return dict(
        id=id,
        objective=objective,
        budget_min=minimum,
        budget_max=maximum or minimum,
        image_occurrences=images,
        video_examples=videos,
        sequence_lengths=list(lengths),
        attention_phase=attention,
        depends_on=list(depends),
        budget_scope=scope,
        required_evidence=list(requires),
        status="not_started",
    )


def main():
    output = ROOT / "configs/strategies"
    specifications = [
        (
            "minikimik3",
            "01-MiniKimi-K3-全流程训练与结构改造方案.md",
            KimiVisionConfig,
            0.1,
            [0, 1],
            [
                phase(
                    "K0",
                    "ce_tokens",
                    500_000,
                    2_000_000,
                    lengths=(64, 256, 512),
                    scope="diagnostic",
                    requires=("numerical_contracts", "native_media_contracts", "tokenizer_frozen"),
                ),
                phase(
                    "K-pilot",
                    "ce_tokens",
                    20_000_000,
                    lengths=(512, 1024),
                    depends=("K0",),
                    scope="recipe_pilot",
                    requires=("optimizer_comparison", "mtp_coefficient_comparison"),
                ),
                phase(
                    "K1",
                    "ce_tokens",
                    200_000_000,
                    images=200_000,
                    lengths=(512, 1024),
                    depends=("K-pilot",),
                ),
                phase(
                    "K2",
                    "ce_tokens",
                    1_200_000_000,
                    images=1_200_000,
                    videos=20_000,
                    lengths=(1024, 2048),
                    depends=("K1",),
                ),
                phase(
                    "K3",
                    "ce_tokens",
                    400_000_000,
                    images=500_000,
                    videos=10_000,
                    lengths=(2048, 4096),
                    depends=("K2",),
                ),
                phase(
                    "K4",
                    "ce_tokens",
                    200_000_000,
                    images=100_000,
                    lengths=(1024, 2048, 4096),
                    depends=("K3",),
                ),
                phase(
                    "K-SFT",
                    "assistant_tokens",
                    80_000_000,
                    200_000_000,
                    depends=("K4",),
                    scope="posttraining",
                    requires=("qat_consistency", "base_language_and_vision"),
                ),
                phase(
                    "K-teachers",
                    "unique_prompts_per_teacher",
                    10_000,
                    30_000,
                    depends=("K-SFT",),
                    scope="teachers_9",
                    requires=("sft_to_rl", "behavior_probability", "tool_action_mask"),
                ),
                phase(
                    "K-MOPD",
                    "response_tokens",
                    20_000_000,
                    60_000_000,
                    depends=("K-teachers",),
                    scope="posttraining",
                    requires=("qualified_teachers_9", "sampled_token_mopd"),
                ),
                phase(
                    "K-draft",
                    "draft_positions",
                    10_000_000,
                    30_000_000,
                    depends=("K-MOPD",),
                    scope="draft",
                    requires=("lk_overlap_loss", "unroll_7", "target_frozen"),
                ),
                phase(
                    "K-demo",
                    "fixed_scenarios",
                    20,
                    depends=("K-draft",),
                    scope="release",
                    requires=(
                        "incremental_cache",
                        "speculative_rollback",
                        "quality_selected_checkpoint",
                    ),
                ),
            ],
        ),
        (
            "miniqwen4",
            "02-MiniQwen4-全流程训练与结构改造方案.md",
            QwenVisionConfig,
            0.1,
            [2, 3],
            [
                phase(
                    "Q0",
                    "ce_tokens",
                    500_000,
                    2_000_000,
                    lengths=(64, 256, 512),
                    scope="diagnostic",
                    requires=("numerical_contracts", "native_media_contracts", "tokenizer_frozen"),
                ),
                phase(
                    "Q-pilot",
                    "ce_tokens",
                    20_000_000,
                    lengths=(512, 1024),
                    depends=("Q0",),
                    scope="recipe_pilot",
                    requires=("optimizer_comparison", "mtp_coefficient_comparison"),
                ),
                phase(
                    "Q1",
                    "ce_tokens",
                    300_000_000,
                    images=300_000,
                    lengths=(512, 1024),
                    depends=("Q-pilot",),
                ),
                phase(
                    "Q2",
                    "ce_tokens",
                    1_200_000_000,
                    images=1_200_000,
                    videos=20_000,
                    lengths=(1024, 2048),
                    depends=("Q1",),
                ),
                phase(
                    "Q3",
                    "input_tokens",
                    20_000_000,
                    60_000_000,
                    lengths=(2048, 4096),
                    attention="dense_distill",
                    depends=("Q2",),
                    scope="indexer",
                    requires=("indexer_only_gradients", "dense_teacher_quality"),
                ),
                phase(
                    "Q4",
                    "ce_tokens",
                    1_000_000_000,
                    images=800_000,
                    videos=20_000,
                    lengths=(2048, 4096),
                    attention="sparse_cpt",
                    depends=("Q3",),
                    requires=("dense_sparse_conversion",),
                ),
                phase(
                    "Q5",
                    "ce_tokens",
                    500_000_000,
                    images=200_000,
                    videos=10_000,
                    lengths=(1024, 2048, 4096),
                    attention="sparse_cpt",
                    depends=("Q4",),
                ),
                phase(
                    "Q-SFT",
                    "assistant_tokens",
                    120_000_000,
                    250_000_000,
                    attention="sparse_cpt",
                    depends=("Q5",),
                    scope="posttraining",
                    requires=("base_language_and_vision",),
                ),
                phase(
                    "Q-GRPO",
                    "response_tokens",
                    10_000_000,
                    30_000_000,
                    attention="sparse_cpt",
                    depends=("Q-SFT",),
                    scope="posttraining",
                    requires=("sft_to_rl", "behavior_probability", "tool_action_mask"),
                ),
                phase(
                    "Q-draft",
                    "draft_positions",
                    10_000_000,
                    30_000_000,
                    attention="sparse_cpt",
                    depends=("Q-GRPO",),
                    scope="draft",
                    requires=("target_frozen", "multistream_mtp"),
                ),
                phase(
                    "Q-demo",
                    "fixed_scenarios",
                    20,
                    attention="sparse_cpt",
                    depends=("Q-draft",),
                    scope="release",
                    requires=(
                        "incremental_cache",
                        "speculative_rollback",
                        "quality_selected_checkpoint",
                    ),
                ),
            ],
        ),
        (
            "minideepseekv4",
            "03-MiniDeepSeek-V4-全流程训练与结构改造方案.md",
            DeepSeekVisionConfig,
            0.3,
            [4, 5],
            [
                phase(
                    "D0",
                    "ce_tokens",
                    500_000,
                    2_000_000,
                    lengths=(64, 256, 512),
                    scope="diagnostic",
                    requires=("numerical_contracts", "tokenizer_frozen"),
                ),
                phase(
                    "D-pilot",
                    "ce_tokens",
                    20_000_000,
                    50_000_000,
                    lengths=(512, 1024),
                    depends=("D0",),
                    scope="recipe_pilot",
                    requires=("optimizer_comparison", "mtp_coefficient_comparison"),
                ),
                phase("D1", "ce_tokens", 250_000_000, lengths=(512, 1024), depends=("D-pilot",)),
                phase("D2", "ce_tokens", 500_000_000, lengths=(1024, 2048), depends=("D1",)),
                phase(
                    "D3",
                    "input_tokens",
                    10_000_000,
                    30_000_000,
                    lengths=(2048, 4096),
                    attention="dense_distill",
                    depends=("D2",),
                    scope="indexer",
                    requires=("indexer_only_gradients", "dense_teacher_quality"),
                ),
                phase(
                    "D4",
                    "ce_tokens",
                    1_500_000_000,
                    lengths=(2048, 4096),
                    attention="sparse_cpt",
                    depends=("D3",),
                    requires=("dense_sparse_conversion",),
                ),
                phase(
                    "D5",
                    "ce_tokens",
                    250_000_000,
                    lengths=(1024, 2048, 4096),
                    attention="sparse_cpt",
                    depends=("D4",),
                ),
                phase(
                    "V0",
                    "diagnostic_images",
                    64,
                    512,
                    depends=("D5",),
                    scope="vision_diagnostic",
                    requires=("text_migration_equivalence", "native_media_contracts"),
                ),
                phase(
                    "V1",
                    "ce_tokens",
                    10_000_000,
                    30_000_000,
                    depends=("V0",),
                    attention="sparse_cpt",
                    scope="visual_warmup",
                    requires=("random_vision_trainable_text_frozen",),
                ),
                phase(
                    "V2",
                    "ce_tokens",
                    240_000_000,
                    images=1_200_000,
                    lengths=(1024, 2048),
                    attention="sparse_cpt",
                    depends=("V1",),
                    scope="vision_main",
                ),
                phase(
                    "V3",
                    "ce_tokens",
                    60_000_000,
                    images=300_000,
                    lengths=(1024, 2048, 4096),
                    attention="sparse_cpt",
                    depends=("V2",),
                    scope="vision_main",
                ),
                phase(
                    "D-SFT",
                    "assistant_tokens",
                    100_000_000,
                    220_000_000,
                    attention="sparse_cpt",
                    depends=("V3",),
                    scope="posttraining",
                    requires=("qat_consistency", "base_language_and_vision"),
                ),
                phase(
                    "D-teachers",
                    "unique_prompts_per_teacher",
                    10_000,
                    30_000,
                    attention="sparse_cpt",
                    depends=("D-SFT",),
                    scope="teachers_12",
                    requires=("sft_to_rl", "behavior_probability", "tool_action_mask"),
                ),
                phase(
                    "D-OPD",
                    "response_tokens",
                    30_000_000,
                    80_000_000,
                    attention="sparse_cpt",
                    depends=("D-teachers",),
                    scope="posttraining",
                    requires=("qualified_teachers_12", "full_vocab_reverse_kl"),
                ),
                phase(
                    "D-DSpark",
                    "draft_positions",
                    10_000_000,
                    30_000_000,
                    attention="sparse_cpt",
                    depends=("D-OPD",),
                    scope="draft",
                    requires=("dspark_three_stages_block5", "confidence_overlap", "target_frozen"),
                ),
                phase(
                    "D-demo",
                    "fixed_scenarios",
                    20,
                    attention="sparse_cpt",
                    depends=("D-DSpark",),
                    scope="release",
                    requires=(
                        "incremental_cache",
                        "speculative_rollback",
                        "quality_selected_checkpoint",
                    ),
                ),
            ],
        ),
    ]
    for model, filename, vision, coefficient, gpus, phases in specifications:
        for item in phases:
            if item["budget_scope"] == "recipe_pilot":
                # Comparisons are the pilot's output, not a prerequisite to run it.
                item["produces_evidence"] = item["required_evidence"]
                item["required_evidence"] = ["diagnostic_learnability"]
            if item["id"] in {"K1", "Q1", "D1"}:
                item["required_evidence"] += [
                    "optimizer_comparison",
                    "mtp_coefficient_comparison",
                    "tokenizer_quality_comparison",
                ]
            if item["id"] in {"K0", "Q0", "V0"}:
                item["unique_diagnostic_images"] = [64, 512]
        config = json.loads((ROOT / f"configs/{model}.json").read_text())
        config.update(
            mtp_enabled=True,
            mtp_loss_coef=coefficient,
            forbidden_action_ids=[0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 20],
        )
        if model == "minikimik3":
            config["router_fp32"] = True
        if model == "minideepseekv4":
            config.update(
                window_size=128,
                route_scale=1.5,
                compress_rope_theta=160000.0,
                sequence_balance_coef=1e-4,
            )
        if model != "minideepseekv4":
            config["vision_config"] = asdict(vision())
        config_name = f"{model}-v2.json"
        (output / config_name).write_text(json.dumps(config, indent=2) + "\n")
        if model == "minideepseekv4":
            (output / "minideepseekv4-vision-v1.json").write_text(
                json.dumps(dict(config, vision_config=asdict(vision())), indent=2) + "\n"
            )
        document = ROOT / "docs/training-strategies/2026-09-08" / filename
        plan: dict[str, Any] = dict(
            schema_version=1,
            model=model,
            source_document=str(document.relative_to(ROOT)),
            source_document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
            config=f"configs/strategies/{config_name}",
            gpu_ids=gpus[:1],
            experiment_gpu_ids=gpus,
            execution_override="2026-09-09 user instruction: one GPU per run, up to six independent experiments on GPUs 0-5; active two-GPU trials finish unchanged",
            training_dtype="bf16",
            optimizer_state_dtype="fp32",
            min_disk_free_gib=50,
            context_claims="only lengths with completed training and evaluation",
            performance=dict(warmup_updates=50, measured_updates=200, reserved_device_gib=2),
            tokenizer=dict(
                default_vocab=65536,
                candidates=[32768, 65536],
                shared_training_bytes=True,
                freeze_before_pilot=True,
            ),
            validation=dict(
                source_holdout_min_fraction=0.005,
                periodic_ce_tokens=5_000_000,
                sft_unique_questions=2000,
                vision_unique_groups=1000,
                sealed_test=True,
            ),
            text_mixture_tokens={
                "zh_edu": 0.45,
                "en_edu": 0.25,
                "code": 0.15,
                "verified_math_science": 0.10,
                "dialogue": 0.05,
            },
            visual_mixture_samples={
                "caption": 0.45,
                "ocr_document": 0.25,
                "chart_count_spatial": 0.15,
                "interleaved": 0.10,
                "video": 0.05,
            },
            optional_extensions=["DPO"] if model != "miniqwen4" else ["DPO", "OPD", "QAT"],
            phases=phases,
        )
        if model != "minikimik3":
            plan["text_mixture_tokens"].update(zh_edu=0.40, code=0.20)
        if model == "miniqwen4":
            plan["visual_mixture_samples"] = {
                "caption": 0.35,
                "ocr_document": 0.25,
                "chart_count_spatial": 0.20,
                "interleaved_multiimage": 0.15,
                "video": 0.05,
            }
        if model == "minideepseekv4":
            plan["visual_mixture_samples"] = {
                "caption": 0.35,
                "ocr_document": 0.30,
                "chart_count_spatial": 0.20,
                "screenshot_tool": 0.10,
                "multiimage_multipage": 0.05,
            }
            plan["visual_text_replay_ce_fraction"] = [0.3, 0.5]
        (output / f"{model}-plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
