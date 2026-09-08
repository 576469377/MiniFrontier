# Changelog

## Unreleased

- Standardized display names to MiniQwen4, MiniKimi-K3 and MiniDeepSeek-V4.
- Added source-derived Kimi KDA/MLA/LatentMoE/AttnRes and DeepSeek compression/indexer/mHC text backbones with explicit floating-point training adaptations.
- Added reproducible public-corpus processing, tokenizer training, pretraining, SFT, DPO and sparse-indexer stages; optional verifiable GRPO and multi-teacher on-policy distillation.
- Added deterministic single-GPU/DDP training, atomic checkpoints, exact resume, validation, TensorBoard, stage controllers, CLI generation and a local browser demo.
- Executed real two-GPU short pipelines for all three models and launched separate educational training runs.
- Added source-component, causality, gradient, multi-stage, recovery and distributed training tests.

### Earlier source audit


- Organized current model code as MiniQwen4, MiniKimiK3 and MiniDeepSeekV4 packages.
- Kept source-verified components, tests, immutable upstream snapshots and licenses.
- Connected MiniQwen4 dense pretraining, indexer distillation and sparse CPT objectives, with explicit phase boundaries.
- Removed retired model implementations, recipes, services, datasets, tokenizer, weights and reports.
- Consolidated the model catalog and public CLI, and refreshed packaging and CI.

All models remain research implementations; no complete formal training reproduction is claimed.
