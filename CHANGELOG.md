# Changelog

## Unreleased

- Add MiniDeepSeek-V4.1 (242M): causal encoder/decoder, CSA2 sharing, Single-Pass mHC, Engram and sparse text pretraining. Add MF1.1 (211M): versioned mHC and optimizer changes with native media, without MTP.
- Document all six model versions, add V4.1 and MF1.1 architecture diagrams, and distinguish their training plans and validation scope from earlier results.
- Extend the checkpoint browser to all six versions and the media page to both MF1 versions. Add version-bound generation, saved request parameters, output history, comparisons and JSON export; bundle the browser assets in wheel and sdist.
- Add the independent 228,235,809-parameter MiniFrontier1.0 native fusion model: KDA, CSA-4, QSA-MLA, four-stream GR, LatentMoE, shallow lookup, random ViT and shared-head MTP.
- Add MF1 data, training/resume, stage checks, posttraining, draft, export and diagnostic demo commands. Add experiment/history views and physical GPU selection to the source-model demo.
- Start formal pretraining for the original four models and the two new versions, with separate budgets. Preserve optimizer state and token accounting across explicit stages; allow later-stage data binding while retaining checks on completed stages.
- Add candidate text/code/vision preparation, resumable reads, benchmark exclusions and cross-slice merging. Add compact MF1 shards with bounded mappings and corruption checks.
- Optimize MF1 attention, KDA, lookup, experts and sample batching; fix the single-token KDA training gradient path. Add ordered data prefetch, batched Qwen optimizer blocks and CUDA cache reclamation, with numerical and resume records.
- Publish MF1 CPU learning, CUDA resource and mechanism experiments with reproducible data, token counts and curves. Document checkpoint retention and storage limits.
- Group workflows into data, training, evaluation, inference and commands; preserve CLI and checkpoint formats. Separate browser serving from inference and group TensorBoard metrics into `train`, `eval` and `perf`.
- Rewrite model and operation guides, add MF1 architecture diagrams and attributed report figures, and consolidate current training arrangements. Retain dated results, original strategy documents and old page links; update project metadata and repository navigation.

## 0.1.0 — Research preview (prepared; not published)

- Source-derived MiniQwen4, MiniKimi-K3 and MiniDeepSeek-V4 text, native vision and MTP implementations, with pinned upstream snapshots and numerical tests.
- Offline CPU and single-GPU examples covering generated data, PT, checkpoint pause/resume, SFT, validation and explicit CLI generation.
- Single-GPU/DDP training, actual-token accounting, atomic checkpoints, native inference caches and experimental posttraining/draft entry points.
- Public numeric experiment snapshots and redrawable curves, retaining failure reports and explicit incomplete states.
- Updated architecture/capability documentation, single-GPU defaults, per-component license mapping and distribution license files.
- Installed-wheel acceptance and an actionable Git-checkout requirement for formal strategy training.

No capable chat weights, completed billion-token training, qualified multi-teacher RL or measured trained-draft acceleration are released. Diagnostic completion and declining NLL do not establish model capability. A private reporting contact remains to be specified before a formal release; see [release scope](docs/releases/v0.1.0.md).
