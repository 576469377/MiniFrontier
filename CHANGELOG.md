# Changelog

## Unreleased

- Add a MiniFrontier cover and vector identity, illustrate the fusion architecture, and reorganize the README around hands-on examples, model navigation and experiment evidence.
- Rewrite the public overview, document actual dataset sources, correct model-specific QAT/parameter statements, and clarify dated experiment reports. Normalize local paths in public audit copies while preserving metrics and original report hashes.

- Add the GitHub repository, documentation and issue tracker to project metadata and source checkout instructions.
- Group model workflows by data, training, evaluation, inference and commands; retire the separate MF1 workflow directory, preserve CLI/checkpoint formats and shared import facades. Split browser serving from the common inference runtime.
- Record cleanup of 31 completed legacy optimizer checkpoints (85.16 GiB), retaining weight-only exports and experiment evidence.

- Add the independent 228,235,809-parameter MiniFrontier1.0 native fusion model: KDA, CSA-4, QSA-MLA, four-stream GR, LatentMoE, shallow lookup, random ViT and shared-head MTP.
- Add the `mf1` data, training/resume, stage gate, RL/teacher/OPD/DPO/draft, export and diagnostic demo entry points. Formal training remains gated on actual data and evaluation evidence.
- Publish CPU numerical checks and a bounded text/image/video learning experiment with token ledgers and redrawable curves; retain the three source models as independent baselines.
- Group operating guides under `docs/guides/`, add documentation and script navigation, and retain compatibility links and immutable strategy originals.
- Add explicit experiment/history views and physical single-GPU selection to the source-model demo, preserving the default capability gate.

## 0.1.0 — Research preview (prepared; not published)

- Source-derived MiniQwen4, MiniKimi-K3 and MiniDeepSeek-V4 text, native vision and MTP implementations, with pinned upstream snapshots and numerical tests.
- Offline CPU and single-GPU examples covering generated data, PT, checkpoint pause/resume, SFT, validation and explicit CLI generation.
- Single-GPU/DDP training, actual-token accounting, atomic checkpoints, native inference caches and experimental posttraining/draft entry points.
- Public numeric experiment snapshots and redrawable curves, retaining failure reports and explicit incomplete states.
- Updated architecture/capability documentation, single-GPU defaults, per-component license mapping and distribution license files.
- Installed-wheel acceptance and an actionable Git-checkout requirement for formal strategy training.

No capable chat weights, completed billion-token training, qualified multi-teacher RL or measured trained-draft acceleration are released. Diagnostic completion and declining NLL do not establish model capability. A private reporting contact remains to be specified before a formal release; see [release scope](docs/releases/v0.1.0.md).
