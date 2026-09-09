# Changelog

## Unreleased

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

No capable chat weights, completed billion-token training, qualified multi-teacher RL or measured trained-draft acceleration are released. Diagnostic completion and declining NLL do not establish model capability. Repository/contact metadata remains a public-release prerequisite; see [release scope](docs/releases/v0.1.0.md).
