# Contributing

MiniFrontier develops the MiniFrontier1.0 native fusion model and three source-aligned baselines. Contributions must distinguish implemented, verified, inferred and undisclosed behavior. Start with the [documentation index](docs/README.md) and [code layout](docs/architecture.md).

1. Pin an authoritative upstream revision and retain its license.
2. Scale capacity explicitly; do not silently replace algorithmic modules.
3. Add independent same-weight forward/gradient tests against the upstream source.
4. Update the model catalog and per-model documentation without overstating training readiness.
5. Keep raw datasets, weights, credentials and full local outputs out of commits and distributions. Small, reviewed numeric records and plots belong in `docs/experiments/`; remove machine identifiers and sample contents.
6. Put reusable instructions in `docs/guides/` and machine-specific scheduling notes in `docs/operations/`. Preserve dated evidence and SHA-bound strategy originals; create a new version for a changed strategy. Do not move active run directories or edit their frozen source copies.
7. Organize Python code by responsibility: `models/`, `data/`, `training/`, `evaluation/`, `inference/` and `commands/`. Keep model-specific behavior in explicitly named modules under those directories. Reuse shared runtime utilities and keep training algorithms out of command/browser modules; follow the [extension and migration rules](docs/architecture.md#扩展与迁移约定).
8. Review public documentation for someone using a fresh clone. Explain internal experiment names, label plans and dated results, and link numerical/capability claims to evidence. Keep installation examples independent of the maintainer's GPU numbering and local paths. State data and component licenses separately, and remove private identifiers or unreviewed samples before publishing reports. Preserve historical metrics when producing a sanitized public copy.

Local checks:

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy minifrontier scripts --ignore-missing-imports
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest -q -m 'not cuda'
uv run python -m build --no-isolation
uv run twine check dist/*
```

Original contributions use Apache-2.0; source-derived changes retain their upstream terms and update THIRD_PARTY_NOTICES.md and LICENSES.

GPU tests require explicitly available devices. Never start a long training run or publish results as formal training merely to satisfy an engineering test. Include hardware, precision, source revision and known limitations with numerical claims.
