# Contributing

MiniFrontier develops a small multimodal fusion model and three source-derived reference architectures. Start with the [documentation index](docs/README.md) and [code layout](docs/architecture.md).

## Changes and evidence

Describe the problem, the resulting behavior and how you checked it. Link performance and model-quality claims to a specific configuration, source revision and measurement. Distinguish implementation checks from trained capability.

For source-derived computation, pin the upstream revision, preserve attribution and state any capacity or algorithm changes. Compare forward values and gradients against the original implementation where possible. Update the model page and source mapping when the architecture changes.

Documentation should work for someone using a fresh clone. Use concise explanations, runnable examples and relative links. Keep hardware identifiers, local paths and discussion history out of general guides; date experiment results and retain their limitations.

## Repository layout

- Put model structures in `models/<model>/` and their data, training, evaluation or inference logic in the corresponding functional directories. Commands and browser handlers call those modules.
- Put reusable instructions in `docs/guides/`, model explanations in `docs/models/`, and dated numeric results in `docs/experiments/`. See the [extension rules](docs/architecture.md#扩展与迁移约定).
- Preserve frozen strategy documents and historical measurements. Amend a plan through a new version; do not modify source checkouts or directories used by active training jobs.
- Keep raw datasets, weights, credentials and complete runtime outputs out of commits and packages. Review public experiment excerpts for sample content and machine identifiers.

## Checks

Run the checks relevant to the change. Documentation-only edits need link, example and layout checks; training changes need corresponding state or numerical tests.

```bash
uv sync --locked --extra dev --extra monitoring --extra data
uv run ruff check .
uv run ruff format --check .
uv run mypy minifrontier scripts --ignore-missing-imports
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest -q -m 'not cuda'
uv run python -m build --no-isolation
uv run twine check dist/*
```

Use available GPUs for bounded tests when required; include hardware, precision and known limitations in the report. A long training run is not a prerequisite for an unrelated engineering change.

## Licensing

Original contributions use Apache-2.0. Source-derived changes retain their applicable upstream terms; update [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/` when adding a component. Dataset, tokenizer and checkpoint licenses are separate from the code license.
