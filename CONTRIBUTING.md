# Contributing

MiniFrontier develops a small multimodal fusion model and three source-derived reference architectures. Start with the [documentation index](docs/README.md) and [code layout](docs/architecture.md).

## Changes and evidence

Describe the problem, the resulting behavior and how you checked it. Link performance and model-quality claims to a specific configuration, source revision and measurement. Distinguish implementation checks from trained capability.

For source-derived computation, pin the upstream revision, preserve attribution and state any capacity or algorithm changes. Compare forward values and gradients against the original implementation where possible. Update the model page and source mapping when the architecture changes.

Write documentation for a reader using a fresh clone. Explain the operation, expected output and relevant limitations; use runnable examples and relative links. Keep machine identifiers and discussion history out of general guides. Date experimental results and preserve unsuccessful outcomes.

## Repository layout

- Put model structures in `models/<model>/` and their data, training, evaluation or inference logic in the corresponding functional directories. Commands and browser handlers call those modules.
- Put reusable instructions in `docs/guides/`, model explanations in `docs/models/`, and dated numeric results in `docs/experiments/`. See the [extension rules](docs/architecture.md#扩展与迁移约定).
- Maintain current training arrangements in `docs/pretraining-plan.md`. Link to the canonical page instead of copying budgets, commands or status into parallel plans. Keep existing public links working when moving a page.
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

Run GPU tests only on devices available for that work; record the hardware, precision and limitations. On a training host, use a separate development environment and leave active jobs, their source copies and their data unchanged.

## Licensing

Original contributions use Apache-2.0. Source-derived changes retain their applicable upstream terms; update [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/` when adding a component. Dataset, tokenizer and checkpoint licenses are separate from the code license.
