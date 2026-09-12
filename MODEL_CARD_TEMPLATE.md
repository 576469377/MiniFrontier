# Model card: `<checkpoint-name>`

Replace each placeholder before publishing weights. Label diagnostic checkpoints and distinguish completed training from evaluated capabilities.

## Identity and provenance

- MiniFrontier version and immutable commit: `<version> / <sha>`
- Architecture and model config digest: `<architecture> / <sha256>`
- Resolved pipeline/recipe digest: `<sha256>`
- Parent checkpoint(s): `<name, revision, license>`
- Upstream architecture sources: `<paper/repository revisions>`
- Download and integrity check: `<artifact URL, filename, SHA256>`

## Training

- Stages completed: `<pretrain, ...>`
- Tokens/samples per stage: `<main CE, input, auxiliary targets, independent samples and repeated exposures>`
- Hardware, precision and software: `<GPU/count, driver, CUDA, PyTorch>`
- Peak memory, throughput and wall time: `<measured values>`
- Random seeds and determinism settings: `<values>`

## Data and tokenizer

- Dataset manifest URI/digest: `<immutable manifest>`
- Dataset licenses and use restrictions: `<SPDX/exact terms>`
- Personal/sensitive-data handling: `<process>`
- Tokenizer source, vocabulary, digest and license: `<details>`

## Evaluation

Report benchmark versions, prompts, decoding parameters, sample counts, results and uncertainty. Include failed results and modality-specific controls. Give the scope of contamination checks and distinguish validation loss from independent capability evaluation.

## Inference

Provide a tested loading and generation command, required software, tokenizer/config paths, supported precision and measured memory requirements. State any restrictions on quantized or draft execution.

## Intended use and limitations

Describe evaluated languages and modalities, intended users, context limits, architecture approximations and observed failure modes. State which lengths, tasks and deployment settings remain unevaluated.

## License and attribution

State the checkpoint, tokenizer and data licenses independently. MiniFrontier's Apache-2.0 code
license does not automatically apply to trained artifacts. Preserve upstream notices and make clear
that miniature architecture names are descriptive and do not imply upstream affiliation.
