# Model card: `<checkpoint-name>`

> Replace every angle-bracket field before publishing weights. A smoke/toy checkpoint must be
> labelled as such and must not be described as reproducing an upstream model's capabilities.

## Identity and provenance

- MiniFrontier version and immutable commit: `<version> / <sha>`
- Architecture and model config digest: `<architecture> / <sha256>`
- Resolved pipeline/recipe digest: `<sha256>`
- Parent checkpoint(s): `<name, revision, license>`
- Upstream architecture sources: `<paper/repository revisions>`

## Training

- Stages completed: `<pretrain, ...>`
- Tokens/samples per stage: `<counts>`
- Hardware, precision and software: `<GPU/count, driver, CUDA, PyTorch>`
- Peak memory, throughput and wall time: `<measured values>`
- Random seeds and determinism settings: `<values>`

## Data and tokenizer

- Dataset manifest URI/digest: `<immutable manifest>`
- Dataset licenses and use restrictions: `<SPDX/exact terms>`
- Personal/sensitive-data handling: `<process>`
- Tokenizer source, vocabulary, digest and license: `<details>`

## Evaluation

Report versioned benchmark code, prompts, decoding parameters, sample counts and uncertainty.
Separate text, vision, reasoning, safety and contamination checks. Toy loss or successful smoke
generation is engineering evidence, not a capability score.

## Intended use and limitations

Describe supported languages/modalities, expected users, out-of-scope uses, context limits,
known architecture approximations, failure modes and deployment controls. State that model output
is untrusted and must not be executed or used for high-stakes decisions without independent review.

## License and attribution

State the checkpoint, tokenizer and data licenses independently. MiniFrontier's Apache-2.0 code
license does not automatically apply to trained artifacts. Preserve upstream notices and make clear
that miniature architecture names are descriptive and do not imply upstream affiliation.
