# Third-party notices

MiniFrontier-authored code is Apache-2.0. Attributed upstream components and snapshots retain their original licenses; the project license does not override them. No upstream model weights are distributed.

## Distribution license mapping

The wheel and sdist declare `Apache-2.0 AND MIT AND LicenseRef-Kimi-K3` for their combined contents. This preserves the individual component terms and does not relicense the Kimi derivatives as Apache-2.0. Per the [PyPA License-Expression specification](https://packaging.python.org/en/latest/specifications/core-metadata/#license-expression), this field describes the containing distribution archive.

| Distributed component | Applicable terms | Standalone license |
|---|---|---|
| Original MiniFrontier integration, scripts and authored documentation | Apache-2.0 | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| Qwen Transformers and vLLM derived computational/vision/MTP files | Apache-2.0; retained attribution | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| Kimi attributed text, AttnRes, vision/processing, MTP and related derived computation, including its grouped-expert branch | LicenseRef-Kimi-K3; full custom terms, including commercial-use conditions | [Kimi K3](LICENSES/LicenseRef-Kimi-K3.txt) |
| DeepSeek attributed text, compression, vision and DSpark-derived computation | MIT; retained attribution | [MIT](LICENSES/MIT-DeepSeek.txt) |
| MiniFrontier1 fusion: KDA/MLA/LatentMoE adaptations, GR/ViT reuse, CSA-derived pooling and local integration | Respectively LicenseRef-Kimi-K3, Apache-2.0, MIT and Apache-2.0; combining these modules does not remove the source terms | [MF1 source map](configs/minifrontier1/source-map.json) and the standalone licenses above |
| Upstream snapshots (sdist; not wheel reference resources) | Each snapshot's original license as listed below | Original LICENSE beside each snapshot |
| Dependencies installed separately | Their own licenses; not relicensed by this package | Refer to each dependency distribution |

Full standalone license texts and these notices are included in distribution license files. File-level notices and the pinned sources below provide the finer-grained mapping. Source-derived modifications keep the applicable upstream terms. Data and model weights are not distributed in this release; any later artifacts require their own provenance and license declarations.

## Qwen

Source: [Hugging Face Transformers, 4177486a9f199bd7be520eff14431071d5d41ec5](https://github.com/huggingface/transformers/tree/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/qwen4_exp).

The [snapshot](third_party/upstream/qwen4_exp-4177486) retains the original modeling, configuration and cache sources with their Apache-2.0 [LICENSE](third_party/upstream/qwen4_exp-4177486/LICENSE).
Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Files named `upstream_*.py` in [MiniQwen4](minifrontier/models/miniqwen4) extract upstream computational definitions with attribution. Integration adapters remove unavailable Hub dispatch and dependency-specific hooks; computation is checked against the original snapshot. Local model, cache, QSA objective and optimizer adapters are not claims of complete HF API or fused-kernel equivalence.

The separate Qwen3.8-Flash-Next release artifacts use the Qwen Community License 1.0. The Apache-2.0 license of the Transformers source must not be used to describe those separate artifacts.

## Kimi

Source: [moonshotai/Kimi-K3, c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721).

The [snapshot](third_party/upstream/kimi-k3-c5d1dd4) retains the modeling source, configuration and full [Kimi K3 License](third_party/upstream/kimi-k3-c5d1dd4/LICENSE).
[attnres.py](minifrontier/models/minikimik3/attnres.py) and [upstream_layers.py](minifrontier/models/minikimik3/upstream_layers.py) preserve the full license for source and wheel distributions. The latter adapts the KDA/MLA/LatentMoE/decoder definitions for differentiable training; integration changes are listed in the extraction script. This component is **not Apache-2.0**. Read the original terms before use or redistribution.

## DeepSeek-V4

Source: [DeepSeek-V4-Flash, 60d8d70770c6776ff598c94bb586a859a38244f1](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1/inference).

The [snapshot](third_party/upstream/deepseek-v4-60d8d70) retains the inference source and original [MIT LICENSE](third_party/upstream/deepseek-v4-60d8d70/LICENSE), copyright 2023 DeepSeek. The local unquantized expert and compressor are checked against that source. `upstream_layers.py`, `attention.py` and `kernels.py` preserve attribution for source-derived training adaptations; no complete V4 training fidelity is implied.

## Architecture figure excerpts

Architecture figure excerpts in `docs/assets/upstream/` are attributed to the Kimi Team / Moonshot AI, Qwen Team, and DeepSeek-AI. They are limited excerpts from their technical reports, used for architecture explanation and comparison. Original rights are retained; these report figures are not relicensed under MiniFrontier's Apache-2.0 license. Source-code licenses listed above do not automatically cover separately published reports. Exact report revisions, figure/page numbers and extraction records are in the [figure attribution index](docs/assets/upstream/README.md).

## Numerical algorithms and dependencies

Polar Express coefficients in `minifrontier/training/polar_express.py` follow the authors' reference revision `71cc37943d99cae780024c1d198977f2f8795407` and the Qwen report's eight-step schedule. The local numerical implementation and its limitations are documented in the source.

PyTorch and development/monitoring dependencies retain their own licenses. The lockfile records the resolved environment, not a replacement for each dependency's licensing terms.

Architecture names and trademarks belong to their respective owners. MiniFrontier is not affiliated with the upstream organizations.

## Native vision, processing and training MTP adapters

The Kimi snapshot additionally includes the same pinned release's `modeling_kimi_k3.py`,
processor, image/video processing and media utilities. The extracted native vision and
processing files retain the Kimi K3 license; local wrappers preserve attribution.

Qwen native vision and processing definitions follow Transformers `4177486` (Apache-2.0).
The release's preprocessor metadata is separately attributed in
`docs/audits/strategy-source-snapshots.json`. The Qwen MTP adapter follows the
[vLLM source at 96eccb8](https://github.com/vllm-project/vllm/tree/96eccb8f49aff58aa2a11b431bf8331f4b368606/vllm/models/qwen4_exp/nvidia),
whose Apache-2.0 LICENSE is preserved in `third_party/upstream/qwen4-mtp-96eccb8`.

DeepSeek native vision, image layout and visibility follow
[DeepSeek-V4-Flash-Vision-Exp at 6821d6a](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/6821d6ad3681a4b137b066b76094fa82ebd0a380).
Its original MIT LICENSE is preserved beside the snapshot. Local training-specific
adaptations are distinguished from the original inference definitions.

The local Kimi EAGLE-style conversion and Qwen draft reuse these attributed MTP
components. Their unroll objectives, feature selection, optimizer and training
executor are explicit mini adaptations. `minideepseekv4/dspark.py` follows the
same pinned MIT-licensed Vision-Exp prefix/noise attention, mHC and Markov layout.
The local DSpark loss expresses the CE/L1/overlap-confidence equations quoted in
the repository's training strategy; the three text-MTP initializations and rank-64
mini adaptation do not claim released flagship training weights or hyperparameters.
The optional padded expert implementation preserves each source component's
activation and route weighting; its Kimi-derived computation retains the Kimi
K3 license. It remains disabled in the strategy configurations.

The strategy data builder records pinned source and item provenance. Its Python-Edu
pilot retrieves code blobs from Software Heritage and checks each content SHA1;
the index does not resolve each original repository license. That pilot explicitly
records `original-license-unresolved`, and is not represented as a fully reviewed
formal code corpus. Data source terms do not become the project's code license.

## Training data and kernel dependencies

The [data source guide](docs/guides/data-sources.md) distinguishes generated examples, historical experiments and the data used for first-stage formal pretraining. The first-stage text and visual components have been prepared; later stages require their own data bindings. Source terms and unresolved provenance remain attached to each component.

The optional CUDA KDA backend uses `fla-core==0.5.2`, from the MIT-licensed
[Flash Linear Attention project](https://github.com/fla-org/flash-linear-attention).
The local CPU recurrence is independently expressed from the KDA equations and
compared with that backend; FLA code is installed as a dependency, not vendored here.

The data builder acquires pinned prefixes of
[MiniMind-Dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset), whose
card declares Apache-2.0 and CC-BY-NC-2.0. Its provenance and licensing remain separate
from the MiniFrontier code license. Each generated manifest records the revision,
source URL, SHA256, split and tokenizer. Raw or processed data and trained weights
are not included in distributions. Locally generated arithmetic task definitions
are MiniFrontier-authored.
