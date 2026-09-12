<div align="center">

<img src="docs/assets/minifrontier-cover.png" alt="MiniFrontier — 折叠山峰形标识与项目名称" width="960" />

### 在小模型上学习和实践前沿架构

**读懂架构 · 从零训练 · 探索多模态**

<p>
  <a href="https://github.com/576469377/MiniFrontier/actions/workflows/ci.yml"><img src="https://github.com/576469377/MiniFrontier/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" alt="Main branch CI" /></a>
  <a href="docs/releases/v0.1.0.md"><img src="https://img.shields.io/badge/status-research_preview-d99a38?style=flat-square" alt="Research preview" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11+" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/built_with-PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch" /></a>
  <a href="THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/license-per_component-526675?style=flat-square" alt="Component licenses" /></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-287f8e?style=flat-square" alt="Contributions welcome" /></a>
</p>

[快速开始](#快速开始) · [模型家族](#模型家族) · [MF1 架构](#mf1-架构) · [实验与路线图](#实验与路线图) · [文档](docs/README.md) · [参与开发](CONTRIBUTING.md)

</div>

---

**MiniFrontier 是面向学习与实验的小模型项目。** 它参考 MiniMind / MiniMind-V 的实践方式，基于 Kimi、Qwen、DeepSeek 固定版本的公开源码，提供架构实现、数据处理、训练、恢复、评估和推理流程。

主线模型 **MiniFrontier1.0** 将三类架构中的设计组合成一个约 **228M** 参数的原生多模态模型；MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4 保留为独立实现，方便理解各自设计并开展对照实验。

第一次尝试建议从 **MF1 的 CPU 微型示例**开始；希望深入某一种架构时，再选择对应的来源模型。

| <img src="docs/assets/icon-source.svg" width="20" height="20" alt="" /> 阅读与改造 | <img src="docs/assets/icon-experiment.svg" width="20" height="20" alt="" /> 训练与复现 | <img src="docs/assets/icon-multimodal.svg" width="20" height="20" alt="" /> 文本与视觉 |
|:---|:---|:---|
| 对照固定版本的上游源码，理解注意力、MoE、缓存与 MTP 的实现。 | 从数据生成开始，亲手跑通训练、断点恢复、评估和生成。 | 在同一模型中接入文字、图片和视频帧，检查模型是否使用了视觉信息。 |

> [!NOTE]
> 当前为 **v0.1.0 研究预览**。截至 2026-09-12，四个模型均已启动首阶段正式预训练，完整训练尚未完成，尚无通过能力评估的公开聊天权重。先用下方 CPU 示例验证流程；训练进度与结果见[当前实验计划](docs/experiments/current-plan.md)。

## 快速开始

先在 CPU 上跑通一次完整流程。当前支持 **Linux、Python 3.11+**，使用 **[uv](https://docs.astral.sh/uv/getting-started/installation/)** 安装；Windows 用户请在 Linux 环境（如 WSL2）中执行。安装依赖后，示例会在本地生成数据，无需下载训练语料或模型权重。

**1. 安装项目**

```bash
git clone https://github.com/576469377/MiniFrontier.git
cd MiniFrontier
uv sync --locked --extra dev
uv run minifrontier models
```

**2. 生成数据，训练并验证恢复流程**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 quickstart --device cpu --output outputs/mf1-quickstart
```

示例使用约 **132K 参数**的小配置，生成算术、色块图片和视频帧样本，依次执行短程训练、暂停恢复、索引器训练、稀疏注意力续训和监督微调，并保存检查点与验证记录。该示例用于验证操作流程；步数很少，生成结果可能为空或错误。

完成后可在 `outputs/mf1-quickstart/` 查看各阶段记录；`sft/checkpoint.pt` 是下一步使用的权重。每次运行请换一个新输出目录。约 132K 的示例与 228M 研究配置采用不同容量，耗时和显存不能互相外推。

**3. 加载刚训练的权重**

```bash
uv run minifrontier mf1 generate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --prompt 'Color?' --image outputs/mf1-quickstart/data/media/train-32-0.png \
  --max-new-tokens 12 --temperature 0 --device cpu
```

<details>
<summary><b>在浏览器里查看这个实验</b></summary>

```bash
uv run minifrontier mf1 export \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt --output outputs/mf1-export
uv run minifrontier mf1 demo --checkpoint outputs/mf1-export/model.pt \
  --device cpu --port 7861 --allow-unqualified
```

打开 <http://127.0.0.1:7861>，可输入文字、上传图片或采样视频帧。`--allow-unqualified` 显式启用实验模式，页面会标明权重状态。这个微型示例没有经过聊天能力验收。

</details>

| 接下来想做什么？ | 入口 |
|:---|:---|
| 用自己的数据训练 MiniFrontier1.0 | [数据格式与训练命令](docs/guides/minifrontier1.md) |
| 在 CPU / 单张 3090 上运行三个来源模型 | [最小示例与实测耗时](docs/guides/quickstart.md) |
| 比较不同实验的生成结果 | [实验 Demo](docs/guides/demo-experiments.md) |
| 了解源码安装与 wheel 的支持范围 | [版本说明](docs/releases/v0.1.0.md) |

## 模型家族

MiniFrontier1.0 是融合主线；三个来源模型保留各自结构，供源码学习和对照实验。四个模型均从随机初始化开始训练。

| 模型 | 适合研究什么 | 研究配置参数量 | 结构与使用 |
|:---|:---|---:|:---|
| **MiniFrontier1.0** | 将递推记忆、历史压缩、稀疏检索与视觉输入组合在同一模型中 | **228M**，含视觉与 MTP | **[融合主线与结构图](docs/models/minifrontier1.md)** |
| MiniQwen4 | GDN / QSA 混合注意力、四路门控残差、浅层 n-gram 特征 | 513M，含视觉与 MTP | [结构图与运行入口](docs/models/miniqwen4.md) |
| MiniKimi-K3 | KDA / MLA 混合注意力、AttnRes 跨层汇合、潜在空间专家 | 205M，含视觉与 MTP | [结构图与运行入口](docs/models/minikimik3.md) |
| MiniDeepSeek-V4 | 局部窗口与历史压缩、mHC 残差约束、浅层 hash 路由 | 244M，文本与 MTP | [结构图与运行入口](docs/models/minideepseekv4.md) |

模型页包含数据流、配置、来源差异与代码入口。MF1 图示按本地实现绘制；三个来源模型另附[官方报告图示及出处](docs/assets/upstream/README.md)。

<details>
<summary><b>参数统计、配置与命名</b></summary>

| 模型 | 精确浮点参数量 | 统计范围 |
|:---|---:|:---|
| MiniFrontier1.0 | 228,235,809 | 主干、视觉模块与 MTP |
| MiniQwen4 | 513,405,536 | 主干、视觉模块与 MTP |
| MiniKimi-K3 | 204,526,216 | 主干、视觉模块与 MTP |
| MiniDeepSeek-V4 | 243,983,472 | 文本主干与 MTP |

表中为各研究配置的浮点参数总数，MoE 的总参数量与单次前向激活参数量不同。DeepSeek 另有 262,144 个固定整数 hash 路由项，不计入上述浮点参数；直接对全部 `model.parameters()` 求和会得到 244,245,616。微型示例使用更小的配置。

MiniFrontier1.0 的主配置为 **16 层 / hidden 512 / 32K 词表**，主干和视觉编码器均随机初始化。下方结构图展示实际层序；缓存、lookup 与训练目标见[模型说明](docs/models/minifrontier1.md)。

MiniFrontier1.0 的组合方案由本项目设计；MiniQwen4 的名称来自所参考源码中的 `qwen4_exp` 模块。各模型的来源版本与修改对应关系见[第三方说明](THIRD_PARTY_NOTICES.md)和[MF1 来源映射](configs/minifrontier1/source-map.json)。

</details>

## MF1 架构

文字和视觉特征先进入同一序列，再通过 **16 层、四路残差的 Decoder**。大多数层用 KDA 递推累积信息；第 4、12 层用 CSA 压缩历史，第 8、16 层用 QSA-MLA 选择性读取历史。每层都包含路由专家和共享 FFN，浅层 lookup 与辅助 MTP 分支分别标在图中。

<p align="center">
  <a href="docs/assets/minifrontier1-detail.svg"><img src="docs/assets/minifrontier1-detail.svg" alt="MF1 当前 228M 结构：文字与 ViT 特征汇合为四路状态；图示完整 16 层注意力顺序、第 2 层前的 lookup、每层 GR 读写与 LatentMoE，以及主输出和共享词表头的 MTP 分支。" width="960" /></a>
</p>

[单层 GR 与专家结构](docs/models/minifrontier1.md#decoder-与-gr) · [三种注意力对照](docs/models/minifrontier1.md#三种注意力如何读取历史) · [视觉输入](docs/models/minifrontier1.md#视觉输入与位置编码) · [MTP](docs/models/minifrontier1.md#mtp-辅助预测) · [精确参数](docs/models/minifrontier1.md#参数和训练阶段)

这张图描述当前研究实现。主干与视觉编码器均随机初始化；各组件组合后的效果、长上下文成本和草稿加速收益仍需实验验证。

## 实验与路线图

实验档案保留配置、命令、种子、数据与 tokenizer 校验值、实际训练量和曲线；完成、暂停与失败的实验分别标明。

| 范围 | 截至 2026-09-12 的状态 | 记录 |
|:---|:---|:---|
| 架构与流程 | 四个模型已有实现；离线示例覆盖数据、训练、恢复、评估和生成 | [代码结构](docs/architecture.md)、[最小示例](docs/guides/quickstart.md) |
| 小规模实验 | 完成部分配方比较及 MF1 机制学习；结果包含失败和停止项 | [实验档案](docs/experiments.md) |
| 正式预训练 | 四模型首阶段均已启动；完整预算为 Kimi 2B、Qwen 3B、DeepSeek 文本 2.5B、MF1 3B CE token | [当前计划](docs/experiments/current-plan.md)、[预训练方案](docs/pretraining-plan.md) |
| 后训练与权重 | 已有 SFT、RL、教师与草稿训练接口；正式后训练和权重发布待完成 | [后训练指南](docs/guides/posttraining-adaptation.md)、[发布范围](docs/releases/v0.1.0.md) |

<details>
<summary><b>已完成的 MF1 小配置学习实验，学到了什么？</b></summary>

约 132K 参数的小配置，在 CPU 两线程上训练 600 updates，耗时约 242 秒。留出集上的结果如下：

| 任务 | 原始输入 | 对照输入 |
|:---|:---:|:---:|
| 算术 | 0 / 6 | — |
| 色块图片问答 | 4 / 4 | 图片置黑后 1 / 4 |
| 色块视频问答 | 2 / 2 | 帧置黑后 0 / 2 |

模型使用了色块媒体中的信息，但没有解决留出的算术题；任务简单、样本很少，尚不足以证明通用多模态能力。这是微型配置的学习实验，与完整 228M 模型训练分开记录。

[完整报告与逐步曲线](docs/experiments/mf1-reference-v2/README.md) · [早期三个模型的训练失败复盘](docs/training-failure-v1.md)

</details>

当前优先推进已启动的基础预训练，按模型方案进入索引器与后续阶段，再进行评估、后训练和导出。每个模型默认使用一张卡；后续阶段的数据与计算预算单独准备。

MF1 主预训练预算为 **30 亿个参与主语言损失计算的 token**，索引器、辅助目标及后训练另行统计。当前执行顺序见[预训练主计划](docs/pretraining-plan.md)；架构及专项技术细节见[完整方案](docs/training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md)与[阶段配置](configs/minifrontier1)。三个来源模型的路线见[训练指南](docs/guides/training.md)。

## 文档与开发

| 了解项目 | 动手实践 | 核查与复现 |
|:---|:---|:---|
| [文档导航](docs/README.md) | [MF1 全流程指南](docs/guides/minifrontier1.md) | [实验档案](docs/experiments.md) |
| [代码架构](docs/architecture.md) | [CPU / 3090 最小示例](docs/guides/quickstart.md) | [数据来源与处理](docs/guides/data-sources.md) |
| [版本说明](docs/releases/v0.1.0.md) | [后训练实践](docs/guides/posttraining-adaptation.md) | [上游组件与许可](THIRD_PARTY_NOTICES.md) |

<details>
<summary><b>目录结构与开发检查</b></summary>

```text
minifrontier/
├── models/       四个模型的结构实现
├── data/         数据准备、清洗与编码
├── training/     训练、恢复、优化器与后训练
├── evaluation/   生成评测与多模态对照
├── inference/    权重加载、生成、导出与 Demo
└── commands/     命令编排与最小示例
configs/          模型配置与训练配方
scripts/          数据提取、实验调度和分析
tests/            模型、训练和安装测试
docs/             使用指南、模型说明与实验记录
third_party/      固定版本的上游源码及许可
```

本地数据与训练产物分别存放在根目录的 `data/` 和 `outputs/`，不会随源码提交。

```bash
uv sync --locked --extra dev --extra monitoring --extra data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest -q -m 'not cuda'
uv run ruff check .
uv run ruff format --check .
uv run mypy minifrontier scripts --ignore-missing-imports
```

</details>

欢迎改进文档、阅读和实现模型模块，或提交有完整记录的实验。开始前请阅读[贡献指南](CONTRIBUTING.md)，问题与想法可以提交到 [Issues](https://github.com/576469377/MiniFrontier/issues)。

## 数据与许可

离线示例使用程序生成的数据。首阶段正式预训练采用公开中英文教育文本、数学网页、经筛选的 Python 代码和对话数据；视觉模型另加入自然图像、文档、图表及生成 OCR。来源、固定版本、处理方法和已知限制见[数据说明](docs/guides/data-sources.md)。原始语料、检查点与训练产物不随源码或安装包分发。

本项目原创代码采用 [Apache-2.0](LICENSE)。Qwen 的 Transformers / vLLM 组件保留 Apache-2.0，DeepSeek 组件保留 [MIT](LICENSES/MIT-DeepSeek.txt)，Kimi 组件保留含特定商业使用条件的 [Kimi K3 License](LICENSES/LicenseRef-Kimi-K3.txt)。组件对应关系见[第三方说明](THIRD_PARTY_NOTICES.md)，数据按各自来源条款管理。

---

<div align="center">

<img src="docs/assets/minifrontier-mark.svg" width="40" height="40" alt="MiniFrontier 图标" />

感谢 [MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v)，以及 Kimi、Qwen、DeepSeek 等项目公开的源码和研究资料。

</div>
