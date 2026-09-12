<div align="center">

<img src="docs/assets/minifrontier-cover.png" alt="MiniFrontier — 折叠山峰形标识与项目名称" width="960" />

### 小规模语言与多模态模型的实现和训练

<p>
  <a href="https://github.com/576469377/MiniFrontier/actions/workflows/ci.yml"><img src="https://github.com/576469377/MiniFrontier/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" alt="Main branch CI" /></a>
  <a href="docs/releases/v0.1.0.md"><img src="https://img.shields.io/badge/status-research_preview-d99a38?style=flat-square" alt="Research preview" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11+" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/built_with-PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch" /></a>
  <a href="THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/license-per_component-526675?style=flat-square" alt="Component licenses" /></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-287f8e?style=flat-square" alt="Contributions welcome" /></a>
</p>

[模型家族](#模型家族) · [MF1 架构](#mf1-架构) · [快速开始](#快速开始) · [实验与路线图](#实验与路线图) · [文档](docs/README.md) · [参与开发](CONTRIBUTING.md)

</div>

---

MiniFrontier 以 **MiniFrontier1.0（MF1）** 为主线，提供小规模模型的结构实现、数据处理、训练、恢复、评估和推理工具。MF1 将递推注意力、历史压缩、稀疏检索和潜在空间专家组合为一个 **228M 参数的原生多模态模型**；三个来源模型保留独立结构，供源码对照和实验。

> [!NOTE]
> 当前为 **v0.1.0 研究预览**。截至 2026-09-12，四个模型均已启动首阶段正式预训练；完整训练和能力评估尚未完成，仓库不附带聊天权重。训练安排见[预训练主计划](docs/pretraining-plan.md)。

## 模型家族

四个模型均从随机初始化开始训练。点击模型名称查看结构图、模块计算、配置与实现代码。

| 模型 | 研究配置 | 输入接口 | 主要结构 |
|:---|---:|:---|:---|
| **[MiniFrontier1.0](docs/models/minifrontier1.md)** | **228M** | 文本、图像、视频帧 | KDA / CSA / QSA-MLA、四路 GR、LatentMoE、lookup |
| [MiniQwen4](docs/models/miniqwen4.md) | 513M | 文本、图像、视频帧 | GDN / QSA、四路 GR、PLE、MoE |
| [MiniKimi-K3](docs/models/minikimik3.md) | 205M | 文本、图像、视频帧 | KDA / MLA、AttnRes、LatentMoE |
| [MiniDeepSeek-V4](docs/models/minideepseekv4.md) | 244M | 文本；视觉扩展单列 | SWA / CSA / HCA、mHC、MoE（浅层 hash 路由） |

参数量包含 MTP，前三个模型还包含视觉编码器。输入接口的训练覆盖与评估状态在各模型页单列。MF1 结构图按本地实现绘制；三个来源模型附[官方报告图及出处](docs/assets/upstream/README.md)。

<details>
<summary><b>参数统计、配置与命名</b></summary>

| 模型 | 精确浮点参数量 | 统计范围 |
|:---|---:|:---|
| MiniFrontier1.0 | 228,235,809 | 主干、视觉模块与 MTP |
| MiniQwen4 | 513,405,536 | 主干、视觉模块与 MTP |
| MiniKimi-K3 | 204,526,216 | 主干、视觉模块与 MTP |
| MiniDeepSeek-V4 | 243,983,472 | 文本主干与 MTP |

表中为各研究配置的浮点参数总数，MoE 的总参数量与单次前向激活参数量不同。DeepSeek 另有 262,144 个固定整数 hash 路由项，不计入上述浮点参数；直接对全部 `model.parameters()` 求和会得到 244,245,616。微型示例使用更小的配置。

MiniFrontier1.0 的组合方案由本项目设计；MiniQwen4 的名称来自所参考源码中的 `qwen4_exp` 模块。各模型的来源版本与修改对应关系见[第三方说明](THIRD_PARTY_NOTICES.md)和[MF1 来源映射](configs/minifrontier1/source-map.json)。

</details>

## MF1 架构

主配置为 **16 层、hidden 512、32K 词表**。文字与 ViT 视觉特征进入同一序列，经四路残差 Decoder 处理：第 4、12 层用 CSA 压缩历史，第 8、16 层用 QSA-MLA 读取历史，其余层用 KDA 递推。每层包含路由专家和共享 FFN；浅层 lookup 补充局部特征，MTP 分支提供辅助预测目标。

<p align="center">
  <a href="docs/assets/minifrontier1-detail.svg"><img src="docs/assets/minifrontier1-detail.svg" alt="MF1 当前 228M 结构：文字与 ViT 特征汇合为四路状态；图示完整 16 层注意力顺序、第 2 层前的 lookup、每层 GR 读写与 LatentMoE，以及主输出和共享词表头的 MTP 分支。" width="960" /></a>
</p>

[下载完整结构图](docs/assets/minifrontier1-detail.svg) · [单层 GR 与专家](docs/models/minifrontier1.md#decoder-与-gr) · [注意力对照](docs/models/minifrontier1.md#三种注意力如何读取历史) · [视觉输入](docs/models/minifrontier1.md#视觉输入与位置编码) · [MTP](docs/models/minifrontier1.md#mtp-辅助预测)

## 快速开始

以下示例在 CPU 上生成数据、训练并加载 MF1 微型权重。需要 **Linux、Python 3.11+** 和 **[uv](https://docs.astral.sh/uv/getting-started/installation/)**；Windows 可使用 WSL2。安装依赖后无需下载语料或模型权重。

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

示例使用约 **132K 参数**的配置和生成的算术、色块图片、视频帧，执行短程训练、暂停恢复、索引器训练、稀疏续训和 SFT。运行结束后，在 `outputs/mf1-quickstart/` 查看各阶段日志和检查点；下一步加载 `sft/checkpoint.pt`。

每次重跑需换一个新输出目录。微型示例用于检查流程，生成结果可能为空或错误；耗时和显存不适用于 228M 研究配置。

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

打开 <http://127.0.0.1:7861>，可输入文字、上传图片或采样视频帧。`--allow-unqualified` 允许加载尚未通过能力评估的实验权重，页面会显示其状态。

</details>

| 任务 | 入口 |
|:---|:---|
| 用自己的数据训练 MiniFrontier1.0 | [数据格式与训练命令](docs/guides/minifrontier1.md) |
| 在 CPU / 单张 3090 上运行三个来源模型 | [最小示例与实测耗时](docs/guides/quickstart.md) |
| 比较不同实验的生成结果 | [实验 Demo](docs/guides/demo-experiments.md) |
| 了解源码安装与 wheel 的支持范围 | [版本说明](docs/releases/v0.1.0.md) |

## 实验与路线图

| 范围 | 截至 2026-09-12 的状态 | 记录 |
|:---|:---|:---|
| 架构与流程 | 四个模型已有实现；离线示例覆盖数据、训练、恢复、评估和生成 | [代码结构](docs/architecture.md)、[最小示例](docs/guides/quickstart.md) |
| 小规模实验 | 完成部分配方比较及 MF1 机制学习；结果包含失败和停止项 | [实验档案](docs/experiments.md) |
| 正式预训练 | 四模型首阶段均已启动；主 CE 预算为 Kimi 2B、Qwen 3B、DeepSeek 文本 2.5B、MF1 3B token | [阶段、数据与配方](docs/pretraining-plan.md) |
| 后训练与权重 | 已有 SFT、RL、教师与草稿训练接口；正式后训练和权重发布待完成 | [后训练指南](docs/guides/posttraining-adaptation.md)、[发布范围](docs/releases/v0.1.0.md) |

<details>
<summary><b>MF1 小配置：600 次更新的学习结果</b></summary>

约 132K 参数的小配置，在 CPU 两线程上训练 600 updates，耗时约 242 秒。留出集上的结果如下：

| 任务 | 原始输入 | 对照输入 |
|:---|:---:|:---:|
| 算术 | 0 / 6 | — |
| 色块图片问答 | 4 / 4 | 图片置黑后 1 / 4 |
| 色块视频问答 | 2 / 2 | 帧置黑后 0 / 2 |

模型能利用色块信息，未解决留出的算术题。这组微型配置实验的样本量不足以评价通用多模态能力。

[完整报告与逐步曲线](docs/experiments/mf1-reference-v2/README.md) · [早期三个模型的训练失败复盘](docs/training-failure-v1.md)

</details>

后续按各模型计划完成基础预训练、索引器训练和能力评估，再推进后训练与权重发布。CE 预算只统计参与主语言损失的位置，索引器及辅助目标另计。

## 文档与开发

| 了解项目 | 动手实践 | 核查与复现 |
|:---|:---|:---|
| [全部文档](docs/README.md) | [MF1 全流程指南](docs/guides/minifrontier1.md) | [实验档案](docs/experiments.md) |
| [代码架构](docs/architecture.md) | [CPU / 3090 最小示例](docs/guides/quickstart.md) | [数据来源与处理](docs/guides/data-sources.md) |
| [版本说明](docs/releases/v0.1.0.md) | [后训练实践](docs/guides/posttraining-adaptation.md) | [上游组件与许可](THIRD_PARTY_NOTICES.md) |

<details>
<summary><b>目录结构</b></summary>

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

</details>

开发环境、检查命令和提交约定见[贡献指南](CONTRIBUTING.md)。问题反馈使用 [Issues](https://github.com/576469377/MiniFrontier/issues)。

## 数据与许可

离线示例使用程序生成的数据。首阶段正式预训练采用公开中英文教育文本、数学网页、经筛选的 Python 代码和对话数据；视觉模型另加入自然图像、文档、图表及生成 OCR。来源、固定版本、处理方法和已知限制见[数据说明](docs/guides/data-sources.md)。原始语料、检查点与训练产物不随源码或安装包分发。

本项目原创代码采用 [Apache-2.0](LICENSE)。Qwen 的 Transformers / vLLM 组件保留 Apache-2.0，DeepSeek 组件保留 [MIT](LICENSES/MIT-DeepSeek.txt)，Kimi 组件保留含特定商业使用条件的 [Kimi K3 License](LICENSES/LicenseRef-Kimi-K3.txt)。组件对应关系见[第三方说明](THIRD_PARTY_NOTICES.md)，数据按各自来源条款管理。

---

<div align="center">

<img src="docs/assets/minifrontier-mark.svg" width="40" height="40" alt="MiniFrontier 图标" />

感谢 [MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v)，以及 Kimi、Qwen、DeepSeek 等项目公开的源码和研究资料。

</div>
