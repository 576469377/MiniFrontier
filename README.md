# MiniFrontier

MiniFrontier 是一个学习和实践大模型的开源项目。我们参考 MiniMind、MiniMind-V 的教学思路，从 Kimi、Qwen 和 DeepSeek 的公开源码出发，缩小模型规模，提供从数据准备到训练、评估和推理的实践流程。

目前主要开发 **MiniFrontier1.0**：结合三类架构设计的小型原生多模态模型，支持文本、图片和视频帧输入。MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4 则保留为独立实现，方便阅读源码和比较实验结果。

项目以单张 RTX 3090 等消费级显卡上的训练实践为目标，同时提供可在 CPU 上运行的微型示例。当前版本为 **v0.1.0 研究预览**，尚未发布可用的聊天模型权重。

[开始使用](#开始使用) · [文档导航](docs/README.md) · [模型说明](docs/models/minifrontier1.md) · [实验记录](docs/experiments.md) · [数据来源](docs/guides/data-sources.md) · [许可证](#许可证)

## 模型

| 模型 | 主要结构 | 研究配置参数量 |
|---|---|---:|
| **[MiniFrontier1.0](docs/models/minifrontier1.md)** | KDA / CSA / QSA-MLA 多通路注意力、GR、LatentMoE、ViT、MTP | **228,235,809** |
| [MiniQwen4](docs/models/miniqwen4.md) | GDN、GR、PLE、MoE、QSA、原生视觉、MTP | 513,405,536 |
| [MiniKimi-K3](docs/models/minikimik3.md) | KDA、MLA、LatentMoE、AttnRes、MoonViT、MTP | 204,526,216 |
| [MiniDeepSeek-V4](docs/models/minideepseekv4.md) | SWA、CSA/HCA、MoE、mHC、文本 MTP | 243,983,472 |

MiniFrontier1.0 的主配置为 16 层、隐藏维度 512、32K 词表，主干和视觉编码器均采用随机初始化。表中参数量包含各配置实际启用的模块：前三项包含视觉模块，MiniDeepSeek-V4 对应文本配置。微型示例使用更小的配置。

MiniFrontier1.0 的组合方案由本项目设计。MiniQwen4 的名称来自所参考源码的 `qwen4_exp` 模块；各模型的具体来源、修改和许可见[第三方说明](THIRD_PARTY_NOTICES.md)。

## 当前进展

以下状态更新于 **2026-09-09**。

| 工作 | 进展 |
|---|---|
| 模型实现 | 四个模型均有可运行代码；MiniFrontier1.0 已完成完整 228M 配置的 CPU 前后向检查 |
| 训练流程 | 微型示例已跑通数据生成、训练、暂停恢复、评估和生成；后训练与导出提供参考实现 |
| 训练实验 | 三个来源模型已开展小规模配方比较；MiniFrontier1.0 完成了合成数据上的学习实验 |
| 完整模型训练 | MiniFrontier1.0 的正式数据、3090 性能测试及 30 亿 token 主预训练尚未完成 |
| 对话与多模态能力 | 尚无通过能力评估的聊天权重；训练效果和失败案例随实验记录公开 |

MiniFrontier1.0 的小配置在色块图像和视频题上学到了简单规律，但留出的 6 道算术题全部回答错误，详见[小规模学习实验](docs/experiments/mf1-reference-v2/README.md)。此前三个模型的短程训练也未达到基本对话目标，见[失败复盘](docs/training-failure-v1.md)。

## 开始使用

需要 Python 3.11+ 和 `uv`。安装依赖后，下面的 CPU 示例会在本地生成数据，无需下载训练语料或模型权重：

```bash
git clone https://github.com/576469377/MiniFrontier.git
cd MiniFrontier
uv sync --locked --extra dev

uv run minifrontier models
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 quickstart --device cpu --output outputs/mf1-quickstart

uv run minifrontier mf1 generate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --prompt 'Color?' --image outputs/mf1-quickstart/data/media/train-32-0.png \
  --max-new-tokens 12 --temperature 0 --device cpu
```

示例使用约 13.2 万参数的小模型，生成算术、色块图片和视频帧样本，并保存训练记录与检查点。它用于了解完整操作流程；训练步数很少，生成结果可能为空或错误。

- [MiniFrontier1.0 使用指南](docs/guides/minifrontier1.md)：数据格式、分阶段训练、恢复、评估、导出和多模态 Demo。
- [三个来源模型的最小示例](docs/guides/quickstart.md)：CPU 和单张 3090 的运行方法与实测耗时。
- [实验 Demo](docs/guides/demo-experiments.md)：显式加载本地实验权重，观察不同阶段的输出。

浏览器默认只列出通过能力评估的权重。查看微型示例或未完成训练的模型时，请按对应指南使用实验模式。安装包的支持范围见[版本说明](docs/releases/v0.1.0.md)。

## 训练计划

MiniFrontier1.0 按以下顺序推进：

```text
准备数据和词表 → 小规模架构对比 → 多模态预训练
→ 索引器训练与稀疏注意力续训 → 监督微调
→ 强化学习与教师蒸馏 → 草稿模型训练 → 导出和能力评估
```

主预训练计划使用 **30 亿个参与损失计算的 token**，监督微调、索引器训练和其他辅助目标另行统计。默认每个训练任务使用一张卡，多张卡用于独立的配方对比或教师训练。各阶段的预算、数据比例和进入条件见[训练方案](docs/training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md)与[配置目录](configs/minifrontier1)。

三个来源模型保留各自的训练路线，见[训练指南](docs/guides/training.md)。配置、数据版本、随机种子、实际训练量和评估结果整理在[实验档案](docs/experiments.md)中，各组分别注明完成情况。

## 数据

MiniFrontier1.0 目前的微型示例和学习实验使用程序生成的算术、色块图像及视频帧数据，完整模型的训练数据尚未准备完成。

三个来源模型的小规模实验使用过 MiniMind 数据集、Fineweb-Edu-Chinese、FineWeb-Edu、SmolLM-Corpus 的 Python-Edu 子集和 UltraChat；视觉试验使用 FineVision 中的少量 ALLaVA 样本。具体版本、用途、处理方法和已知问题见[数据来源说明](docs/guides/data-sources.md)。

仓库提供数据准备脚本和实验记录。实际训练语料保存在本地，由各来源的数据使用条款约束。

## 项目结构

```text
minifrontier/
├── models/       四个模型的结构实现
├── data/         数据准备、清洗与编码
├── training/     训练、恢复、优化器与后训练
├── evaluation/   生成评测与多模态对照
├── inference/    权重加载、生成、导出与 Demo
└── commands/     命令编排与最小示例
configs/          模型配置与训练配方
scripts/          数据提取、实验调度和分析工具
tests/            模型、训练和安装测试
docs/             使用指南、模型说明与实验记录
third_party/      固定版本的上游源码及许可
```

目录职责和扩展约定见[架构说明](docs/architecture.md)。本地数据与训练产物分别保存在根目录的 `data/` 和 `outputs/`，不会随源码提交。

## 参与开发

欢迎提交问题、改进文档、补充实验或参与实现。请阅读[贡献指南](CONTRIBUTING.md)，并在 [Issues](https://github.com/576469377/MiniFrontier/issues) 中提供可复现的命令和必要环境信息。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest -q -m 'not cuda'
uv run ruff check .
uv run ruff format --check .
uv run mypy minifrontier scripts --ignore-missing-imports
```

## 许可证

本项目原创代码采用 [Apache-2.0](LICENSE)。使用和修改的上游代码保留各自许可证：

| 来源组件 | 许可证 |
|---|---|
| Qwen 的 Transformers / vLLM 实现 | Apache-2.0 |
| Kimi-K3 相关组件 | [Kimi K3 License](LICENSES/LicenseRef-Kimi-K3.txt)，包含特定商业使用条件 |
| DeepSeek-V4 相关组件 | [MIT](LICENSES/MIT-DeepSeek.txt) |

各组件对应文件及来源版本见[第三方说明](THIRD_PARTY_NOTICES.md)。训练数据的许可另见[数据来源说明](docs/guides/data-sources.md)。

感谢 [MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v) 以及 Kimi、Qwen、DeepSeek 等项目公开的源码和研究资料。
