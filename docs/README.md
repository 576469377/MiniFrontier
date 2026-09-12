# 文档导航

首次使用建议先运行 MiniFrontier1.0 的离线示例；对照上游架构学习时，可以选择 MiniQwen4、MiniKimi-K3 或 MiniDeepSeek-V4。训练进度见[当前实验计划](experiments/current-plan.md)，已完成和停止的实验见[实验索引](experiments.md)。

## 开始实践

| 需要做什么 | 入口 |
|---|---|
| 安装并跑通融合模型的数据、训练、恢复和生成 | [MiniFrontier1.0 操作指南](guides/minifrontier1.md) |
| 了解实际使用的数据、处理方法和数据许可 | [数据来源说明](guides/data-sources.md) |
| 理解融合架构和当前能力边界 | [模型说明](models/minifrontier1.md)、[实现与验收](audits/minifrontier1-implementation.md) |
| 运行三个来源模型的离线 CPU / 3090 示例 | [来源模型最小示例](guides/quickstart.md) |
| 查看三个来源模型的训练、后训练与草稿流程 | [训练](guides/training.md)、[后训练](guides/posttraining-adaptation.md)、[草稿](guides/draft-adaptation.md) |
| 在浏览器中观察来源模型实验权重 | [实验 Demo](guides/demo-experiments.md) |
| 阅读实验结果及重画曲线 | [实验索引](experiments.md)、[MF1 小配置学习](experiments/mf1-reference-v2/README.md)、[MF1 228M GPU 实验](experiments/mf1-gpu-mechanism-v1/README.md) |
| 参与代码开发或了解分发范围 | [架构与目录](architecture.md)、[贡献指南](../CONTRIBUTING.md)、[研究预览范围](releases/v0.1.0.md) |

## 模型结构与阅读顺序

| 模型 | 建议关注 | 入口 |
|---|---|---|
| MiniFrontier1.0 | 原生视觉输入、16 层融合顺序、四路 GR 与 MTP 数据流 | [结构图与模块解释](models/minifrontier1.md) |
| MiniQwen4 | GDN / QSA、四路 GR 与浅层 PLE | [结构图与运行入口](models/miniqwen4.md) |
| MiniKimi-K3 | KDA / MLA、AttnRes 与 LatentMoE | [结构图与运行入口](models/minikimik3.md) |
| MiniDeepSeek-V4 | SWA / CSA / HCA、mHC 与 hash routing | [结构图与运行入口](models/minideepseekv4.md) |

先读数据流和模块作用，再查看容量配置与实现代码；训练结果在各页链接的实验档案中单独记录。图中的层号从 1 开始，参数量对应研究配置，不对应微型示例。

## 实验文档中的常用术语

| 术语 | 含义 |
|---|---|
| CE token | 实际参与主语言预测损失的 token 位置；不等于全部输入位置，也不包含单独计数的 MTP 目标 |
| microbatch / 梯度累积 | 一次前后向处理的样本数，以及完成一次优化器更新前累积的微批次数 |
| pilot / acceptance | 小预算诊断或配方试验，独立于正式主训练记账；完成诊断不代表模型具备可用能力 |
| checkpoint / 精确恢复 | 包含权重及训练状态的检查点；恢复时还要核对代码、数据、词表和配置 |
| 版本与校验值 | Git 提交用于定位代码，文件哈希用于核对内容；详细标识放在报告或复现说明中 |
| 历史快照 | 截至所标日期的记录；“运行中”描述采集时的状态，不是实时监控 |

## 目录约定

| 目录 | 保存内容 |
|---|---|
| `guides/` | 可在其他机器执行的安装、数据、训练、推理操作指南 |
| `models/` | 四条模型的结构、来源及能力状态 |
| `training-strategies/` | 按日期保存的研究方案；描述目标设计和预算，阅读方式见[方案索引](training-strategies/README.md) |
| `experiments/` | 可分享的小型配置、指标、CSV、图表和解释；不包含权重和原始语料 |
| `audits/` | 数值检查和历史运行快照；时间与脱敏约定见[记录说明](audits/README.md) |
| `operations/` | 实验登记、资源调度、存储与监控方法；历史机器安排明确标注 |
| `releases/` | 软件分发支持范围、发布准备与验收记录 |
| `legacy/` | 历史设计和失败对照；不作为当前操作入口 |

旧的根层指南链接保留短入口，实际内容集中在 `guides/`。[2026-09-08 项目审查](project-review.md)与[训练失败复盘](training-failure-v1.md)保留其历史背景。

当前项目处于研究预览阶段。模型页说明已实现的功能，实验报告说明实际训练结果；具体训练效果请结合报告日期、配置、数据和检查点阅读。
