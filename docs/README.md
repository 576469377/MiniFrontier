# 文档导航

MiniFrontier 文档分为模型说明、操作指南和实验记录。首次使用从[首页 CPU 示例](../README.md#快速开始)开始，可在本地完成 MF1 的数据生成、训练、恢复和推理。

[项目首页](../README.md) · [预训练主计划](pretraining-plan.md) · [实验结果](experiments.md) · [发布范围](releases/v0.1.0.md)

## 开始实践

| 需要做什么 | 入口 |
|---|---|
| 安装并跑通融合模型的数据、训练、恢复和生成 | [MiniFrontier1.0 操作指南](guides/minifrontier1.md) |
| 了解实际使用的数据、处理方法和数据许可 | [数据来源说明](guides/data-sources.md) |
| 运行三个来源模型的离线 CPU / 3090 示例 | [来源模型最小示例](guides/quickstart.md) |
| 查看三个来源模型的训练、后训练与草稿流程 | [训练](guides/training.md)、[后训练](guides/posttraining-adaptation.md)、[草稿](guides/draft-adaptation.md) |
| 在浏览器中观察来源模型实验权重 | [实验 Demo](guides/demo-experiments.md) |
| 阅读实验结果、查看进度或重画曲线 | [实验索引](experiments.md)、[实验管理](operations/experiment-management.md) |
| 参与代码开发或了解分发范围 | [架构与目录](architecture.md)、[贡献指南](../CONTRIBUTING.md)、[研究预览范围](releases/v0.1.0.md) |

## 模型结构与阅读顺序

| 模型展示页 | 阅读重点 | 操作入口 |
|---|---|---|
| [MiniFrontier1.0](models/minifrontier1.md) | 原生视觉输入、16 层融合顺序、四路 GR 与 MTP 数据流 | [MF1 指南](guides/minifrontier1.md) |
| [MiniQwen4](models/miniqwen4.md) | GDN / QSA、四路 GR 与浅层 PLE | [来源模型示例](guides/quickstart.md) |
| [MiniKimi-K3](models/minikimik3.md) | KDA / MLA、AttnRes 与 LatentMoE | [来源模型示例](guides/quickstart.md) |
| [MiniDeepSeek-V4](models/minideepseekv4.md) | SWA / CSA / HCA、mHC 与 hash routing | [来源模型示例](guides/quickstart.md) |

模型页依次说明数据流、模块计算、配置、来源差异和验证范围。图中层号从 1 开始，参数量对应研究配置；微型示例使用更小的容量。

## 实验文档中的常用术语

| 术语 | 含义 |
|---|---|
| CE token | 实际参与主语言预测损失的 token 位置；不等于全部输入位置，也不包含单独计数的 MTP 目标 |
| microbatch / 梯度累积 | 一次前后向处理的样本数，以及完成一次优化器更新前累积的微批次数 |
| pilot / acceptance | 小预算诊断或配方试验，独立于正式主训练记账；完成诊断不代表模型具备可用能力 |
| checkpoint / 精确恢复 | 包含权重及训练状态的检查点；恢复时还要核对代码、数据、词表和配置 |
| 版本与校验值 | Git 提交定位源码，文件哈希核对内容；精确标识保存在复现记录中 |
| 历史快照 | 截至所标日期的记录；“运行中”描述采集时的状态，不是实时监控 |

## 目录约定

| 目录 | 保存内容 |
|---|---|
| [guides/](guides/README.md) | 安装、数据、训练、评估和推理步骤 |
| [models/](models) | 四个模型的结构图、模块计算、来源与能力状态 |
| [training-strategies/](training-strategies/README.md) | 按日期保存的研究方案及现行安排的差异 |
| [experiments/](experiments.md) | 实验条件、指标、曲线和结论；不包含权重与原始语料 |
| [audits/](audits/README.md) | 实现检查、数值对照和历史审计 |
| [operations/](operations/experiment-management.md) | 实验登记、资源调度、存储和监控；历史机器安排单列 |
| [releases/](releases/v0.1.0.md) | 软件分发范围与发布准备 |
| [legacy/](legacy) | 历史设计、旧接口和失败对照 |

旧的根层指南保留跳转链接。研究方案原文和上游源码快照保留固定内容；执行安排的变化写入主计划，实验更正写在对应报告内。历史设计见 [legacy/](legacy)，早期问题见[项目审查](project-review.md)与[训练失败复盘](training-failure-v1.md)。

文档修改与开发检查见[贡献指南](../CONTRIBUTING.md)；图示文件及来源见[素材索引](assets/README.md)。
