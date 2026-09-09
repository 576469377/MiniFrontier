# 文档导航

MiniFrontier1.0 是当前融合主线，MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4 保留为来源架构对照。模型系列的 “1.0” 与软件包的 `v0.1.0` 研究预览版本分开计。

## 开始实践

| 需要做什么 | 入口 |
|---|---|
| 安装并跑通融合模型的数据、训练、恢复和生成 | [MiniFrontier1.0 操作指南](guides/minifrontier1.md) |
| 理解融合架构和当前能力边界 | [模型说明](models/minifrontier1.md)、[实现与验收](audits/minifrontier1-implementation.md) |
| 跑原三个模型的离线 CPU / 3090 示例 | [来源模型最小示例](guides/quickstart.md) |
| 查看原三个模型的训练、后训练与草稿流程 | [训练](guides/training.md)、[后训练](guides/posttraining-adaptation.md)、[草稿](guides/draft-adaptation.md) |
| 在浏览器中观察原模型实验权重 | [实验 Demo](guides/demo-experiments.md) |
| 阅读实验结果及重画曲线 | [实验索引](experiments.md)、[MF1 学习探测](experiments/mf1-reference-v2/README.md) |
| 参与代码开发或了解分发范围 | [架构与目录](architecture.md)、[贡献指南](../CONTRIBUTING.md)、[研究预览范围](releases/v0.1.0.md) |

## 目录约定

| 目录 | 保存内容 |
|---|---|
| `guides/` | 可在其他机器执行的安装、数据、训练、推理操作指南 |
| `models/` | 四条模型的结构、来源及能力状态 |
| `training-strategies/` | 按日期保存的原始方案；正文 SHA 与训练绑定，修订另建版本 |
| `experiments/` | 可分享的小型配置、指标、CSV、图表和解释；不包含权重和原始语料 |
| `audits/` | 数值检查、阶段验收与历史运行快照；按对应日期和源码理解 |
| `operations/` | 当前工作站的调度、端口和目录记录 |
| `releases/` | 软件分发支持范围、发布准备与验收记录 |
| `legacy/` | 历史设计和失败对照；不作为当前操作入口 |

旧的根层指南链接保留短入口，实际内容集中在 `guides/`。[2026-09-08 项目审查](project-review.md)与[训练失败复盘](training-failure-v1.md)保留其历史背景。

正式训练数据、3090 全阶段性能及模型能力仍需验收。具体判断以模型页和绑定权重的评测为准；代码可执行、阶段结束或 loss 下降均不能单独证明模型可用。
