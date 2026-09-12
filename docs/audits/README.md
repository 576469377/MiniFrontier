# 检查记录

这里保存实现检查、数值对照、性能测量和失败复盘。按要核对的问题选择报告；训练曲线与实验结果集中在[实验档案](../experiments.md)。

[返回文档导航](../README.md) · [当前训练计划](../pretraining-plan.md) · [实时状态与监控](../operations/experiment-management.md)

## 阅读入口

| 要核对什么 | 报告 |
|---|---|
| 文档覆盖、修正项与验证结果 | [文档核对](documentation-review.md) |
| 数据预取、Qwen 优化器、Kimi 显存恢复与 TensorBoard | [训练执行优化](training-infrastructure.md) |
| MF1 注意力、专家、加载器和 packed 位置计算的性能变化 | [MF1 执行性能](minifrontier1-execution-performance.md) |
| MF1 首轮模块范围与 CPU 小实验 | [实现记录](minifrontier1-implementation.md)、[验证清单](minifrontier1-validation.json) |
| 来源模型 2026-09-08—09 的数值与短程训练结果 | [方案执行记录](strategy-implementation-v2.md) |
| educational-v1 的生成失败、数据偏置与验证缺陷 | [早期训练失败](../training-failure-v1.md) |

每份报告的结论只覆盖所列日期、源码、配置与输入。运行快照中的 `running` 描述采集时状态，当前进度从上方的监控入口读取。

## 公开副本

公开报告保留指标、seed、源码与数据校验值。部分本机路径替换为 `${WORKSPACE}`，相关文件的 `publication` 字段记录原文件校验值和修改范围；本地原件位于 `outputs/`。

少量合成题目和生成输出用于说明记忆或失败现象。原始语料、完整训练日志和权重不在本目录分发。
