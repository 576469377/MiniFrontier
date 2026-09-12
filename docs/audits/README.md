# 检查记录

这里保存模型数值检查、短程训练验证、效果复盘和调度快照。报告以记录日期、源码版本、配置和数据为准；文件名中的 `active` 或状态字段中的 `running` 描述采集时刻，不代表 GitHub 页面上的实时状态。

## 阅读入口

- [文档核对记录](documentation-review.md)：本轮逐页范围、修正项与验证边界。
- [正式训练执行优化](training-infrastructure.md)：数据预取、Qwen 优化器、Kimi 缓存恢复及统一 TensorBoard。
- [MF1 执行性能与正确性修复](minifrontier1-execution-performance.md)。
- [MiniFrontier1.0 实现检查](minifrontier1-implementation.md)与[验证清单](minifrontier1-validation.json)。
- [三个来源模型的方案执行记录](strategy-implementation-v2.md)。
- [早期训练失败复盘](../training-failure-v1.md)。
- 可比较的曲线与实验配置见[实验档案](../experiments.md)。

## 公开副本

报告保留性能和评测数值、随机种子、源码与数据校验值。部分早期报告的本机绝对路径已在公开副本中替换为 `${WORKSPACE}`；进程编号仅在复现调度行为需要时保留。被整理的文件带有 `publication` 字段，记录原文件校验值和修改范围。本地原始记录保留在 `outputs/`。

少量合成题目和模型输出作为失败或记忆实验的证据保留，不能作为训练数据集使用。原始语料、完整训练日志与权重不放入本目录。
