# 本地产物保留规则

训练目录和冻结源码保持稳定，代码重构不迁移正在运行的产物。`outputs/` 为本地目录，不进入 Git；可公开的指标和曲线整理到 `docs/experiments/`。

- 运行中、等待后续阶段或仍需恢复的实验：保留完整 `checkpoint.pt`、优化器状态、tokenizer、数据清单和冻结源码。
- 已结束的历史实验：保留配置、日志、TensorBoard 事件、评测及可重评估的 `model.pt`。不再继续训练且已有推理权重时，可删除重复的恢复检查点。
- 清理按明确的目录和文件清单执行，核对实验完成状态、保留权重和进程引用，记录实际删除项；不通过全局 `*.pt` 通配删除。
- 删除恢复检查点后，该旧实验不再支持精确续训；保留的推理权重仍可用于复盘或新实验初始化。历史报告中的校验值和运行结果不随清理重写。

## 2026-09-09 清理

本次只清理原三个模型的 `educational-v1`、`acceptance-v1`、`acceptance-context` 和 `memorization-probe-v1` 中已完成阶段的 `checkpoint.pt`：共 31 份，释放约 85.16 GiB。

对应的全部 `model.pt`、tokenizer、配置、训练/验证日志及复盘材料保留。strategy-v2、MiniFrontier1.0 及运行中的实验未列入删除范围。清理完成后的瞬时可用空间约 231.52 GiB，之后随实验写入而变化。

逐文件记录见 [清理清单](2026-09-09-legacy-checkpoint-cleanup.json)。
