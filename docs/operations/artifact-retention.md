# 本地产物保留规则

训练目录和实验代码副本保持稳定，代码重构不迁移正在运行的产物。`outputs/` 为本地目录，不进入 Git；可公开的指标和曲线整理到 `docs/experiments/`。

- 运行中、等待后续阶段或仍需恢复的实验：保留完整 `checkpoint.pt`、优化器状态、tokenizer、数据清单和实验代码副本。
- 已结束的历史实验：保留配置、日志、TensorBoard 事件、评测及可重评估的 `model.pt`。不再继续训练且已有推理权重时，可删除重复的恢复检查点。
- 清理按明确的目录和文件清单执行，核对实验完成状态、保留权重和进程引用，记录实际删除项；不通过全局 `*.pt` 通配删除。
- 删除恢复检查点后，该旧实验不再支持精确续训；保留的推理权重仍可用于复盘或新实验初始化。历史报告中的校验值和运行结果不随清理重写。

## 2026-09-09 清理

本次只清理原三个模型的 `educational-v1`、`acceptance-v1`、`acceptance-context` 和 `memorization-probe-v1` 中已完成阶段的 `checkpoint.pt`：共 31 份，释放约 85.16 GiB。

对应的全部 `model.pt`、tokenizer、配置、训练/验证日志及复盘材料保留。strategy-v2、MiniFrontier1.0 及运行中的实验未列入删除范围。清理完成后的瞬时可用空间约 231.52 GiB，之后随实验写入而变化。

逐文件记录见 [清理清单](2026-09-09-legacy-checkpoint-cleanup.json)。

## 2026-09-10 清理与后续保留

另有 20 项已结束实验的优化器/RNG 检查点完成核对后回收，共 51.58 GiB；保留模型、tokenizer、最佳权重与报告。逐项结果见[回收凭据](../experiments/2026-09-10-pretraining-cutover/checkpoint-retirement.json)。该记录不改变上面 2026-09-09 清理的统计范围。

2026-09-11 另按依赖关系回收旧文本编码，约 8.23 GiB；原文、词表、manifest 与审计保留，见[编码缓存回收记录](../experiments/2026-09-10-pretraining-cutover/encoded-cache-retirement.json)。这部分是数据缓存，不计入上面的检查点回收量。

正式预训练的当前/上一恢复点、阶段推理权重和磁盘保留线统一见[主计划](../pretraining-plan.md#resources)。

## 2026-09-12 编码归档

69 份旧 OCR、自然图像和诊断编码已归档。远端独立重读校验全部 1,292 个文件后，回收本地 996 个载荷文件，实际释放 **8.14 GiB**；保留元数据、tokenizer 和恢复清单。64 个活跃编码依赖保持完整，一份原始来源尚未核实的旧文档/图表编码继续保留。

编码缓存现占 **14.79 GiB / 15 GiB**，新增编码仍需单独安排空间，不能把归档释放量全部当作预算内余量。逐项计数与校验范围见[归档结果](../experiments/2026-09-10-pretraining-cutover/encoding-archive-2026-09-12.json)。
