# 当前执行计划

六个模型版本预训练的阶段、预算、数据与后续安排统一维护在 **[MiniFrontier 预训练主计划](../pretraining-plan.md)**。

本页保留旧链接的兼容入口，不维护第二份计划。机器可读模板见 [configs/experiments.json](../../configs/experiments.json)；具体运行使用各自冻结的配置快照。

阶段衔接与数据供给见主计划的[当前任务](../pretraining-plan.md#execution)与[数据](../pretraining-plan.md#data)。新增 V4.1 / MF1.1 的结构取舍和独立训练安排见[联合方案](../training-strategies/2026-09-14/07-v41-mf11-implementation-and-training.md)。

执行优化与阶段接续见[审计记录](../audits/training-infrastructure.md)，原四模型固定权重的生成结果见[9 月 14 日检查点诊断](checkpoint-diagnostics-2026-09-14/README.md)。两类检查分别说明性能和生成问题，均不代表完整能力验收。

原四模型在 2026-09-12 的数据补充、采样比例及准备计数保留在[当日快照](2026-09-10-pretraining-cutover/next-phase-preparation-2026-09-12.json)。
