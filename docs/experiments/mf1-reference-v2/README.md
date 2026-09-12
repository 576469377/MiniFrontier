# MF1 小配置学习实验：分组修正版（2026-09-09）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**修正数据分组后，小模型仍能记住训练题，留出算术仍为 0/6。** 本轮修复 [v1](../mf1-reference-v1/README.md) 的 val/test/demo 算术输入重叠，以相同版本的训练和评估代码重跑约 132K 参数模型。

报告时间：**2026-09-09 12:58 UTC**。代码、数据、词表和权重身份见[报告](report.json)。

## 实验条件与结果

训练集为 32 条算术、32 张色块图像、8 段色块视频，CPU 两线程。表中训练题、val 与媒体扰动各自统计，分母为相应任务的样本数。

| 实测 | 结果 |
|---|---|
| 训练预算 | 600 updates，9,404 CE，42,732 input，8,248 vision token |
| CPU 两线程耗时 | 约 242 秒 |
| 训练题生成 | 算术 32/32、图像 32/32、视频 8/8 |
| val 生成 | 算术 0/6、图像 4/4、视频 2/2 |
| val 置黑媒体 | 图像 1/4、视频 0/2 |
| val 倒序 / 重复首时间组 | 视频均为 1/2 |

结果仍表现为训练题记忆、色块媒体依赖和留出算术失败。视觉样本量很小；完整 228.24M 配置另做的 CPU 前后向/缓存检查不属于这条学习曲线。

## 曲线与检查范围

[CSV](overfit-curve.csv)可重画 loss、梯度和 token 曲线：横轴用 `step` 或累计 `ce_tokens`，纵轴用 `train_lm_loss`。这是训练集上的拟合过程，留出效果由上表的生成检查记录。

[验证记录](../../audits/minifrontier1-validation.json)另列安装、导出与 Demo API 检查；报告中的 `full_model` 与 `overfit` 分别保存完整配置工程检查和小配置学习结果。

## 同规格复现

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run minifrontier mf1 prepare-fixture \
  --output outputs/mf1-reproduce-data --seed 42
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run minifrontier mf1 train \
  --phase pilot --config configs/minifrontier1/model_tiny.json \
  --data outputs/mf1-reproduce-data --output outputs/mf1-reproduce \
  --steps 600 --input-batch-tokens 64 --lr 0.001 --vision-lr 0.001 \
  --save-every 100 --eval-every 100
uv run minifrontier mf1 evaluate --checkpoint outputs/mf1-reproduce/checkpoint.pt \
  --data outputs/mf1-reproduce-data --split val --generation \
  --output outputs/mf1-reproduce/generation.json
```

当前代码可运行同规格实验；逐位复现需要报告绑定的源码、数据、词表和原检查点。原始权重、媒体与完整日志不随源码分发。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [完整配置 GPU 实验](../mf1-gpu-mechanism-v1/README.md)
