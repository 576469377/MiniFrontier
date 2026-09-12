# MF1 小配置学习实验：独立分组修正版

本页记录 2026-09-09 的小配置实验。完整 228.24M 配置完成过 CPU 随机图像前后向与缓存对照；本页的学习曲线来自约 132K 参数小配置。两者不是同一个训练能力结果。

此次小配置使用第 2 版合成诊断数据：32 条算术、32 条色块图片、8 条色块视频，val/test/demo 的算术输入也独立分组。训练和生成评测使用相同版本的代码；相对初版的变化，以及代码、数据、词表和权重的校验值见[实验报告](report.json)。

| 实测 | 结果 |
|---|---|
| 训练预算 | 600 updates，9,404 CE，42,732 input，8,248 vision token |
| CPU 两线程耗时 | 约 242 秒 |
| 训练题生成 | 算术 32/32、图像 32/32、视频 8/8 |
| val 生成 | 算术 0/6、图像 4/4、视频 2/2 |
| val 置黑媒体 | 图像 1/4、视频 0/2 |
| val 倒序 / 重复首时间组 | 视频均为 1/2 |

这说明小配置可以记忆训练题并从色块媒体中获取信息；留出算术完全失败，样本量与视觉任务也不足以证明通用能力。未消费正式 3B 预算，不代表完整 228M 模型已经训练好，也不能用于估计 3090 的正式吞吐。

[逐步 CSV](overfit-curve.csv) 可重画 loss、梯度和 token 曲线。[工程验收记录](../../audits/minifrontier1-validation.json)单列 CPU 回归、安装包、导出和 Demo API 的验证范围。

可用当前 checkout 做同规格机制复现：

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

这些命令用于重复同规格实验，不保证与历史结果逐位一致。精确恢复需要报告所记录的代码内容、数据、词表和原检查点；仅检出一个 Git 提交不足以满足全部条件。原始权重、媒体和完整日志不随源码分发。
