# MF1 语言与性能实验更新（2026-09-10）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**lookup 改造后，512 长度短测吞吐提高约 6%–8%；两组语言诊断尚无最终结果。**

采集时间：**2026-09-10 07:51 UTC**。两组 30K CE 语言诊断仍在训练，512 短筛已经完成，持续测量完成 17/100 次。后续 100K 扩展启动后因约 2 CE/s 停止，见[停止记录](../2026-09-10-batch-frontier/README.md#处理决定)；之后的执行修复见[性能报告](../../audits/minifrontier1-execution-performance.md)。

## 这些实验要决定什么

语言诊断检查已有 228M 权重能否记住 32 条问答的格式、答案和 EOS；性能实验测量 lookup 改造与 microbatch 的收益。500K 完成结果、数据和重建方法见[前一档案](../mf1-language-performance-v1/README.md)，正式配置由[主计划](../../pretraining-plan.md)维护。

## 语言诊断：快照时的训练进度

两组共享完整模型、初始化检查点、32K tokenizer、seed、数据与源码，仅改变学习率。[学习记录](learning.json)保存配置、恢复点和校验值。表中训练进度取最新更新；两列 NLL 则固定为第 50 步，供同进度比较。

| 方案 | 最新更新 | 已训练 CE / 预算 | 第 50 步全验证 NLL | 第 50 步语言子集 NLL |
|---|---:|---:|---:|---:|
| AdamW LR 2e-5 | 63 | 10,717 / 30,000 | 4.4437 | 5.5153 |
| AdamW LR 1e-4 | 62 | 10,545 / 30,000 | 4.6934 | 5.8254 |

同为第 50 步、8,516 CE 时，高 LR 组训练 loss 下降更快，但留出语言 NLL 更高，说明此时主要是更快拟合训练题，不能据此选最终 LR。曲线：[2e-5](adamw-lr2e-5.metrics.jsonl)、[1e-4](adamw-lr1e-4.metrics.jsonl)；恢复点每 25 步保存。

当时扩展条件为训练问答精确匹配≥29/32、EOS≥30/32、视觉留出≥28/32，达到后继续 100K CE，见[条件快照](continuation.json)。本页采集时扩展尚未启动。

## 性能短筛已完成

完整 228M、单 GPU、文本长度 512、每更新 4096 input，预热 1 次、测量 2 次。基线见[原始测量](../mf1-language-performance-v1/baseline-batches.json)，新版见[短筛结果](batch-screen-512.json)。下表固定每次更新的 input 预算，通过调整累积次数比较 microbatch；速度的分母为有效 CE token。

| microbatch | 累积次数 | 原实现 CE/s | lookup 改造后 CE/s | 改造后峰值 reserved |
|---:|---:|---:|---:|---:|
| 1 | 8 | 56.50 | 59.91 | 4.34 GiB |
| 4 | 2 | 63.85 | 68.26 | 5.58 GiB |
| 8 | 1 | 63.92 | 68.81 | 6.70 GiB |

短测提升约 6%–8%，microbatch 4→8 增益很小；主机并行负载与短窗口波动未受完全控制。队列选择 8 继续 20＋100 测量，快照时完成全部预热与 17 次测量：[逐次记录](steady-partial-rows.json)、[配置](steady-512.json)。

**扩展输入短筛：** 以下两项改变了输入长度或视觉占位，分别记录速度与显存。

| 输入场景 | 每次输入预算 / microbatch | CE/s | 峰值 reserved |
|---|---|---:|---:|
| 2048-token 文本 | 2048 / 1 | 64.83 | 8.03 GiB |
| 512-token 输入，196 个图像特征位置 | 512 / 1 | 36.26 | 4.27 GiB |

[2048 文本](length-2048.json)与[图像](image-196.json)也仅完成短筛。视觉位置屏蔽 CE，因此不同模态吞吐需同时看输入量与监督比例；当时未测 8K、视频矩阵和真实混合训练。

TensorBoard 使用 `train / eval / perf`，配置见[操作说明](../../operations/mf1-language-performance.md#tensorboard-分组)。本页数值保持采集时状态。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [语言诊断完整条件](../mf1-language-performance-v1/README.md)
