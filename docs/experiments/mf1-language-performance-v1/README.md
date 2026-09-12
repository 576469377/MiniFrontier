# MF1 语言诊断与性能对照（2026-09-10）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**两组 500K CE 实验能够利用简单色块媒体，但基础文字回答仍重复、错误且缺少 EOS。**

本轮据此启动 30K CE 问答诊断，并检查低吞吐的执行原因。后续进度见[07:51 更新](../2026-09-10-mf1-update/README.md)，100K 扩展的停止原因见[实验复核](../2026-09-10-batch-frontier/README.md#处理决定)，之后修复见[性能报告](../../audits/minifrontier1-execution-performance.md)。

本页包含三项不同的检查，均为 2026-09-10 记录：

| 检查 | 结果与状态 | 详情 |
|---|---|---|
| 500K CE 学习对照 | 两组完成；文字回答失败，色块媒体有学习信号 | [完成结果](#已完成的-500k-ce-实验) |
| 30K CE 语言诊断 | 从较低 LR 的完成检查点开始，采集时仍在训练 | [条件与进度](#语言学习实验快照时仍在运行) |
| lookup 与执行性能 | 数值检查完成，部分短筛完成；持续性能待测 | [数值](#缓存与-lookup-检查)、[性能](#性能基线短测与优化版早期结果) |

## 已完成的 500K CE 实验

两组各 **477 updates、500,853 CE**，完整 **228,235,809 参数**、seed 42。配置和产物身份见[完成记录](completed-500k.json)，曲线见[LR1.5e-4](adamw-lr1.5e-4.completed.metrics.jsonl)、[LR3e-4](adamw-lr3e-4.completed.metrics.jsonl)。下表的视觉正确率只判断首个生成颜色 token；LM NLL 使用当时的验证前缀，详细范围见表后说明。

| 项目 | LR 1.5e-4 | LR 3e-4 |
|---|---:|---:|
| 原验证前缀 LM NLL | 6.3779 | 6.4048 |
| 留出图片：首个生成颜色 token 正确 | 16/16 | 16/16 |
| 留出视频：首个生成颜色 token 正确 | 16/16 | 14/16 |
| 图片置黑后正确 | 4/16 | 5/16 |
| 视频置黑后正确 | 5/16 | 8/16 |
| 自我介绍、7+5 的生成抽查 | 重复、错误、未输出 EOS | 重复、错误、未输出 EOS |

视觉报告：[1.5e-4](adamw-lr1.5e-4.visual.json)、[3e-4](adamw-lr3e-4.visual.json)。低 LR 组换入不同答案的媒体后，32 个输出全部跟随替换后的颜色；这说明模型使用了色块输入，任务尚不足以评价通用视觉能力。

旧 NLL 只覆盖验证前 12 条：4 英文、2 中文、2 数学、2 图片、2 视频，图片/视频各仅一种答案。本轮因此改为完整验证，监督分母与训练累计 CE 分开；当时的 `validation/` 标签后来统一为 `eval/`。新旧诊断集不同，NLL 不能直接比较。

## 语言学习实验：快照时仍在运行

两组从已完成的 LR1.5e-4 检查点初始化，使用相同代码、32K tokenizer、数据和 seed；视觉 LR 均为 5e-6。第 2 步退出并由新进程从第 3 步恢复，见[快照](sft-running-snapshot.json)及[2e-5](adamw-lr2e-5.running.metrics.jsonl)、[1e-4](adamw-lr1e-4.running.metrics.jsonl)曲线。

| 控制项 | 设置 |
|---|---|
| 模型 | 完整 MF1 228M，视觉、MTP、lookup 保留 |
| 主干及标量 LR | 2e-5 / 1e-4，AdamW |
| 预算 | 每组 30,000 CE，按完整更新可能略超出；最多 10,000 updates |
| 每次更新输入预算 | 512 token，使用完整记录，不截断答案 |
| CE 比例 | 基础问答 90%、图片回放 7.5%、视频回放 2.5% |
| 训练集 | 项目编写的 32 条基础问答、32 张图片、8 段视频 |
| 留出集 | 106 条算术/复制任务、16 张图片、16 段视频，共 138 条 |
| 恢复与评估 | 第 2 次更新后退出并重载；每 25 次更新完整验证；训练后执行 greedy 生成、精确答案匹配及 EOS 统计 |

32 条训练问答的精确匹配衡量记忆；106 条独立文本任务用于观察泛化。初始化尚未训练稀疏索引器，因此本轮以 `--diagnostic-attention dense_pretrain` 保留原注意力路径；该设置限于 SFT 诊断。

[控制集](control-data.json)按题目划分复制、加法的全部语言与改写形式，复制原 tokenizer，复用带 hash 的生成媒体。[扩展集](instruction-data.json)含 886 条训练记录，快照时尚未训练。每组达到精确匹配≥29/32、EOS≥30/32、视觉留出≥28/32 后，才续接 100K CE；详见[条件](continuation-policy.json)。

<details>
<summary>复现说明与命令（需要已有数据和检查点）</summary>

```bash
# 已准备原机制数据，并有其 LR 1.5e-4 的完成检查点。
uv run python scripts/prepare_mf1_language_data.py \
  --source data/mf1-mechanism-public-v3 --output data/mf1-language-v1 \
  --config configs/minifrontier1/model_228m_native.json

git worktree add --detach outputs/mf1-source-language-v1 d77c51c
uv run python scripts/run_mf1_trial.py \
  --source-checkout outputs/mf1-source-language-v1 \
  --data data/mf1-language-v1/control --output outputs/mf1-language-sft-v1/adamw-lr2e-5 \
  --task language-sft --init outputs/mf1-mechanism-v1/adamw-lr1.5e-4/checkpoint.pt \
  --gpu 0 --lr 2e-5 --token-budget 30000 --input-batch-tokens 512 \
  --memory-gib 6 --reserve-gib 5
```

设备编号为示例。精确重建还需对应版本的数据、初始化权重及源码，离线首次体验见[微型示例](../../guides/minifrontier1.md)。

</details>

## 缓存与 lookup 检查

固定 14-token 输入、关闭 TF32，对比完整前向与“7-token 前缀＋逐 token 解码”，见[原始数值](cache-checks.json)。误差列比较同一输入的完整前向与增量解码；greedy 一致率只描述该样本。

| KDA 路径 | FP32 最大绝对误差 | BF16 最大绝对误差 | 该样本各位置 greedy 一致率 |
|---|---:|---:|---:|
| 自动融合内核 | 0.006961 | 0.609375 | 100% |
| PyTorch 参考递推 | 0.00000286 | 0.0625 | 100% |

参考 FP32 通过 `atol=rtol=2e-4`，融合内核未通过该阈值。两条路径在此样本 greedy 一致，但没有覆盖任意输入。

lookup 预填充由逐 token 投影/卷积改为批量投影与带重置 mask 的窗口，保留参数和检查点键。CPU 覆盖控制符、媒体、padding、样本边界、梯度与分段缓存；CUDA 完整配置的 FP32 输出/梯度最大误差约 **1.86e-9 / 2.91e-11**，BF16 输出约 **4.99e-6**，状态一致。见[模块报告](lookup-correctness.json)及[脚本](lookup-cuda-runner.txt)。

实现前全量 CPU 为 **302 passed、1 skipped**；后续 53 项定向回归覆盖共享显存和 lookup autocast 修正。Ruff、格式、mypy 通过；41 项既有 CUDA 未纳入 CPU 命令，CUDA 结果来自上述专项运行。

## 性能：基线短测与优化版早期结果

单张独占 3090、完整配置、文本 512、每更新 4096 input，固定随机输入与辅助目标，预热 1 次、测量 2 次。原始算子表见[profiler](baseline-profile.json)。

| microbatch | 梯度累积 | 原实现 CE/s | lookup 改造后 CE/s |
|---:|---:|---:|---:|
| 1 | 8 | 56.50 | 59.91 |
| 2 | 4 | 61.48 | 未测 |
| 4 | 2 | 63.85 | 68.26 |
| 8 | 1 | 63.92 | 运行中 |

[基线](baseline-batches.json)显示 microbatch 4→8 几乎无增益。[优化版快照](optimized-batches-running.json)当时完成 1 和 4，短测改善约 6%–7%；主机负载不同，尚无稳定加速比。

队列随后测量 512 的 microbatch 1/4/8、2048 文本和 196 特征图像，再对选中配置做 20＋100 测量。当时未覆盖 8K、视频矩阵或真实混合训练。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_mf1.py \
  --config configs/minifrontier1/model_228m_native.json --output outputs/mf1-batch-screen \
  --length 512 --input-tokens 4096 --batches 1 4 8 --warmup 1 --updates 2
# 按短筛结果设置 --batches，并改用 --warmup 20 --updates 100 进行持续测量。
```

本目录保存指标与曲线，原始数据和完整检查点保留在本地。TensorBoard 设置见[操作说明](../../operations/mf1-language-performance.md#tensorboard-分组)。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [后续语言与性能快照](../2026-09-10-mf1-update/README.md)
