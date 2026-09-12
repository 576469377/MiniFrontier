# 单卡 microbatch 测速与配方复验（2026-09-10）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**合成输入下，microbatch 2→16 的短测吞吐提高 4.53–6.33 倍。** 三个来源模型随后完成目标为 80K CE 的真实数据预检，并启动三组 batch16、20M CE 复验。本页记录测速结果和当时的启动状态。

**后续更正：** 真实数据的实际全局窗口会随 microbatch 改变，此组不属于固定全局 batch 的纯性能对照；DeepSeek 的 Muon LR 与 `--lr` 共用，旧 `--muon-lr 0.01` 未生效。修正后的实验见[固定窗口对照](../2026-09-10-batch-frontier/README.md)，最终选择见[工作配方](../2026-09-10-pretraining-cutover/working-recipes.md)。

环境为单张 RTX 3090 24 GiB、PyTorch `2.13.0+cu130`、CUDA `13.0`，每卡同时一个测试。各项使用固定代码版本。

阅读顺序：[合成短测](#固定输入的短测速) → [真实数据预检](#真实数据短训) → [20M 复验启动情况](#已启动的-20m-ce-复验)。MF1 的执行瓶颈[单独记录](#mf1-的不同瓶颈)。

## 固定输入的短测速

长度 512、seed 42、BF16、梯度检查点。每更新固定 32 条随机序列，即 **16,384 input / 16,352 CE**；microbatch 2→32 对应累积 16→1。各档重新初始化相同模型，包含前后向、裁剪和对应语义优化器。下表每行固定模型与更新输入，只改变 microbatch；最后一列是该行 16 档相对 2 档的倍率。

| 模型 | batch 2 CE/s | batch 4 CE/s | batch 8 CE/s | batch 16 CE/s | batch 32 CE/s | 2 → 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniQwen4 | 359 | 646 | 1,099 | 1,624 | 2,096 | 4.53× |
| MiniKimi-K3 | 729 | 1,484 | 2,712 | 4,441 | 6,571 | 6.09× |
| MiniDeepSeek-V4 | 641 | 1,263 | 2,318 | 4,058 | 5,802 | 6.33× |

每档仅 **1 次预热＋2 次测量**，不含真实数据、媒体、验证及训练器路由偏置更新。batch32 合成吞吐更高，但 DeepSeek reserved 已约 **19.23 GiB**，故本轮先用 16 进入真实数据路径。

原始报告：[Qwen](miniqwen4-synthetic.json)、[Kimi](minikimik3-synthetic.json)、[DeepSeek](minideepseekv4-synthetic.json)、[CSV](synthetic.csv)。CSV 内 MF1 的输入预算不同。

## 真实数据短训

三组复用带校验值的 64K 数据，Qwen/Kimi 包含真实图像，执行 MTP、路由与数据读取。目标每组 **80K CE**，视觉配额缩为 4 张次；保留原 20M 配方的 WSD 与 400K CE warmup，因此全部更新仍在预热期。

| 模型 | microbatch | 测得 CE/s | 平均秒/更新 | 峰值 reserved GiB | 实际累计 CE | 更新次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniQwen4 | 16 | 1,213 | 15.35 | 9.77 | 92,650 | 5 |
| MiniKimi-K3 | 16 | 3,473 | 5.30 | 5.41 | 92,360 | 5 |
| MiniDeepSeek-V4 | 16 | 3,271 | 5.72 | 11.54 | 92,567 | 5 |

上表列出真实数据路径的完成计数；性能取 1 次预热后的 3 次更新，包含数据与优化，不含验证和保存。训练器完成整个窗口后停止，故实际 CE 超过 80K。旧 `input_batch_tokens=16384` 是目标下限，microbatch 改变越过下限的位置；统计窗口均为 3 个微批，完整复验中也会出现 4 个。

三组梯度和验证损失有限，仅验证 8 例。短训 Muon LR 为 Qwen **0.003**、Kimi **0.005**；DeepSeek 两分支共用 **0.0003**。读取分别约 **0.055/0.060/0.008 秒/更新**；Qwen 优化器约 **4.01 秒/更新**。异机 batch2 的时段和负载不同，不能直接相除计算加速比。

逐更新报告：[Qwen](miniqwen4-mb16-real.json)、[Kimi](minikimik3-mb16-real.json)、[DeepSeek](minideepseekv4-mb16-real.json)、[汇总](real-summary.json)。对应 `*-run.json` 记录输入身份，`*-metrics.jsonl` 保存曲线。

## MF1 的不同瓶颈

[旧实现 profiler](minifrontier1-synthetic.json)包含 1 次预热、1 次剖析和 1 次普通测量。最后一次以 microbatch 1 处理 512 input，耗时 **10.66 秒、47.94 CE/s**。

剖析的累计 Self CPU 为 **12.927 秒**、Self CUDA 为 **736.210 毫秒**，包含大量小型 `bmm`、索引、`nonzero` 和复制。结合 CSA/QSA-MLA 的逐 query 循环，后续转向批量注意力与专家优化；修复见[MF1 性能报告](../../audits/minifrontier1-execution-performance.md)。算子累计时间不能直接换算 GPU 利用率。

## 已启动的 20M CE 复验

| 模型 | 候选学习率 | seed | microbatch | 预算 | 启动状态 |
| --- | --- | ---: | ---: | ---: | --- |
| MiniQwen4 | Muon 0.01 / Adam 分支 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |
| MiniKimi-K3 | Muon 0.005 / Adam 分支 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |
| MiniDeepSeek-V4 | Muon / Adam 分支共用 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |

三组保留原候选的模型、数据、tokenizer、seed、400K warmup/WSD 和 MTP，单列实际窗口差异。Qwen 20M 使用 reference LR 0.01，区别于上方短测的 0.003；性能窗口恢复为 50＋200，保留完整验证。

[启动快照](batch16-start.json)记录实际更新。它们属于 `acceptance`；原异机 batch2 使用原配置，TensorBoard 新增 `batch16/` 与 `performance-real-mb16/`。这批旧队列后来结束，结果用于首版工作参数选择。

## 复现与档案范围

通用入口为[测速脚本](../../../scripts/benchmark_training_batch.py)，当时原文保存在 [runner 快照](benchmark_training_batch.py.txt)，hash 与合成报告一致。MF1 脚本位于对应源码版本的 `scripts/benchmark_mf1.py`。

队列档案：[首批测速](strategy-performance-gpu-v1-plan.json)、[DeepSeek 测速](strategy-performance-gpu-v2-plan.json)、[Qwen/Kimi 真实短训](strategy-real-batch-gpu-v2-plan.json)、[DeepSeek 真实短训](strategy-real-batch-gpu-v3-plan.json)、[20M 复验](strategy-batch16-gpu-v1-plan.json)。复现需要各自的数据和前置产物；公开路径以 `${WORKSPACE}` 表示。

首次真实预检给旧训练器传入不支持的 `--stop-after-updates`，在参数解析时退出，更新与 CE 均为 0。随后使用支持的 CE 预算并新建输出；详见[失败记录](preflight-failures.json)。本目录仅分发配置、数值和执行记录。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [固定窗口复验与更正](../2026-09-10-batch-frontier/README.md)
