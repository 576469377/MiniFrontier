# 共卡 MTP 对照实验

> 2026-09-09 的历史共卡试验，已由单卡独占调度替代。下文只解释当时的实验条件，当前安排见[预训练主计划](../pretraining-plan.md)。

2026-09-09 的 MTP 权重对照在 GPU 0–5 每卡增加一个独立实验进程，由共卡控制器接管原单卡队列。以下资源数字和目录均属于该次运行。

## 配方

每组从零训练 20M 有效 CE tokens，seed 42、64K tokenizer、seq512、microbatch 2、实际输入 batch 至少 16,384、400K token warmup/WSD。Kimi/Qwen 保留原生视觉和 1,000 次图像暴露；DeepSeek 使用 Text-v2。MTP 辅助目标单独计数。

每个新配置只改变 `mtp_loss_coef`；λ=0 时仍保留 MTP 模块，以保持初始化布局。优化器分组、数据、混合比例和 tokenizer 与各自基准一致。

<details>
<summary>历史设备分配、显存与磁盘预算</summary>

| GPU | 原任务／后续原队列任务 | 新增任务 | 新增 MTP loss 权重 |
|---|---|---|---:|
| 0 | Kimi 单卡 LR 基准 | Qwen MTP 对照 | 0 |
| 1 | Kimi 单卡较低 LR | Qwen MTP 对照 | 0.2 |
| 2 | Qwen 双卡 AdamW／单卡 LR 基准 | Kimi MTP 对照 | 0 |
| 3 | Qwen 双卡 AdamW／单卡较低 LR | Kimi MTP 对照 | 0.2 |
| 4 | DeepSeek 双卡 AdamW／单卡 LR 基准 | DeepSeek MTP 对照 | 0 |
| 5 | DeepSeek 双卡 AdamW／单卡较低 LR | DeepSeek MTP 对照 | 0.1 |

Kimi、Qwen 原基准权重为 0.1，DeepSeek 原基准为 0.3；新增项补齐原方案规定的 0/0.1/0.2 与 0/0.1/0.3 对照。它们复用现有三种容量，根据显存余量混搭模型。

新启动工作进程的 PyTorch allocator 上限分别为 Kimi 5 GiB、Qwen 11 GiB、DeepSeek 7 GiB；每个进程另预估 1 GiB 的上下文等开销。分配前要求物理空闲显存覆盖该预留和 2 GiB 公共余量，每卡最多两个计算进程。allocator 上限不是整卡显存硬隔离，驱动和第三方分配仍由物理显存监控检查。

若物理空闲显存低于 3 GiB，控制器只停止新加的伴随实验，保留原有主要任务。被停止实验保留已有检查点并记录中断状态。所有训练器仍执行 2 GiB 设备余量检查。

启动前额外核算 92 GiB 的未来产物及最大原子写入重叠空间，包含原队列未完成部分和新增六组。当次启动保留 50 GiB 磁盘余量，各次 checkpoint 写入均执行空间检查。之后的旧产物清理另见[清理记录](artifact-retention.md)。

</details>

## 进度、看板与可比较性

```bash
uv run python scripts/training_status.py --run strategy-v2
```

`outputs/strategy-shared-gpu-v2/queue-plan.json` 保存12组原有/新增任务的命令、配置和资源预算；`queue.json` 保存状态。原 `strategy-single-gpu-v2/queue.json` 同步原有六组的状态，并记录新管理器。后续 Qwen/DeepSeek 原单卡实验在各自前驱完成、显存允许后接续启动，无需等待同卡伴随实验结束。

TensorBoard 6007 的 `shared-gpu/` 前缀展示新增六组；`single-gpu/` 继续展示原单卡 LR 实验。新目录通过链接接入原看板，无需重启训练。

各 run 的 `co_residency.json` 记录每张卡的共驻留时段和伴随任务；状态与档案导出会标记 `shared_gpu_during_run`。共卡期间得到的性能曲线不能继续作为独占单卡对双卡的速度证据。比较总实验效率应对齐同一时段，将一张卡上的有效 CE 吞吐相加；不能只看显存或瞬时 GPU utilization。不同 MTP 权重的计算开销也不同。跨模型混跑还需分别记录相对各自独占基准的速度变化，不能只凭混合 token 总数判断同等计算工作是否加速。

此前 Kimi 第51–100步的独占单卡/双卡对照仍可保留为早期基准。原单卡完整性能窗口可能跨越共卡启用时刻，因此状态工具停止自动输出该窗口的“独占单卡提速倍数”。

这些 acceptance 试验的 CE 单独计量。MTP 的工作参数及选择依据见[配方记录](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)。

<details>
<summary>复现信息：训练器版本</summary>

本轮训练器的 Git 提交为 `75a3364f586936d764c31376e185747973d2bed6`。每组的实际参数、输入文件和控制器校验值保存在对应运行记录中。

</details>
