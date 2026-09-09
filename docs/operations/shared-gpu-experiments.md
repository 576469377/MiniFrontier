# 共卡 MTP 对照实验

2026-09-09，按用户追加六组实验的要求，在 GPU 0–5 每卡增加一个独立进程。原有训练继续执行，旧单卡队列控制器由共卡控制器接管；已有 Kimi 工作进程没有重启。GPU 6、7 的任务不在本次调度范围内。

## 分配与配方

| GPU | 原任务／后续原队列任务 | 新增任务 | 新增 MTP loss 权重 |
|---|---|---|---:|
| 0 | Kimi 单卡 LR 基准 | Qwen MTP 对照 | 0 |
| 1 | Kimi 单卡较低 LR | Qwen MTP 对照 | 0.2 |
| 2 | Qwen 双卡 AdamW／单卡 LR 基准 | Kimi MTP 对照 | 0 |
| 3 | Qwen 双卡 AdamW／单卡较低 LR | Kimi MTP 对照 | 0.2 |
| 4 | DeepSeek 双卡 AdamW／单卡 LR 基准 | DeepSeek MTP 对照 | 0 |
| 5 | DeepSeek 双卡 AdamW／单卡较低 LR | DeepSeek MTP 对照 | 0.1 |

Kimi、Qwen 原基准权重为 0.1，DeepSeek 原基准为 0.3；新增项补齐原方案规定的 0/0.1/0.2 与 0/0.1/0.3 对照。它们复用现有三种容量，根据显存余量混搭模型。

每组仍从零训练 20M 有效 CE tokens，seed42、64K tokenizer、seq512、microbatch2、实际输入 batch 至少16,384、400K token warmup/WSD。Kimi/Qwen 保留原生视觉和 1,000 次图像暴露；DeepSeek 使用 Text-v2。MTP 不额外计入独立 CE 预算。

训练器继续使用冻结的 `75a3364f586936d764c31376e185747973d2bed6`。每个新配置只改变 `mtp_loss_coef`；即使 λ=0 也保留 MTP 模块，以保持初始化布局。各自基准 Muon LR、AdamW 分组 LR、数据、混合比例和 tokenizer 均保持一致。新控制器和内存上限包装器独立冻结并记录 hash，不修改原训练源码或原始策略正文。

## 显存与磁盘

新启动工作进程的 PyTorch allocator 上限分别为 Kimi 5 GiB、Qwen 11 GiB、DeepSeek 7 GiB；每个进程另预估 1 GiB 的上下文等开销。分配前要求物理空闲显存覆盖该预留和 2 GiB 公共余量，每卡最多两个计算进程。allocator 上限不是整卡显存硬隔离，驱动和第三方分配仍由物理显存监控检查。

若物理空闲显存低于 3 GiB，控制器只停止新加的伴随实验，保留原有主要任务。被停止实验保留已有检查点并记录状态，不自动覆盖或冒充预算完成。所有训练器仍执行 2 GiB 设备余量检查。

启动前额外核算 92 GiB 的未来产物及最大原子写入重叠空间，包含原队列未完成部分和新增六组。磁盘继续保留 50 GiB；不复制语料、不删除历史权重，各次 checkpoint 写入仍受原子写入和空闲空间检查保护。

## 进度、看板与可比较性

```bash
uv run python scripts/training_status.py
```

`outputs/strategy-shared-gpu-v2/queue-plan.json` 保存12组原有/新增任务的命令、配置和资源预算；`queue.json` 保存状态。原 `strategy-single-gpu-v2/queue.json` 同步原有六组的状态，并记录新管理器。后续 Qwen/DeepSeek 原单卡实验在各自前驱完成、显存允许后接续启动，无需等待同卡伴随实验结束。

TensorBoard 6007 的 `shared-gpu/` 前缀展示新增六组；`single-gpu/` 继续展示原单卡 LR 实验。新目录通过链接接入原看板，无需重启训练。

各 run 的 `co_residency.json` 记录每张卡的共驻留时段和伴随任务；状态与档案导出会标记 `shared_gpu_during_run`。共卡期间得到的性能曲线不能继续作为独占单卡对双卡的速度证据。比较总实验效率应对齐同一时段，将一张卡上的有效 CE 吞吐相加；不能只看显存或瞬时 GPU utilization。不同 MTP 权重的计算开销也不同。跨模型混跑还需分别记录相对各自独占基准的速度变化，不能只凭混合 token 总数判断同等计算工作是否加速。

此前 Kimi 第51–100步的独占单卡/双卡对照仍可保留为早期基准。原单卡完整性能窗口可能跨越共卡启用时刻，因此状态工具停止自动输出该窗口的“独占单卡提速倍数”。

所有新增组仍为 acceptance 配方试验，不计正式主训练预算。MTP 选择需要同数据/预算的验证 LM NLL 与稳定性结果；吞吐更高或显存用满不代表模型质量更好。
