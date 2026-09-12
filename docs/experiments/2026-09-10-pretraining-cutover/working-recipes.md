# 首版预训练配方与阶段继承

2026-09-10 依据既有实验选定首版参数，四模型于 2026-09-11 进入正式首阶段。本文解释选择依据和阶段状态继承；当前进度集中维护在[主计划](../../pretraining-plan.md)。机器可读模板为 `configs/experiments.json`，实际执行以每次运行的冻结快照为准。

## 已选工作参数

四模型从随机初始化开始，使用 seed=42、BF16 计算、FP32 优化状态和梯度裁剪 1.0。下表记录正式首阶段的 microbatch 上限；实际微批由样本长度和窗口决定，后续阶段需重新核对容量。

| 模型 | 全局非 padding input 目标 | 首阶段 microbatch | 峰值 LR | 主日程 |
|---|---:|---:|---|---|
| Kimi | 32,768 | 64 | Muon 0.005，Adam 3e-4；视觉 1e-4，projector 3e-4 | 累计主 CE：前 20M warmup，随后 cosine，2B 时降至 0.1 peak |
| Qwen | 16,384 | 32 | Q1/Q2 Muon 0.01、Adam 3e-4；视觉 1e-4，projector 3e-4 | Q1/Q2 共 1.5B，前 15M warmup，其后保持 peak；Q4/Q5 另见下文 |
| DeepSeek | 32,768 | 16 | 主干 Muon/Adam 共用 3e-4 | 前 25M 主 CE warmup，保持到累计 2B，最后 500M cosine 至 0.1 peak；固定 batch |
| MF1 | 16,384 | 8 | 主干 3e-4，视觉/标量 1e-4 | 沿用前 2M warmup、保持至 2.4B、最后 600M cosine 至 0.1 peak |

三个来源模型使用 WD=0.1，标量和特殊参数按各自分组处理；Adam eps=1e-8、betas=(0.9,0.95)。DeepSeek 的 eps 是本项目的 FP32 稳定性适配。MF1 的矩阵/视觉 WD=0.1、embedding/head/lookup WD=0.01、标量 WD=0，betas=(0.9,0.95)、eps=1e-8。

## 选择依据与限制

[配方证据](working-recipe-evidence.json)保存六项 20M CE 对照的配置、数据/tokenizer 身份、曲线和校验值。各项均从随机初始化开始；比较使用同一 seed 和旧数据，与正式训练的数据分别记录。

| 对照 | 最终固定验证 NLL | 决定 |
|---|---|---|
| Kimi Muon 0.005 / 0.01，Adam 同为 3e-4 | 5.0802 / 5.2561 | 首版采用 0.005 |
| DeepSeek 共享 LR 3e-4 / 1e-4 | 5.3335 / 6.0566 | 首版采用 3e-4；不再把无效的 `--muon-lr 0.01` 当实际 LR |
| Qwen Muon / AdamW，Adam 分支均为 3e-4 | 5.0179 / 5.6559 | 采用已测的 Muon 0.01；未完成其他 LR 的质量比较 |

这些旧对照采用 WSD 与 400K warmup。Kimi 正式的 cosine 与整条 2B 的 1% warmup 来自主计划/专项方案，不能把旧曲线当新日程的实测结果。

[固定窗口测量](../2026-09-10-batch-frontier/README.md)表明，扩大微批可减少已测负载的执行开销。Kimi/DeepSeek 选用 32K 全局 input，在执行开销与更新次数之间折中，尚无同配置的独立质量对照。Qwen/MF1 保留 16K。DeepSeek 首版使用固定 batch，batch 增长另行研究。

2026-09-11 的[执行调整](direct-training-start.json)取消了 Kimi、Qwen、MF1 额外的独立 smoke、20M pilot 和 50＋200 性能检查，改在正式更新中观察；DeepSeek 完成既有检查和修复后开训。失败尝试、历史检查预算和实际配置见[启动记录](execution.md)与[运行快照](formal-starts.json)。

## Qwen 与 DeepSeek 的索引器和后续阶段

- Q3：20M input，indexer AdamW peak 3e-4，前 400K input warmup，随后 cosine 至 0.1 peak。主干、视觉与 MTP 冻结。
- Q4/Q5：主干 Muon 0.004、Adam 1.2e-4，即 Q2 peak 的 0.4 倍；视觉 4e-5、projector 1.2e-4。前 5M 新增主 CE warmup，保持至累计 2.5B，Q5 的最后 500M cosine 至 0.1 peak。indexer peak 单列为 1e-4，使用此阶段的 warmup/cooldown 因子。
- D3：10M input，indexer AdamW peak 3e-4，前 200K input warmup，随后 cosine 至 0.1 peak。D4/D5 的 indexer peak 为 1e-4，主干继续原累计主 CE 日程。
- D5：MTP 系数显式由 0.3 改为 0.1，权重形状和主优化状态保留，不能作为普通精确恢复。

索引器 AdamW 使用现有混合优化器的 Adam 分支；Qwen 无需改变其参数算法。DeepSeek 仅在显式 pretraining program 中将 indexer 归入 Adam 分支，旧命令的 Muon 分组保持可复现。主干状态在索引器阶段暂停，未凭阶段名称将输入位置计为语言训练 CE。

## 显式阶段状态

来源模型通过 `--pretraining-program <运行配方.json> --pretraining-phase <phase> --schedule program` 继承阶段状态。运行配方从 `configs/experiments.json` 模板生成，并绑定 `--run-kind strategy` 所需的数据和执行证据。普通 `--init` 新建优化器与计数。

| 状态 | 同阶段恢复 | 后续 program 阶段 |
|---|---|---|
| 权重/config/tokenizer | 完整身份匹配 | 绑定真实前一阶段；只允许声明的配置迁移 |
| 优化器 | 完整恢复 | 按参数名和算法契约继承；冻结主干状态留在 CPU，重新激活时取回 |
| router / QK clip / RNG | 完整恢复 | 继承；重建模型产生的随机数不推进训练 RNG |
| 主 CE / 当前阶段账本 | 原账本 | 继承累计主 CE，当前阶段从零计数；索引器只推进 input |
| sampler | 原游标与 RNG | 数据、长度、混合与游标形状相同才继承；变化时重建并记录理由 |

检查点只额外保存当前优化器未持有的冻结状态，避免重复序列化一整份活跃优化器。`pretraining-transition.json` 和 checkpoint 的 `pretraining_state` 保存继承表、父权重 hash、阶段账本及谱系。缺失完整 program 状态、前阶段未完成、跨执行种类、变更总配方或试图从旧试验启动首阶段均拒绝。

CPU 小配置回归覆盖连续训练与跨阶段继承、索引器冻结及取回主干状态，以及 DeepSeek MTP 0.3→0.1 的迁移。真实数据上的稀疏转换仍需阶段评估，检查记录见[启动档案](execution.md)和[训练基础设施审计](../../audits/training-infrastructure.md)。

## 分阶段绑定数据

program 检查点保存完整日程，以及当前和已训练阶段的数据、配置、tokenizer 与 microbatch 绑定。后续阶段可补充自己的数据；修改已有阶段绑定、完整 LR 日程、优化器或预算会被拒绝。转换时先核对父阶段，再检查新阶段的数据与质量证据。旧格式检查点不会自动升级。

正式训练固定源码 checkout。执行配方从配置复制到 `outputs/` 的运行快照，通过 `--pretraining-program` 指定；新增后续阶段绑定写入新的执行快照，并保留原快照。运行期间不要修改冻结 checkout 中的 `configs/`，因为它也是源码身份的一部分。单独补数据不需要改变训练源码或重新跑前一阶段。

三条来源模型的 CPU 保存、恢复和阶段切换检查表明：补充后续数据不改变连续训练的参数、优化器、RNG 和 token 账本；修改已训练阶段数据会被拒绝。

## DeepSeek 既有 MTP 对照的复用

[MTP 对照记录](working-recipe-evidence.json)中的三组均完成 20,013,084 CE、1,195 次更新，固定验证为 732 条、291,035 CE。除 MTP 系数、路径和开始时间外，记录的设置一致。

| MTP 系数 | 最终主语言 NLL |
|---|---:|
| 0 | 5.334227 |
| 0.1 | 5.330807 |
| 0.3 | 5.333452 |

表中比较主语言 NLL，总 loss 含不同权重的 MTP 项，不适合直接排名。旧对照使用单 seed、16K input、512 长度、400K warmup 和旧 tokenizer；0.1 与 0.3 的差值约 0.00265，不足以支持调整首版配方。因此保留 D1–D4 的 0.3 和 D5 的 0.1。
