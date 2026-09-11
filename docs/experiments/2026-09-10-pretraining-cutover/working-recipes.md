# 首版 base 工作参数与阶段继承

依据[主计划](../../pretraining-plan.md)及现有试验选择一套工作参数。机器记录在 `configs/experiments.json` 的 `execution_program.models.*.training_recipe`，状态为 `selected_pending_data_and_resource_freeze`。这意味着优化器、全局输入目标与主日程已经选定；数据/tokenizer、真实媒体微批和生产资格尚未完成，不能据此派发正式训练。

## 已选工作参数

所有模型使用 seed=42、BF16 计算、FP32 优化状态与梯度裁剪 1.0。首阶段从随机初始化开始；旧试验及性能验收权重不作为初始化。下表的 microbatch 是待真实首阶段最大长度/媒体配置确认的上限候选，不是已经通过的容量。

| 模型 | 全局非 padding input 目标 | microbatch 候选 | 峰值 LR | 主日程 |
|---|---:|---:|---|---|
| Kimi | 32,768 | 64 | Muon 0.005，Adam 3e-4；视觉 1e-4，projector 3e-4 | 累计主 CE：前 20M warmup，随后 cosine，2B 时降至 0.1 peak |
| Qwen | 16,384 | 32 | Q1/Q2 Muon 0.01、Adam 3e-4；视觉 1e-4，projector 3e-4 | Q1/Q2 共 1.5B，前 15M warmup，其后保持 peak；Q4/Q5 另见下文 |
| DeepSeek | 32,768 | 32 | 主干 Muon/Adam 共用 3e-4 | 前 25M 主 CE warmup，保持到累计 2B，最后 500M cosine 至 0.1 peak；固定 batch |
| MF1 | 16,384 | 8 | 主干 3e-4，视觉/标量 1e-4 | 沿用前 2M warmup、保持至 2.4B、最后 600M cosine 至 0.1 peak |

前三个来源模型保留 WD=0.1，标量及各自规定的特殊参数继续使用原分组，Adam eps=1e-8、betas=(0.9,0.95)。DeepSeek 的 eps 是本项目的 FP32 稳定性适配，不声称复刻官方全部数值。MF1 保留矩阵/视觉 WD=0.1、embedding/head/lookup WD=0.01、标量 WD=0，以及原参数分组。

## 选择依据与限制

[轻量证据](working-recipe-evidence.json)保存六项已结束 20M CE 对照的来源、数据/tokenizer 身份、完整验证曲线及原始记录校验值。运行命令记录的 init 均为空。比较限于相同旧数据、同一 seed 和当时实现；这些语料与验证不等于正式准入语料。

| 对照 | 最终固定验证 NLL | 决定 |
|---|---|---|
| Kimi Muon 0.005 / 0.01，Adam 同为 3e-4 | 5.0802 / 5.2561 | 首版采用 0.005 |
| DeepSeek 共享 LR 3e-4 / 1e-4 | 5.3335 / 6.0566 | 首版采用 3e-4；不再把无效的 `--muon-lr 0.01` 当实际 LR |
| Qwen Muon / AdamW，Adam 分支均为 3e-4 | 5.0179 / 5.6559 | 采用已有 Muon 0.01 路径；没有证据宣称其优于尚未完成的所有其他 LR |

这些旧对照采用 WSD 与 400K warmup。Kimi 正式的 cosine 与整条 2B 的 1% warmup 来自主计划/专项方案，不能把旧曲线当新日程的实测结果。

[固定窗口执行矩阵](../2026-09-10-batch-frontier/README.md)已证明扩大微批可减少执行开销；它没有选出最优全局 batch。Kimi/DeepSeek 的 32K 是原候选中的中间工作档：减少小窗口的更新开销，同时保留比 64K 更多的更新。没有把 16K/64K 的结果插值成一次不存在的 32K 质量试验。Qwen 保留 16K，控制较大模型和随机视觉的首阶段容量；MF1 沿用既定 16K。DeepSeek 首版暂用固定 batch，官方增长设计保留为后续研究，未伪造正式 ramp 节点。

不恢复 batch×LR 网格。固定正式数据后，每模型完成一次规定的 50+200 验收，具体失败只允许已登记的一次修复复验。1024 最大单样本、单 rank 的输入上界分别为 Kimi/DeepSeek 8,447,750，Qwen/MF1 4,351,750；实际 CE、媒体曝光另记。MF1 仍需独立共享生产代码的性能入口，不能绕过既有 2M acceptance 上限。

## Qwen 与 DeepSeek 的索引器和后续阶段

- Q3：20M input，indexer AdamW peak 3e-4，前 400K input warmup，随后 cosine 至 0.1 peak。主干、视觉与 MTP 冻结。
- Q4/Q5：主干 Muon 0.004、Adam 1.2e-4，即 Q2 peak 的 0.4 倍；视觉 4e-5、projector 1.2e-4。前 5M 新增主 CE warmup，保持至累计 2.5B，Q5 的最后 500M cosine 至 0.1 peak。indexer peak 单列为 1e-4，使用此阶段的 warmup/cooldown 因子。
- D3：10M input，indexer AdamW peak 3e-4，前 200K input warmup，随后 cosine 至 0.1 peak。D4/D5 的 indexer peak 为 1e-4，主干继续原累计主 CE 日程。
- D5：MTP 系数显式由 0.3 改为 0.1，权重形状和主优化状态保留，不能作为普通精确恢复。

索引器 AdamW 使用现有混合优化器的 Adam 分支；Qwen 无需改变其参数算法。DeepSeek 仅在显式 pretraining program 中将 indexer 归入 Adam 分支，旧命令的 Muon 分组保持可复现。主干状态在索引器阶段暂停，未凭阶段名称将输入位置计为语言训练 CE。

## 显式阶段状态

来源模型入口新增 `--pretraining-program configs/experiments.json --pretraining-phase <phase> --schedule program`。它依旧受原 `--run-kind strategy` 准入约束；工作参数状态不能通过正式入口。普通 `--init` 的既有行为保留，只有显式 program 才执行下列状态继承。

| 状态 | 同阶段恢复 | 后续 program 阶段 |
|---|---|---|
| 权重/config/tokenizer | 完整身份匹配 | 绑定真实前一阶段；只允许声明的配置迁移 |
| 优化器 | 完整恢复 | 按参数名和算法契约继承；冻结主干状态留在 CPU，重新激活时取回 |
| router / QK clip / RNG | 完整恢复 | 继承；重建模型产生的随机数不推进训练 RNG |
| 主 CE / 当前阶段账本 | 原账本 | 继承累计主 CE，当前阶段从零计数；索引器只推进 input |
| sampler | 原游标与 RNG | 数据、长度、混合与游标形状相同才继承；变化时重建并记录理由 |

检查点只额外保存当前优化器未持有的冻结状态，避免重复序列化一整份活跃优化器。`pretraining-transition.json` 和 checkpoint 的 `pretraining_state` 保存继承表、父权重 hash、阶段账本及谱系。缺失完整 program 状态、前阶段未完成、跨执行种类、变更总配方或试图从旧试验启动首阶段均拒绝。

CPU 回归比较 Kimi 的连续训练与两个阶段的模型、优化器、路由、QK clip、RNG 和数据位置；Qwen/DeepSeek 比较索引器前后的冻结主干及 moments、暂停/恢复，并验证稀疏阶段取回主状态。DeepSeek 另覆盖连续两个 sparse 阶段中 MTP 0.3→0.1 的迁移。测试使用小配置验证机制，不代表真实数据、稀疏转换质量或生产性能已经通过。

program 与既有策略准入的定向回归共 8 项通过；合并 MF1 位置修复后的完整 CPU 回归为 379 passed / 1 skipped，Ruff、格式与项目 CI 的 174 源文件 mypy 检查通过。

尚未完成的最终绑定：正式数据/媒体与 tokenizer、按阶段的长度消费与实际微批、固定评估、资源资格，以及绑定实际父权重的阶段质量门槛。四模型正式主 CE 当前均为零。


## 分阶段绑定数据

来源模型的 program 检查点保存完整训练日程，以及截至该检查点阶段的全部数据、配置、tokenizer 和 microbatch 绑定。后续阶段补入或调整自己的数据绑定，不改变已有阶段的身份；当前及已训练阶段的绑定、完整 LR 日程、优化器或预算发生变化，恢复仍会拒绝。进入下一阶段时先核对父阶段身份，再独立检查下一阶段的数据准入与质量证据。这个修复不放宽完整 `run_spec`、源码或状态继承检查，也不自动升级旧格式检查点。

正式训练固定源码 checkout。执行配方从配置复制到 `outputs/` 的运行快照，通过 `--pretraining-program` 指定；新增后续阶段绑定写入新的执行快照，并保留原快照。运行期间不要修改冻结 checkout 中的 `configs/`，因为它也是源码身份的一部分。单独补数据不需要改变训练源码或重新跑前一阶段。

三条来源模型均已用真实 CPU 保存/恢复与阶段切换验证：补入后续数据后，恢复结果与连续训练的模型参数、优化器、RNG 和 token 账本一致；已训练阶段的数据改动会被拒绝。来源模型共 11 项 program 测试通过，属于工程回归，不计正式训练 token。


## DeepSeek 既有 MTP 对照的复用

已补入[轻量证据](working-recipe-evidence.json)中的 `deepseek_mtp_comparison`：MTP 0、0.1、0.3 三组均完成 20,013,084 CE、1,195 次更新，完整验证同为 732 条、291,035 CE。原始 run、状态、曲线与保留模型的哈希均已核对；除 MTP 系数、输出/配置文件路径和开始时间外，记录的训练设置一致。

| MTP 系数 | 最终主语言 NLL |
|---|---:|
| 0 | 5.334227 |
| 0.1 | 5.330807 |
| 0.3 | 5.333452 |

比较使用主语言损失，不能拿包含不同权重 MTP 项的总 loss 排名。0.1 与 0.3 相差约 0.00265；单 seed、旧数据的一次结果不足以支持改动已经选定的首版配方。保留 D1–D4 的 0.3 与 D5 的 0.1，没有重跑训练或新开系数网格。这些旧对照使用 16K input、512 长度、400K warmup 和旧 tokenizer，只作为配方依据，不替代新数据质量、当前执行配置与正式模型能力验收。
