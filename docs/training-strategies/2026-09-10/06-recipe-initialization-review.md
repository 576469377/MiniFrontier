# 正式预训练初始化与现有配方复核

> **历史记录，已合并。** 本文保留当时的规划或源码复核证据；其中时刻、运行状态、速度和空间数字不代表实时状态。最新数据、配方、初始化和执行规则统一见 [MiniFrontier 预训练主计划](../../pretraining-plan.md)，无需将本文作为额外执行步骤。

核查时间：2026-09-10 20:49（北京时间）。核查对象：本项目 Git checkout，HEAD `433791f8cf3c82a802c887ea25a264edd3849074`，以及上次改写前保留的实验计划。工作区有其他正在进行的改动，本次只修正文档和计划记录。

## 结论及前次方案的错误

**本轮四个模型的正式首阶段均从随机初始化开始。旧诊断、旧 20M pilot、旧 batch 实验，以及这些实验派生的 SFT 权重，都不作为正式首阶段的初始化。** 配方、模型、tokenizer、数据及预算冻结后，在新目录建立正式运行，从 0 记录训练消耗。

前次方案有三处需要撤回：

1. 将“相同正式运行内部的检查点恢复”与“旧实验切换到新正式配方”写得不够明确，而且允许复用旧权重的例外与这次从零主训练目标不一致。
2. 用旧实验的 batch/LR 直接冻结正式配置。当前实验计划已经要求区分三种来源模型的日程与 batch 策略；这些候选尚未定案，不能因为要收束实验就假装已经选好。
3. 将 MF1 的旧低 LR 诊断值 `1.5e-4` 提升为正式默认。现有专项文档 §9.1 和训练器的默认主干 peak 都是 `3e-4`，`1.5e-4/3e-4/6e-4` 是候选范围，诊断结果不足以替换正式基线。

## 实际读到的四套配方

下表将“现有计划的候选”和“已确定的训练组织方式”分开。候选值不是已选正式超参数，也不是本次授权启动的实验队列。

| 模型 | 最新计划/实现 | 前次方案误用之处 | 正式开训前要确定 |
|---|---|---|---|
| MiniKimi-K3 | 当前候选：global input 16K/32K/64K；Muon LR 0.003/0.005/0.01、Adam 分支 3e-4；cosine。20M pilot 的 warmup 为 200K，即 1%。 | 把旧 WSD 实验的 Muon 0.005 与固定 16K 直接作为正式组合；新 cosine 组合未得到同条件质量确认。 | 选定一个 batch/LR 组合；明确 K1–K4 主 CE 轴与 cosine 的边界。若选择整条 2B 的 1% warmup，才是 20M；不能把这个算术结果当作现有跨阶段实现已经支持。 |
| MiniQwen4 | global input 固定 16K 或 32K，不默认 batch warmup；Muon LR 0.003/0.01 候选。20M pilot 本地 WSD、400K LR warmup。Q3 indexer 与 Q4 稀疏阶段另有优化安排。 | 直接冻结 16K/0.01；没有完整区分 pilot 和正式阶段的 warmup/冷却。 | 固定 batch 与 LR；正式 warmup、dense→indexer→sparse 转换及分组 LR。原专项方案 Q4 主干为 Q2 稳定 peak 的 0.3–0.5 倍、短 warmup，不能被统一“全程不重启 LR”覆盖。 |
| MiniDeepSeek-V4 | 候选为固定 16K、固定 32K、8K→16K→32K；20M pilot 的增长节点为已消费 CE 的 0/400K/2M；WSD、400K LR warmup；两分支实际共用 LR 1e-4/2e-4/3e-4 候选。 | 未先确定正式配方便取消 batch ramp 候选、冻结 16K/3e-4。 | 选择固定或增长方案并给出正式 token 节点；20M pilot 节点不能未经分析照搬到 2.5B。当前 `--muon-lr` 不控制 DeepSeek 的有效 LR。D5 的 MTP 0.3→0.1 也需要显式阶段配置。 |
| MiniFrontier1.0 | P0/P1/P2/P3 均配置 input batch 16,384；AdamW 主干默认 peak 3e-4、视觉 1e-4、标量 1e-4。前 2M 主 CE warmup；至累计 2.4B 保持 peak，最后 600M cosine 到 0.1 peak；indexer 独立计数与日程。 | 擅自改为 1.5e-4；没有充分引用已经实现的跨阶段主 CE 日程。 | 保留现有 AdamW/16K/3e-4 为文档与实现基线，若调整则明确新的冻结记录与理由；真实混合数据下测定 microbatch。P0 随机初始化，后续按既有阶段链继承主干状态。 |

证据路径：`docs/experiments/2026-09-10-pretraining-cutover/previous-experiments.json` 的三个 batch/LR 条目；`previous-current-plan.md.txt`；三份 `configs/strategies/*-plan.json`；2026-09-08 三份专项文档；2026-09-09 MF1 专项文档 §9；`configs/minifrontier1/p0.json` 至 `p3.json`。上次改写后的 `configs/experiments.json` 不能反过来作为用户原配方的证据。

## 代码实际允许什么

### 三个来源模型

- `minifrontier/training/train.py:475–540` 把模型配置、batch size、grad accum、global input、batch schedule、长度、seed、数据/tokenizer、LR、warmup、预算、source 与策略身份等写入 `run_spec`。
- `train.py:567–576` 要求保存的 `run_spec` 与当前完整一致，否则直接报 `resume recipe differs ...`。通过后才恢复 optimizer、step、data offset、RNG 和 token ledger；采样器及路由状态另在后续分支恢复。
- 因而当前代码中，**修改 microbatch 也不能声称严格 resume**；更改有效全局 batch、WSD/cosine、总训练预算或语料，当然也不行。将 20M 改为 2B 本身就是 recipe 变化，即使 batch 恰好相同也不会通过。
- `train.py:299–335` 的 `--init` 加载允许阶段的模型状态；新的 optimizer 随后创建，`first_step/offset/ledger` 从 0 起，只有 `--resume` 分支加载 optimizer 与计数。因此普通 `--init` 是新优化运行，不是连续恢复。
- `train.py:322–324` 在加载时检查 tokenizer hash；不允许用换过 tokenizer 的新语料直接套旧权重。
- `train.py:1074–1102` 按**当前运行**预算和 ledger 计算 LR，没有自动把 K1/K2 等几个独立 `--init` 运行拼成统一主 CE 轴。这是前次计划的实现缺口，不能只写一句“连续训练”就算完成。

### MF1

- `minifrontier/training/minifrontier1.py:333–350,377–390` 的严格 resume 比对完整运行身份，包含 batch、预算、优化器分组、数据/source/config 等；恢复 sampler、optimizer、router、ledger、RNG。
- `minifrontier1.py:369–403` 对 P0→P1→indexer→P2→P3 的显式阶段转换另有 `continuous` 分支：按参数名继承主优化状态，继承路由状态及累计 main CE，并禁止静默替换优化器；采样桶与权重一致时才继承 sampler。它与同阶段 exact resume 是两种语义。
- `minifrontier1_strategy.py:148–159` 已实现前述 2M/2.4B/3B 日程。`minifrontier1.py:314–322` 默认主干 LR 为 3e-4；indexer 与 SFT 有自己的 LR。
- P0 配置没有要求先拿 pilot 权重初始化；本轮直接从随机 P0 开始。之后的阶段学习依赖前一正式阶段，不应每一阶段都随机重来。

## batch 变化为何需要区分

有效全局 batch 改变每次更新所覆盖的样本/token、固定 token 预算内的更新次数及梯度统计；本项目还要考虑 optimizer moments、路由更新和按 token 的日程。旧实验与新正式配方混接后，不能把结果描述为新配方“从零训练”。

只改变 microbatch、并保持完整样本窗口、全局 batch 和 CE 归一化，有可能只是执行调整；这不是“数学上任何 batch 变化都必须重训”。但是本项目现有严格恢复接口把 microbatch 也绑定在身份中，本轮不绕过它。需要支持这类变更时，应单独实现并验证兼容迁移，不能删除检查字段。

预先声明的 DeepSeek batch ramp 则从第 0 步就属于同一配方。运行到既定 CE 节点后自然改变目标 batch，不需要每个节点重新随机初始化。恢复时必须使用原样的整个 schedule，并恢复原 ledger。

## 修正后的执行约束

1. **正式首阶段**：K1/Q1/D1/P0 各自新目录；`init=null`、`resume=null`，随机模型和视觉模块，优化器/采样/RNG/账本按冻结 seed 与 manifest 从零建立。旧实验只引用其报告，不导入权重或累计 CE。硬件准入运行的权重也不作为正式初始化。
2. **同一正式运行中断**：只恢复本运行自己的检查点，配置、数据、tokenizer、预算、source 和全部状态符合当前严格校验。正式 20M/50M 等观测点是这个既定大预算运行内部的存盘点，不能先启动一个 20M 预算作业，结束后临时改成 2B 并称精确续训。
3. **正式阶段转换**：绑定前一正式阶段的真实 checkpoint/hash，记录权重、优化器、路由、主 CE/阶段 CE、采样器/RNG 各自如何继承或重建。MF1 使用既有连续阶段路径；三个来源模型要按专项方案补齐阶段状态转移/日程，或明确选择并记录有依据的阶段重启方式。不得把 `--init` 的重置隐藏掉，也不得把所有模型改成 MF1 的同一日程。
4. **配方定案**：先核对原候选与已有证据，形成每模型一份可启动的完整配置；撤回“旧值已冻结”的表述。完整 batch×LR 网格和无条件补 seed 仍不自动启动，但决定生产 batch/LR/日程这件事不能拖到 base 完成之后。若剩余歧义确实需要实验，先补一张有明确问题、候选上限、token/GPU 小时预算及停止条件的记录，未登记就不派发。
5. **生产准入**：使用选定配方的真实 batch 与数据，不能为了统一表格强制全模型 16K。50 次性能预热＋200 次测量与 CE 上限需一并核算：64K input 的 250 次更新约 16.38M input，若大部分是 CE 就不可能塞进原 5M CE 上限。该模型的准入预算须重新冻结；不偷偷缩 batch 或减少测量后宣称原配方通过。性能预热与 LR warmup 是不同概念。
6. **存储**：旧权重按原候选清单保留/归档；本次不删除。新正式模型从零不代表删掉历史证据，也不代表需要同时保存每一个完整 optimizer checkpoint。

本次完成的是计划纠错与源码核查，没有启动或停止训练，没有修改训练器或模型实现，也没有把尚未完成的配方/阶段转换标为已通过。
