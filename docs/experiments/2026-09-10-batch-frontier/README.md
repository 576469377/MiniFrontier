# Microbatch 边界与固定全局窗口对照（2026-09-10）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**固定样本窗口后，较大的 microbatch 带来约 1.33–1.92 倍吞吐，14 组执行矩阵全部完成。** 最初的真实数据短测同时改变了实际全局 batch，修正后的比较组才具有一致的样本与 token/media 账本。首版后来据此选择[工作参数](../2026-09-10-pretraining-cutover/working-recipes.md)。

本页为历史记录。MF1 当时约 66 CE/s 的实现随后完成[性能修复](../../audits/minifrontier1-execution-performance.md)；当前安排见[预训练计划](../../pretraining-plan.md)。首批与复核采集时间分别保存在[首批索引](index.json)、[复核台账](reviewed-registry.json)。

阅读重点：[固定窗口 Kimi 对照](#固定全局-batch-后的-kimi-复验) · [14 组矩阵结果](#执行矩阵结果) · [失败、更正与停止决定](#处理决定)。

## 实测与解释

这里的 **microbatch** 是一次前后向的样本上限，**input** 是非 padding 输入量，**CE** 是参与语言损失的监督 token。`reserved` 表示 PyTorch 分配器保留的显存；表中速度为指定更新窗口内的 CE/s。

**初始真实数据短测：实际全局窗口尚未固定。** 下表用于观察吞吐和显存边界，受控加速比见后两组表。

| 真实 1M CE 短测 | microbatch | CE/s | 平均 input/update | peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| 远端 Kimi | 16 | 3,223 | 18,717 | 5.41 |
| 远端 Kimi | 32 | 4,446 | 23,767 | 7.37 |
| 远端 Kimi | 128 | 8,053 | 48,158 | 19.18 |
| 本机 Kimi | 128 | 8,295 | 48,158 | 19.80 |
| 远端 DeepSeek | 16 | 2,995 | 18,219 | 11.54 |
| 远端 DeepSeek | 32 | 4,517 | 22,955 | 19.37 |
| 本机 DeepSeek | 32 | 4,662 | 22,955 | 19.37 |
| 远端 Qwen | 16 | 1,079 | 18,218 | 9.86 |
| 远端 Qwen | 32 | 1,489 | 22,913 | 12.69 |

上表性能窗口为 **3 次预热＋8 次更新**，不含启动、完整验证和存盘。另有合成边界扫描固定 65,536 input positions/update，与更早的 16,384 预算不同。21 GiB allocator 上限下，DeepSeek 64、Qwen 128 触及边界，Kimi 128 可运行。

Kimi 约 1M CE 短训中，microbatch 16/32/128 分别执行 **54/42/21 次更新**，终点 NLL 为 **8.0135/8.2245/9.0761**。较大微批改变了跨越 input 目标的位置，连带改变更新次数；LR 也未单独调优。这组结果只能用于发现执行候选。

MF1 完整 228M、长度 512 的 20＋100 测量为 **66.32 CE/s、6.72 GiB reserved**，batch8/16 短筛仍约 68 CE/s，说明瓶颈主要在当时的 reference 调度。

### 固定全局 batch 后的 Kimi 复验

四组均完成 **16 updates、1,045,978 CE、1,052,262 input、50 次图像曝光**，逐更新样本位置和全部账本一致。microbatch 128 使用本机 GPU 4，16 使用 GPU 5，未交叉换卡；性能取 3＋8 窗口，验证固定 32 例。

| 实现 / microbatch 上限 | 固定 input 目标 | CE/s | peak reserved GiB | 终点 NLL |
| --- | ---: | ---: | ---: | ---: |
| 首次有界取样 / 16 | 65,536 | 2,855 | 5.52 | 9.47127 |
| 首次有界取样 / 128 | 65,536 | 4,348 | 19.17 | 9.48141 |
| 完整窗口合批 / 16 | 65,536 | 3,523 | 5.29 | 9.47779 |
| 完整窗口合批 / 128 | 65,536 | 6,772 | 19.17 | 9.47726 |

`4fada3e` 固定整个更新的取样边界；`dd2b519` 再将完整窗口按 microbatch 上限合批，减少尾部小批。后一实现中，128 相对 16 吞吐约 **1.92 倍**，短测 NLL 接近。旧约 8K CE/s 使用不同全局窗口，不能纳入这个加速比。

逐更新记录及权重 hash 见[受控对照](controlled-batch-comparison.json)。四项临时权重在完成核对后清理，日志与校验值保留。

### 后续执行矩阵的输入修正

[执行计划](execution-matrix-plan.json)中的首批六组 Kimi 错用了 Qwen 文本比例 **40/25/20/10/5**。复核后停止未完成项，将已完成结果列入 `excluded_initial_trials`；六组以 Kimi 原比例 **45/25/15/10/5**、相同 seed 和重新初始化的权重重跑。八组 Qwen/DeepSeek 的比例正确。

### 执行矩阵结果

有效 **14/14 组完成**：Kimi 6、Qwen 4、DeepSeek 4。六个比较组的全部更新窗口及 input/CE/media 一致；Kimi 64K 新增上限也与此前 16/128 的 16 个窗口一致。详见[完整结果](execution-matrix-results.json)。下表的箭头对应同一模型、同一全局 input 目标下的两档 microbatch；只在该行内计算倍率。

| 模型 / 全局 input 目标 | microbatch 上限变化 | CE/s | 倍率 | 终点验证 NLL |
| --- | --- | --- | ---: | --- |
| Kimi / 16K | 16 → 64 | 3,058 → 5,798 | 1.90× | 7.9181 → 7.9201 |
| DeepSeek / 16K | 16 → 32 | 2,814 → 3,749 | 1.33× | 8.0956 → 8.0973 |
| DeepSeek / 64K | 16 → 32 | 3,218 → 4,709 | 1.46× | 9.8604 → 9.8589 |
| Qwen / 16K | 16 → 64 | 945 → 1,745 | 1.85× | 8.0647 → 8.0408 |
| Qwen / 64K | 16 → 64 | 1,372 → 2,630 | 1.92× | 10.2988 → 10.2977 |

Kimi 固定 16K 时，61 次更新每次最多 53 个样本，microbatch 64 与 128 因而产生相同实际微批和终点 NLL。固定 64K 的 64 档为 **7,054 CE/s、10.57 GiB reserved**；此前 128 档为 **6,772 CE/s、19.17 GiB**。两者物理卡与时段不同，约 4% 的速度差不足以判断快慢，64 档的显存占用更低。

同窗口较大微批没有出现明显的终点 NLL 偏移，支持用合批减少开销；本轮没有选出普遍最优的全局 batch。

## 处理决定

- 停止七项旧 microbatch 2 运行。远端六项尚未到保存间隔，观测 CE 高于可恢复计数，未生成新检查点。
- 三项本机 batch16 与四项已启动的远端 Kimi128/DeepSeek32 长试验保留为旧配置探索；远端后续派发和待启动 Qwen 长试验暂停。
- 两项 MF1 100K 指令扩展约 2 CE/s，停止为 `stopped_for_review`，已保存恢复点与未保存进度分别登记。
- 新实现分离 microbatch 上限与全局 input 目标，增加按 CE 节点的 batch ramp，保留完整样本和媒体。
- 取消按“最快 5% 内最小 batch”直接晋级长期训练的规则；测速选择仅用于最多 1M CE 的执行确认。
- 修正 DeepSeek LR 解释：该优化器的 Muon/AdamW 共用 `--lr`，旧 `--muon-lr 0.01` 未生效；reference 实际为 **3e-4**，低 LR 对照为 **1e-4**。

## 复核范围

台账覆盖本机及已同步远端的 **215 条记录、13 类用途**，包含父扫描、训练阶段和安装 fixture。没有未分类项或检测到的活跃输出/GPU 分配冲突；31 条旧记录缺源码身份，4 条缺数据/tokenizer 身份，1 条历史 RL 接口试验缺结束证据，保留 `unverified`。

逐项状态、预算、用途和缺项见[复核 CSV](reviewed-registry.csv)，原始绑定在 JSON 中。记录条数包含父子阶段，不能作为独立实验数量。

## 档案结构

| 文件 | 内容 |
| --- | --- |
| `index.json`、原 `registry-snapshot.json` | 首批快照时刻及当时台账，保留原状态 |
| `reviewed-registry.json` / `.csv`、`review-summary.json` | 本轮复核后的全部登记记录、覆盖类别、缺项和统计 |
| `batch-comparison.json` | 本轮真实短测逐配置汇总；完整原始更新、验证记录位于对应 run 档案 |
| `controlled-batch-comparison.json`、`controlled-*-queue*.json` | 修正全局 batch 和窗口合批后的四项完成结果与冻结执行计划 |
| `execution-matrix-plan.json` | 随后派发的 14 组固定全局 batch 短测命令、数据身份和采集时状态；不改写此前 215 条复核快照 |
| `execution-matrix-results.json` | 执行矩阵完成结果、可重画曲线、逐步窗口核对和完整作业耗时 |
| `stopped-batch2.json`、`mf1-stopped-review.json` | 主动停止原因、已观察进度与已保存进度 |
| `*-plan.json`、`*-state.json` | 原队列命令、来源和依赖；某些策略已经被本次复核替代，不能当作推荐启动计划 |
| `remote-239--*.json`、`local--*.json` | 各扫描/真实试验的配置、指标、预算、性能行及临时权重 hash 清单 |
| `mf1-steady512-complete.json` | 已完成 MF1 持续测量 |
| `*-runner.py.txt` | 当时冻结 worker 原文；仅作历史证据，新操作用源码树的共用入口 |

本目录不含数据、tokenizer 或权重。公开副本以 `${WORKSPACE}` 替换本机根目录，并去除设备唯一标识；原运行文件的 hash 与路径归一后的公开副本可能不同。复现时按原始源码、数据、tokenizer 和运行配置绑定输入。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [前一轮 batch 测速](../2026-09-10-batch-tuning/README.md)
