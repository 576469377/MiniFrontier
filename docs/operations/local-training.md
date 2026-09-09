# 本机 strategy-v2 调度记录

本页记录 2026-09-09 的工作站实验安排，供研究复盘参考。目录、端口和设备编号对应当时的环境，实时状态以本机日志为准。通用操作见[训练指南](../guides/training.md)。

三个来源模型按 [2026-09-08 三份方案](../training-strategies/2026-09-08/) 执行；随后增加的融合模型实验见 [MF1 共卡记录](mf1-mechanism-experiments.md)。
旧 educational-v1 的训练量、数据和生成效果未达到目标，
[旧接口说明](../legacy/training-educational-v1.md) 保留作历史对照，不能作为本轮配方。

## 环境与实时状态

```bash
uv sync --locked --extra dev --extra training --extra monitoring --extra data
uv run minifrontier doctor
uv run python scripts/training_status.py
```

最初双卡分配为 Kimi GPU 0–1、Qwen 2–3、DeepSeek 4–5。
按 2026-09-09 用户调整，后续每个 run 使用一张卡，首先安排 GPU 0–5 并发六组；
6、7 上的既有任务不动。正式计划的默认 GPU 为 Kimi 0、Qwen 2、DeepSeek 4，
对应 `experiment_gpu_ids` 提供每个模型的两张独立试验卡。
真实训练从 `outputs/strategy-source-pilot-v2` 冻结源码运行，开发根目录可以继续迭代。
`outputs/strategy-source-posttraining-v2` 是后训练接口的独立验证快照，尚未用于正式 RL。
新版配方曲线在 `http://127.0.0.1:6007`；6006 保留旧 educational-v1 对照。

6007 不展示 smoke / quickstart 的工程检查曲线。`mf1-quickstart-v2` 的原始日志仍保留，
但不在 `outputs/tensorboard-strategy-v2` 下建立展示链接；后续工程冒烟检查也不要接入此看板。

6007 通过 `outputs/tensorboard-strategy-v2` 的目录链接读取：
`dual-gpu/` 对应 `strategy-recipe-pilots-v2`，`single-gpu/` 对应
`strategy-single-gpu-v2`。看板每 5 秒扫描日志，新实验首次写入事件后自动出现。
只排队、尚未启动的实验没有曲线。浏览器若保留旧 run 筛选，清空后选择
`single-gpu/minikimik3/reference/tensorboard` 或 `lower-lr/tensorboard`。

重新启动看板时沿用服务记录 `outputs/services/tensorboard-strategy-v2.json`
中的命令，或在对应端口空闲时运行：

```bash
uv run tensorboard --logdir outputs/tensorboard-strategy-v2 \
  --host 127.0.0.1 --port 6007 --reload_interval 5
```

这些链接只组织看板目录，不复制日志、数据或权重。更换看板读取目录只需重启
TensorBoard，不需要重启训练。

2026-09-09 追加六组 MTP 共卡对照，6007 另有 `shared-gpu/` 前缀。
每卡最多两个实验、原队列接管方式和资源上限见[共卡实验记录](shared-gpu-experiments.md)。

同日再在 GPU 4、5 各增加一组完整 228M MF1 机制实验，这两张卡各三组并行，
另由 MF1 监督进程执行显存和磁盘保护。6007 的 `mf1-mechanism/` 展示新增两组，
`mf1-reference/` 保留此前小配置学习曲线；随机输入的显存探测不接入看板。

工作盘写入默认保留 50 GiB，单卡预留 2 GiB 显存；数据、下载缓存和 kernel 缓存
都位于 `${WORKSPACE}`。每个原子 checkpoint 的临时重叠空间也计入估算。
失败复盘及保留产物见[权重清理记录](artifact-retention.md)；当前可恢复点不会按中间权重清理。

## 数据与配置

新模型配置使用 `configs/strategies/*-v2.json`；DeepSeek 视觉接入另有
`minideepseekv4-vision-v1.json`。机器可读阶段预算在对应 `*-plan.json`，原文不修改。

| 数据目录 | 用途与限制 |
|---|---|
| `data/strategy-tokenizers-v2` | 同一训练字节上的 32K/64K；当前按方案默认冻结 64K，质量比较未完成 |
| `data/strategy-diagnostic-v2`、`strategy-diagnostic-<family>-v2` | 算术记忆和原生颜色依赖诊断；不是泛化基准 |
| `data/strategy-recipe-public-v2`、`strategy-recipe-encoded-64k-v2` | 约 45.49M train CE 的公开文本/可核验合成数学配方池 |
| `data/strategy-recipe-joint-v2`、`strategy-recipe-<family>-64k-v2` | Kimi/Qwen 配方池再加 96 张真实图像；未满足正式视觉分布 |

数据有来源/revision/许可、分组去重和 tokenizer/文件校验；来源文件前缀、少数图片、
算术模板都不能代表完整正式数据。许可、自然科学、OCR/图表/视频、独立验证规模
和唯一图像数量仍未通过正式准入。

## 当前自动执行范围

K0/Q0/D0 在方案允许的 0.5–2M CE 范围内验证可学习性。Kimi 和 DeepSeek 的
训练内算术 16/16 通过，留出 0/9；这只证明记忆能力。Qwen 首轮 14/16 未通过，
另开 500K CE 续诊断，累计 1,000,900 CE 后训练内 16/16、留出仍 0/9；
正确图 16/16、错图 0/16、遮图 8/16。优化器重置和初始权重哈希单独记录。
续诊断通过后，Qwen 已在 `miniqwen4-after-q0-extension` 启动独立配方试验。

诊断通过后，每个模型从零分别训练 Muon 20M CE 与 AdamW 20M CE，保持相同
数据、seed 和预算。当前 seq512，每 rank microbatch 1，按实际非 padding 输入
累积到全局至少 16,384；LR 3e-4，Muon 语义配方另用 0.01，400K token warmup/WSD。
每 200 次更新保存并执行完整本地验证；先 50 次预热，再记录 200 次真实更新性能。
Kimi/Qwen 配方试验只使用小型 caption 图池，另设 1,000 次图像暴露，不冒充正式混合。

每个试验目录有 `run.json`、`metrics.jsonl`、`status.json`、`checkpoint.pt`、
`best-model.pt`、完成后的 `model.pt` 和 `performance.json`。父目录 `pilot.json`
记录诊断结果及两次独立试验状态。低 NLL 只用于同一验证集选权重，不能自动晋级。

## 后续单卡六组试验

`scripts/run_single_gpu_queue.py` 等待对应模型原有 Muon/AdamW 两组均结束，
并确认目标卡没有计算进程后启动；用 GPU UUID 隔离、每卡锁防止队列重复占用。
队列位于 `outputs/strategy-single-gpu-v2`，进度一并由 `training_status.py` 展示。

| GPU | 模型 | Muon 配方的学习率变量 |
|---|---|---|
| 0 / 1 | Kimi | Muon 原始 LR 0.01 / 0.005；AdamW 分组保持 3e-4 |
| 2 / 3 | Qwen | Muon 原始 LR 0.01 / 0.003；AdamW 分组保持 3e-4 |
| 4 / 5 | DeepSeek | 主 LR 3e-4 / 1e-4，按其独立 Muon 缩放实现 |

每组从随机初始化开始，20M CE、seed42、seq512、microbatch2，按实际输入累积到
至少16,384；64K tokenizer、数据、MTP 权重、模态配额与冻结训练源码均与前一轮相同。
单卡每次取两条样本，对应原双卡每 rank 一条的全局采样顺序。
这是 Muon 学习率筛选，最终优化器仍需结合 AdamW 完整对照决定。

前50次更新预热、随后200次记录吞吐和显存。对照参考组的单卡速度 S 与原双卡速度 D：
S/D 衡量单个 run 的速度，2S/D 估算同样两张卡并行独立试验的总吞吐；
不同时间的共享主机负载会影响测量，不能只凭 GPU 利用率断言通信瓶颈。
长上下文和正式图文分布需重新测量；单卡数据吞吐记录不能替代后续阶段的显存验收。

原始方案正文和在跑冻结源码保持原样，硬件安排调整记录在机器计划中。
六组仍是试验，不自动启动正式预算或替换 demo。新队列不直接恢复双卡优化器/采样器，
从而避免将跨卡数恢复误称为精确续训。磁盘继续保留50 GiB，检查点沿用原子写入保护。

## 正式阶段顺序与恢复

```text
Kimi：诊断 → 配方比较 → 联合 PT 2B CE → SFT/QAT → 9 教师 → MOPD → 草稿 → 验收
Qwen：诊断 → 配方比较 → dense/indexer/sparse/cooldown（CE 共 3B，indexer 另计 input）→ SFT/GRPO → 草稿 → 验收
DeepSeek：Text-v2 2.5B CE → 冻结文本接视觉 → Vision CPT 300M CE → SFT/QAT → 12 教师 → OPD → DSpark → 验收
```

正式入口 `--run-kind strategy` 需要 `--strategy-plan`、`--strategy-phase`、
`--strategy-evidence`，验证来源、配置、前驱证据和对应长度/模态/卡数 profile。
学习率、MTP、tokenizer 质量及补种子对照未完成前，不能将两组 20M 试验直接晋级。

同一阶段使用相同配方和 `--resume <run>/checkpoint.pt` 恢复优化器/数据/RNG；
阶段迁移使用 `--init`，记录新优化器与新预算。需要结构或精度变更时显式使用
`--init-transition text-to-vision / qat / mtp-weight`；DeepSeek V1 配合
`--visual-warmup`，只更新视觉、aligner 和新视觉标记。

SFT 的 control-v1 编码、原生媒体与同步工具 RL 见
[后训练适应说明](../guides/posttraining-adaptation.md)；独立目标冻结草稿和投机推理见
[草稿适应说明](../guides/draft-adaptation.md)。这些入口已有工程验证，正式学习预算还未完成。

## 生成与效果验收

```bash
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '你好' --device cuda:0
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '图中有什么？' \
  --image /path/to/image.jpg --mode direct --effort low --device cuda:0
uv run minifrontier demo --root outputs --device cpu
```

CLI 的图像和模式选项仅表明输入路径存在，实际能力必须有对应训练和留出验收。
浏览器默认只列出通过能力验收且绑定该检查点的权重，目前没有新的合格模型。
不能把训练脚本正常退出、loss 下降、诊断记忆成功或单元测试通过说成模型可用。
