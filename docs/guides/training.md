# 三个来源模型的训练与推理

本页适用于 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4；融合模型请阅读 [MiniFrontier1.0 指南](minifrontier1.md)。首次使用先运行[离线 CPU／单张 3090 示例](quickstart.md)，熟悉数据生成、预训练、恢复、监督微调和生成。完整配置的训练按[研究方案](../training-strategies/README.md)及对应配置执行。

## 数据与配置

`configs/strategies/*-v2.json` 是当前研究容量配置，默认单卡；DeepSeek 接入视觉另用 `minideepseekv4-vision-v1.json`。根目录三个配置保留文本兼容用途，不应与当前图文/MTP 参数量混淆。

准备公开语料前安装 `uv sync --locked --extra data`，再通过 `prepare-public-data`、`prepare-tokenizers`、`encode-data` 构造；各命令的 `--help` 说明参数与输出路径。下载、词表训练与编码是三个独立步骤，分别记录来源版本、清洗、分组划分及校验值。已用数据和审核进展见[数据来源说明](data-sources.md)。GPU KDA 还需 `--extra training`；使用 TensorBoard 时增加 `--extra monitoring`。

工作盘默认保留 50 GiB，原子检查点的临时重叠空间也要计入。将数据、下载缓存与训练产物放在容量充足的工作盘。各模型先单卡测量；多卡 DDP 每张卡都保存完整模型与优化器，实际吞吐需要测量。

本轮四模型 base 的数据、预算、配方与更严格的 80/60/50 GiB 存储规则统一见[预训练主计划](../pretraining-plan.md)。来源模型的连续阶段路径使用 `--pretraining-program`、`--pretraining-phase` 与 `--schedule program`；它保留主干优化状态、累计主 CE 和阶段谱系，仍需原正式策略的全部准入证据。普通 `--init` 不等于这条连续路径。工作参数和状态继承说明见[执行配方](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)。

## 训练阶段与启动条件

```text
Kimi：诊断 → 配方比较 → 联合 PT 2B CE → SFT/QAT → 9 教师 → MOPD → 草稿 → 验收
Qwen：诊断 → 配方比较 → dense/indexer/sparse/cooldown（CE 共 3B，indexer 另计 input）→ SFT/GRPO → 草稿 → 验收
DeepSeek：Text-v2 2.5B CE → 冻结文本接视觉 → Vision CPT 300M CE → SFT/QAT → 12 教师 → OPD → DSpark → 验收
```

这描述目标依赖，不是完成清单。`--run-kind strategy` 必须提供 `--strategy-plan`、`--strategy-phase`、`--strategy-evidence`，通过固定源码、数据准入、配置、前驱和对应长度/模态/卡数的性能检查。正式流程目前要求 Git checkout；wheel 的显式 acceptance 路径适用于工程验证。

小规模诊断和 20M-token 配方比较使用 `--run-kind acceptance`，单独记录预算。本轮 base 已依据旧对照选择工作参数；后续按主计划完成正式数据、词表和一次生产准入，不自动恢复参数网格或多 seed 探索。各正式模型使用一张卡，按实际资源独立推进。

## 保存、恢复与评估

每个 run 保存 `run.json`、`metrics.jsonl`、`status.json`、`checkpoint.pt`、`best-model.pt` 和完成后的 `model.pt`。JSONL 包含 LM NLL、实际 token 账本与验证结果；性能采样写入 `performance.json`，可选同步到 TensorBoard。

同一阶段保持相同预算、配置、卡数和数据，追加 `--resume <run>/checkpoint.pt` 恢复模型、优化器、数据游标与各 rank RNG。`--stop-after-updates N` 可在第 N 次更新保存后暂停，保留原总预算；续训时移除该选项。改变卡数、batch 或预算不能称为精确恢复。

阶段迁移用 `--init` 接入权重并重建优化器；结构或精度变更显式设置 `--init-transition text-to-vision / qat / mtp-weight`。DeepSeek 视觉接入使用 `--visual-warmup`，分别设置视觉与投影 LR。后训练和草稿入口见[后训练适应](posttraining-adaptation.md)与[草稿适应](draft-adaptation.md)。

## CLI 与浏览器

```bash
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '你好' --device cuda:0
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '图中有什么？' \
  --image /path/to/image.jpg --mode direct --effort low --device cuda:0
uv run minifrontier demo --root outputs --device cpu
```

图像和模式选项只表示输入路径可执行，实际能力需要对应训练及留出验收。浏览器默认只列出与权重 hash 绑定、通过能力验收的检查点；当前没有可用聊天权重。quickstart 完成后先通过 CLI 显式加载刚生成的权重。

本机实时状态与设备安排另见[运行记录](../operations/local-training.md)，公开数值档案见[实验记录](../experiments.md)。
