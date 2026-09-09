# 训练与推理

先运行[离线 CPU／单张 3090 示例](quickstart.md)，验证安装、数据生成、PT、暂停恢复、SFT、评估和 CLI 生成。完整模型训练使用 Git checkout，按[原始策略](training-strategies/2026-09-08/)与机器计划执行。当前能力状态见[研究预览版说明](releases/v0.1.0.md)。

## 数据与配置

`configs/strategies/*-v2.json` 是当前研究容量配置，默认单卡；DeepSeek 接入视觉另用 `minideepseekv4-vision-v1.json`。根目录三个配置保留文本兼容用途，不应与当前图文/MTP 参数量混淆。

真实来源的试验数据通过 `prepare-public-data`、`prepare-tokenizers`、`encode-data` 构造；各命令的 `--help` 给出可配置来源、限制和输出路径。下载、tokenizer 训练与编码是三个独立步骤，记录来源版本、清洗、分组切分及校验值。试验池不能自动作为正式训练数据；数据条款见[第三方说明](../THIRD_PARTY_NOTICES.md)。

工作盘默认保留 50 GiB，原子检查点的临时重叠空间也要计入。将数据、下载缓存与训练产物放在容量充足的工作盘。各模型先单卡测量；多卡 DDP 每张卡都保存完整模型与优化器，实际吞吐需要测量。

## 阶段与正式准入

```text
Kimi：诊断 → 配方比较 → 联合 PT 2B CE → SFT/QAT → 9 教师 → MOPD → 草稿 → 验收
Qwen：诊断 → 配方比较 → dense/indexer/sparse/cooldown（CE 共 3B，indexer 另计 input）→ SFT/GRPO → 草稿 → 验收
DeepSeek：Text-v2 2.5B CE → 冻结文本接视觉 → Vision CPT 300M CE → SFT/QAT → 12 教师 → OPD → DSpark → 验收
```

这描述目标依赖，不是完成清单。`--run-kind strategy` 必须提供 `--strategy-plan`、`--strategy-phase`、`--strategy-evidence`，通过固定源码、数据准入、配置、前驱和对应长度/模态/卡数的性能检查。正式流程目前要求 Git checkout；wheel 的显式 acceptance 路径适用于工程验证。

诊断和 20M-token 配方比较均为 `--run-kind acceptance`，不计正式主预算。只完成 Muon/AdamW 两组还不够，学习率、MTP、tokenizer 与补种子对照仍须完成。新队列按单卡独立试验调度，允许六组并发；不自动启动正式主训练。

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

本机实时状态与设备安排另见[运行记录](operations/local-training.md)，公开数值档案见[实验记录](experiments.md)。
