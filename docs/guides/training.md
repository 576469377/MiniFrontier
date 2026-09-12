# 三个来源模型的训练与推理

本页适用于 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4；融合模型请阅读 [MiniFrontier1.0 指南](minifrontier1.md)。首次使用先运行[离线 CPU／单张 3090 示例](quickstart.md)，熟悉数据生成、预训练、恢复、监督微调和生成。完整配置的训练按[研究方案](../training-strategies/README.md)及对应配置执行。

## 数据与配置

`configs/strategies/*-v2.json` 是当前研究容量配置，默认单卡；DeepSeek 接入视觉另用 `minideepseekv4-vision-v1.json`。根目录三个配置保留文本兼容用途，不应与当前图文/MTP 参数量混淆。

公开语料准备需要 `uv sync --locked --extra data`。`prepare-public-data` 提供小规模来源采样，`python -m minifrontier.data.pretraining` 处理本轮文本候选及合并；随后通过 `prepare-tokenizers`、`encode-data` 冻结词表并编码。各入口的 `--help` 说明参数。来源版本、清洗、划分和校验值见[数据说明](data-sources.md)。CUDA KDA 另需 `--extra training`；TensorBoard 需 `--extra monitoring`。

工作盘默认保留 50 GiB，原子检查点的临时重叠空间也要计入。将数据、下载缓存与训练产物放在容量充足的工作盘。各模型先单卡测量；多卡 DDP 每张卡都保存完整模型与优化器，实际吞吐需要测量。

本轮四模型 base 的数据、预算、配方与更严格的 80/60/50 GiB 存储规则统一见[预训练主计划](../pretraining-plan.md)。来源模型的连续阶段路径使用 `--pretraining-program`、`--pretraining-phase` 与 `--schedule program`；它保留主干优化状态、累计主 CE 和阶段谱系，启动条件由对应阶段的证据文件记录。普通 `--init` 不等于这条连续路径。工作参数和状态继承说明见[执行配方](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)。

同一模型的多份原生媒体编码可以组合成一个训练入口。组件须已完成各自的全量编码审计，并引用同一份文本编码；模型、tokenizer 和媒体处理配置必须一致。例如：

```python
from minifrontier.data.native_components import assemble_native_components

assemble_native_components(
    ["data/kimi-native-natural", "data/kimi-native-ocr"],
    "data/kimi-native-combined",
)
```

组合只保存 tokenizer 和索引描述，文本只采样一次，媒体从原组件读取。现有 `StageDataset` 自动识别该入口，继续使用原来的领域采样与恢复逻辑。需要排除样本或修正任务时，先更新各媒体组件，再重新组合。组合成功只证明这些编码可以共同消费，来源质量、跨池近重复和正式阶段准入仍按主计划核对。

## 训练阶段与启动条件

```text
Kimi：联合 PT 2B CE → SFT/QAT → 9 教师 → MOPD → 草稿 → 验收
Qwen：dense/indexer/sparse/cooldown（CE 共 3B，indexer 另计 input）→ SFT/GRPO → 草稿 → 验收
DeepSeek：Text-v2 2.5B CE → 冻结文本接视觉 → Vision CPT 300M CE → SFT/QAT → 12 教师 → OPD → DSpark → 验收
```

上述顺序是完整训练目标。`--run-kind strategy` 使用 `--strategy-plan`、`--strategy-phase`、`--strategy-evidence` 绑定源码、数据、配置和前驱状态；具体检查按阶段执行。正式流程要求 Git checkout，wheel 支持显式 acceptance 训练。

截至 2026-09-12，四个模型均已启动首阶段正式预训练。工作参数来自已有配方对照；首阶段数据和 tokenizer 已绑定，训练中继续记录实际资源与验证指标。新的短诊断使用 `--run-kind acceptance` 并单独记账，不要求在每次正式恢复前重跑配方比较或长性能实验。

## 保存、恢复与评估

每次运行保存 `run.json`、`metrics.jsonl`、`status.json`、`checkpoint.pt`、`best-model.pt` 和完成后的 `model.pt`。JSONL 包含 LM NLL、实际 token 账本、验证和运行事件。TensorBoard 按 `train`、`eval`、`perf` 分组；`performance.json` 在配置的采样窗口完成后写入。检查点状态是保存时的快照，最近更新以 JSONL 为准。

同一阶段保持相同预算、配置、卡数和数据，追加 `--resume <run>/checkpoint.pt` 恢复模型、优化器、数据游标与各 rank RNG。`--stop-after-updates N` 可在第 N 次更新保存后暂停，保留原总预算；续训时移除该选项。改变卡数、batch 或预算需要新实验身份。尚未训练的后续阶段可以补充数据绑定，已训练阶段及当前阶段的数据和配方仍严格校验。

普通阶段迁移用 `--init` 接入权重并重建优化器。正式预训练的 program 路径同时使用前述显式阶段参数，以继承符合条件的优化器状态与累计主 CE；不能仅凭 `--init` 宣称连续训练。结构或精度变更显式设置 `--init-transition text-to-vision / qat / mtp-weight`。DeepSeek 视觉接入使用 `--visual-warmup`，分别设置视觉与投影 LR。后训练和草稿入口见[后训练适应](posttraining-adaptation.md)与[草稿适应](draft-adaptation.md)。

## CLI 与浏览器

```bash
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '你好' --device cuda:0
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '图中有什么？' \
  --image /path/to/image.jpg --mode direct --effort low --device cuda:0
uv run minifrontier demo --root outputs --device cpu
```

图像和模式选项只表示输入路径可执行，实际能力需要对应训练及留出验收。浏览器默认只列出与权重 hash 绑定、通过能力验收的检查点；当前没有可用聊天权重。quickstart 完成后先通过 CLI 显式加载刚生成的权重。

实时状态的读取方法见[实验管理](../operations/experiment-management.md)，历史设备安排见[运行记录](../operations/local-training.md)，公开数值档案见[实验记录](../experiments.md)。
