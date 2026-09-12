# 三个来源模型的训练与推理

本页介绍 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4 的数据入口、阶段迁移、恢复与推理。首次运行见[离线示例](quickstart.md)，MF1 见[独立指南](minifrontier1.md)。当前阶段、预算和工作参数统一见[预训练计划](../pretraining-plan.md)；完整后续路线见[研究方案](../training-strategies/README.md)。

## 数据与配置

`configs/strategies/*-v2.json` 是当前研究容量配置，默认单卡；DeepSeek 视觉阶段使用 `minideepseekv4-vision-v1.json`。`configs/` 根目录的三个模型配置用于早期文本兼容，容量与当前图文/MTP 配置不同。

公开语料准备需要 `uv sync --locked --extra data`：`prepare-public-data` 用于小规模来源采样，`python -m minifrontier.data.pretraining` 处理文本候选与合并，再通过 `prepare-tokenizers`、`encode-data` 冻结词表并编码。参数见各入口的 `--help`，来源、清洗和划分见[数据说明](data-sources.md)。GPU 训练加装 `--extra training`，TensorBoard 加装 `--extra monitoring`。

工作盘默认保留 50 GiB，正式运行按[存储安排](../pretraining-plan.md#resources)保留更高余量，并计入原子保存时新旧检查点的重叠空间。DDP 每张卡保存完整模型与优化器，增加卡数不会降低单卡模型状态占用。

同一模型的多份原生媒体编码可组合为一个训练入口。组件须通过各自的全量编码审计，引用同一份文本编码，并使用相同模型、tokenizer 和媒体处理配置：

```python
from minifrontier.data.native_components import assemble_native_components

assemble_native_components(
    ["data/kimi-native-natural", "data/kimi-native-ocr"],
    "data/kimi-native-combined",
)
```

组合保存 tokenizer 和索引描述，文本只采样一次，媒体从原组件读取。`StageDataset` 自动识别组合目录，沿用领域采样和恢复逻辑。排除样本或修正任务时，先更新组件，再重新组合；组合不会替代来源质量、跨池去重和阶段准入检查。

## 训练阶段与启动条件

```text
Kimi：联合 PT 2B CE → SFT/QAT → 9 教师 → MOPD → 草稿 → 验收
Qwen：dense/indexer/sparse/cooldown（CE 共 3B，indexer 另计 input）→ SFT/GRPO → 草稿 → 验收
DeepSeek：Text-v2 2.5B CE → 冻结文本做视觉 warmup（另计）→ Vision CPT 300M CE → SFT/QAT → 12 教师 → OPD → DSpark → 验收
```

以上是完整路线，三个模型均已启动首阶段正式预训练，后续状态见[主计划](../pretraining-plan.md)。正式运行使用 `--run-kind strategy`，并通过 `--strategy-plan`、`--strategy-phase`、`--strategy-evidence` 绑定源码、数据、配置和前驱状态；需要 Git checkout。小规模诊断使用 `--run-kind acceptance`，可从 wheel 运行，结果单独记账。

连续预训练还需 `--pretraining-program`、`--pretraining-phase` 和 `--schedule program`，以继承主干优化状态、累计主 CE 和阶段谱系。普通 `--init` 只加载权重并重建优化器。完整启动参数与状态继承规则见[执行配方](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)。

## 保存、恢复与评估

| 文件 | 用途 |
|---|---|
| `run.json` / `metrics.jsonl` / `status.json` | 运行身份、损失、token 账本、验证及状态事件 |
| `checkpoint.pt` | 恢复训练；内容是保存时的快照 |
| `best-model.pt` / `model.pt` | 验证最优权重 / 阶段完成后的权重 |
| `performance.json` | 完成配置采样窗口后的性能统计 |

TensorBoard 按 `train`、`eval`、`perf` 分组。检查最近更新时读取日志，不能用检查点的保存进度代替实时计数。

同一阶段追加 `--resume <run>/checkpoint.pt`，恢复模型、优化器、数据游标及各 rank RNG。预算、配置、卡数和当前数据须保持一致；改变卡数、batch 或预算需要新实验。`--stop-after-updates N` 在第 N 次更新保存后暂停，保留总预算，恢复时移除。尚未训练的后续阶段可补充数据绑定，当前与已训练阶段仍严格校验。

结构或精度迁移通过 `--init-transition` 指定 `text-to-vision`、`qat` 或 `mtp-weight`。DeepSeek 视觉接入使用 `--visual-warmup`，分别设置视觉塔与投影 LR。后续入口见[后训练](posttraining-adaptation.md)和[草稿训练](draft-adaptation.md)。

## CLI 与浏览器

```bash
uv run minifrontier generate --checkpoint /path/to/model.pt \
  --prompt 'Once upon a time' --completion --device cuda:0
uv run minifrontier generate --checkpoint /path/to/model.pt --prompt '图中有什么？' \
  --image /path/to/image.jpg --mode direct --effort low --device cuda:0
uv run minifrontier demo --root outputs --device cpu
```

第一条使用预训练文本续写，第二条使用对话模板输入图片；应选择对应阶段的检查点。浏览器默认只列出通过能力验收且绑定权重 hash 的模型，目前没有可用聊天权重。观察本地训练产物可使用[实验视图](demo-experiments.md)。

日志读取见[实验管理](../operations/experiment-management.md)，公开结果见[实验记录](../experiments.md)。
