# 实验档案

四个模型已于 2026-09-11 开始正式预训练。当前阶段、预算与执行安排见[预训练主计划](pretraining-plan.md)；本页索引历史实验、数值记录和复现方法。带日期的快照保留采集时状态，实时进度从本地训练日志读取。

## 阅读顺序

| 内容 | 入口 |
| --- | --- |
| 首版配方及选择依据 | [工作参数](experiments/2026-09-10-pretraining-cutover/working-recipes.md) |
| 四模型正式训练进度与可重画曲线 | [2026-09-12 数值快照](experiments/2026-09-10-pretraining-cutover/formal-progress.json)；固定验证全部保留，训练曲线按 50 步抽样 |
| 正式数据、失败修复与开训记录 | [2026-09-10—11 启动档案](experiments/2026-09-10-pretraining-cutover/execution.md) |
| 正式训练中的数据预取、优化器与显存处理 | [2026-09-11 执行优化](audits/training-infrastructure.md) |
| MF1 注意力、KDA、专家和数据加载优化 | [MF1 性能报告](audits/minifrontier1-execution-performance.md) |
| microbatch、实际全局 batch 与 128 档位分析 | [batch 边界与复核](experiments/2026-09-10-batch-frontier/README.md) |
| Muon、AdamW、学习率与 MTP 对照 | [2026-09-10 配方快照](experiments/2026-09-10-recipe-snapshot/README.md) |
| MF1 小配置学习与完整配置语言诊断 | [CPU 小配置](experiments/mf1-reference-v2/README.md)、[228M GPU 实验](experiments/mf1-gpu-mechanism-v1/README.md)、[语言诊断](experiments/mf1-language-performance-v1/README.md)、[后续快照](experiments/2026-09-10-mf1-update/README.md) |
| 早期实验与安装示例 | [2026-09-09 快照](experiments/2026-09-09-preview/README.md)、[离线示例实测](experiments/preview-quickstart/README.md) |
| 调度、TensorBoard 与产物保留 | [实验管理](operations/experiment-management.md)、[清理记录](operations/artifact-retention.md) |

旧数据上的配方对照与正式训练分别计数。公开档案提供配置、数值和曲线，训练权重尚未发布。

## 如何读结果

- **预算完成**：实际训练消耗达到该次运行声明的预算。测试、测速和旧配方试验不计入正式预训练。
- **损失下降**：在相同验证成员、分词器与监督规则下比较学习进度。不同模型的 NLL 不作能力排名。
- **生成能力**：检查完整输出及独立任务。训练题记忆、有限梯度和低 NLL 都不能单独证明模型能正常对话。
- **吞吐**：先核对 CE/input 分母、全局 batch、模态、精度、测量窗口及共卡时段。合成计算速度与实际训练端到端速度分别报告。

早期模型记忆训练题但未通过留出题的结果，见[配方快照](experiments/2026-09-10-recipe-snapshot/README.md)；更早的语言退化见[失败复盘](training-failure-v1.md)。

## 运行产物

来源模型使用 `minifrontier train`，MF1 使用 `minifrontier mf1 train`。两类训练器的状态格式不同，读取时应按实际阶段解释。

| 文件 | 用途 |
| --- | --- |
| `run.json` | 配置、命令、seed、batch、预算、源码及数据/tokenizer 身份 |
| `metrics.jsonl` | 训练、验证和运行事件；记录频率以该次命令为准 |
| `status.json` | 保存检查点时的步数、token 账本与阶段状态；运行进度可能更新得更快 |
| `checkpoint.pt` | 模型、优化器、数据位置和随机状态，用于恢复 |
| `tensorboard/` | 训练器原始事件；正式看板从原日志生成独立的 `train / eval / perf` 视图 |
| 数据 manifest、来源审计 | 来源版本、采样、清洗、分组、编码和文件校验值 |

来源模型还会保存 `best-validation.json`、`best-model.pt` 和阶段导出的 `model.pt`。历史 `performance.json` 按各自记录的测量窗口阅读。

MF1 另有 `resolved_config.json`、`optimizer_groups.json`、`router_metrics.jsonl` 和 `checkpoint_manifest.json`。其 `budget_complete_unqualified` 表示该次预算完成，能力尚未通过；推理用 `model.pt` 由 `mf1 export` 显式导出。

## 本地状态与公开快照

正式产物位于 `outputs/strategy-base-pretraining-v1/<model>/<phase>/`。`outputs/` 与 `data/` 被 Git 忽略；克隆仓库不会获得原始语料、完整日志或权重。以下命令用于已有运行记录的训练工作区；训练时保留独立源码目录，避免开发改动影响进程。

```bash
python -m scripts.training_status --run formal
python -m scripts.experiment_registry
```

台账包含父任务、数据准备、评估与训练阶段，记录条数不等于独立实验数。具体状态语义、事件接续和 TensorBoard 操作见[实验管理](operations/experiment-management.md)。

导出前核对配置、命令、seed、运行身份、实际预算、指标和曲线是否完整。保留失败与主动停止状态，明确标记缺失的历史校验值。

```bash
# 从本地记录导出到新的日期目录，保留旧快照
uv run --no-sync python -m scripts.export_experiments --workspace "$PWD" \
  --output docs/experiments/YYYY-MM-DD-snapshot
# 绘图依赖无需加入训练环境
uv run --no-project --with matplotlib==3.10.7 python scripts/plot_experiments.py \
  docs/experiments/2026-09-09-preview
```

![2026-09-09 验证曲线](experiments/2026-09-09-preview/validation.svg)

上图横轴为优化器更新次数，纵轴为同一模型本地验证集的 LM NLL。曲线对应 2026-09-09 采集时刻；部分运行当时尚未完成，后续结果见配方快照。CSV 保留逐条曲线，JSON 补充 token 账本、性能和运行身份。

## 复现与分发范围

复现历史实验时，将 `${WORKSPACE}` 绑定到自己的工作目录，检出报告所记录的源码，并核对数据、tokenizer 和配置。校验值用于核对文件身份；大数据流程尚未完成外部逐字节重建验证，重新下载和训练词表不保证生成相同文件。首次体验使用[离线示例](guides/quickstart.md)。

公开副本归一化工作路径，移除访问凭据、设备和容器标识。数值与处理规则保留；少量项目生成的诊断题用于解释失败，外部原文、图片和完整训练日志不随档案分发。原始策略与历史快照按记录时刻解释，阅读背景见[研究方案索引](training-strategies/README.md)。

数据、图片、上游代码和模型产物各自适用来源条款；项目代码许可证不覆盖它们。来源及分发说明见[第三方说明](../THIRD_PARTY_NOTICES.md)。
