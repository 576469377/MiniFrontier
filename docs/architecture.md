# 架构与目录

MiniFrontier 按模型、数据、训练、评估和推理划分代码。MF1 与三个来源模型遵循相同目录规则，通过 [models/factory.py](../minifrontier/models/factory.py) 构造。[configs/models.json](../configs/models.json) 记录模型入口及能力字段；实际训练进度见[预训练主计划](pretraining-plan.md)。

## 模型与流程

模型结构详解：[MF1 总图、单层、注意力与 MTP](models/minifrontier1.md#模型结构) · [Kimi 官方图与 Mini 适配](models/minikimik3.md#模型结构) · [Qwen 官方图与 Mini 适配](models/miniqwen4.md#模型结构) · [DeepSeek 官方图与 Mini 适配](models/minideepseekv4.md#模型结构)。本页侧重工程目录，张量形状、层序和模块计算见对应模型页。

| 路径 | 职责 |
|---|---|
| `minifrontier/models/minifrontier1/` | KDA / CSA / QSA-MLA、四流 GR、LatentMoE、lookup、随机 ViT、MTP、缓存与草稿 |
| `minifrontier/models/miniqwen4/` | 固定 Qwen 计算与原生视觉、PLE/GDN/QSA、四流 MTP 和缓存 |
| `minifrontier/models/minikimik3/` | 固定 Kimi decoder、KDA/MLA、LatentMoE、AttnRes、原生视觉与 MTP |
| `minifrontier/models/minideepseekv4/` | 固定 DeepSeek block、压缩注意力、MoE/mHC、视觉迁移、MTP 与 DSpark |
| `minifrontier/data/` | 四个模型的数据来源、清洗、分组、tokenizer、媒体和训练编码；`text.py` 保留旧文本流程，`corpus.py` 管理来源与去重，`pretraining.py` 构造/合并候选切片，`evaluation.py` 准备固定评测排除清单，`minifrontier1*.py` 执行融合数据规则 |
| `minifrontier/training/` | 四个模型的训练、恢复与后训练；`runtime.py` 等共用基础能力，`minifrontier1*.py` 保存融合配方差异，与已有模型专用优化器并列 |
| `minifrontier/evaluation/` | 独立生成评测、视觉/时序对照；训练中的验证损失仍由训练器调用 |
| `minifrontier/inference/` | `runtime.py` 统一加载四模型权重并生成；`demo.py` 负责来源模型实验页，`minifrontier1*.py` 负责融合媒体生成、导出和 Demo |
| `minifrontier/commands/` | 命令编排和离线示例；`minifrontier1.py` 承接现有 `mf1` 子命令，`quickstart.py` 承接三个来源模型最小示例 |
| `minifrontier/cli.py` | 顶层命令解析与委派，保留已有用户命令 |
| `configs/minifrontier1/` | 融合配置、各阶段预算、数据/教师/评测规则及来源映射 |
| `configs/strategies/` | 三个来源模型机器计划与策略配置 |

训练器分别管理模型的阶段累计与恢复。MF1 在 indexer 阶段暂存主干优化器状态；三个来源模型通过显式 `pretraining-program` 按参数名继承状态。三个来源模型的普通 `--init` 路径新建优化器与计数，详见[状态继承表](experiments/2026-09-10-pretraining-cutover/working-recipes.md#显式阶段状态)。操作命令见[指南](guides/README.md)。

## 扩展与迁移约定

1. 新模型结构放入 `models/<完整模型名>/`；专用的数据、训练和推理逻辑放入对应功能目录，使用完整模型名前缀。
2. `commands/` 和浏览器服务负责组合模块。模型计算、数据处理与训练算法保持独立，可直接调用和测试。
3. 复用检查点、随机状态、存储和分布计算模块；不同训练器的损失与阶段状态分别定义，共用实现前检查行为一致性。
4. 根包仅保留入口及硬件、存储、来源等跨功能模块。扩展既有功能目录，避免新增包含整套流程的模型专属根目录或版本化副本。
5. 实验登记在 `configs/experiments.json`，实际命令保存在本地 `outputs/<cohort>/queue-plan.json`。复用[统一队列](operations/exclusive-gpu-queue.md)，历史 launcher 仅供复现。当前安排见[主计划](pretraining-plan.md)，管理方法见[实验管理](operations/experiment-management.md)，日期结果见[实验索引](experiments.md)。

原 `minifrontier/mf1/` 的数据、训练和推理模块已迁入功能目录。`minifrontier mf1 ...` CLI 保持兼容；使用旧 `minifrontier.mf1.*` 或 `data_v2` 内部路径的脚本需更新导入。

精确恢复会检查代码、数据、词表和运行配置是否与原检查点一致，代码目录迁移也可能影响这项检查。恢复历史实验时应使用原记录的代码版本与产物；将旧权重用于推理或新实验初始化则通过共用加载器完成。

## 来源与工程边界

`third_party/upstream/` 保存固定的官方源码和许可证；各模型的 `upstream_*.py` 是可读、可打包的派生文件，不在运行时下载或动态执行远程代码。提取规则见 [scripts](../scripts/README.md)，原始条款见 [LICENSES](../LICENSES)。原始策略正文和上游快照保留字节内容，以便复核 SHA。

MF1 的组合、预算和媒体分段由本项目设计。来源对照测试检查借用的计算模块；完整模型的学习效果与运行效率另行测量，见[实验档案](experiments.md)和[执行性能报告](audits/minifrontier1-execution-performance.md)。

## 可分享材料与本地产物

| 路径 | 内容和分发约定 |
|---|---|
| `docs/guides/`、`docs/models/` | 通用操作、模型结构和当前能力状态 |
| `docs/training-strategies/` | 按日期冻结的方案，修订另建版本 |
| `docs/experiments/`、`docs/audits/` | 轻量配置/指标/曲线与有时间边界的审计 |
| `docs/operations/` | 实验管理、监控、资源调度与历史工作站记录 |
| `docs/legacy/` | 历史设计和旧接口；现行操作从指南进入 |
| `data/`、`tokenizers/` | 本地数据与独立训练产物；不提交 |
| `outputs/` | 训练状态、检查点、完整日志、实验代码副本、TensorBoard 与本地构建结果；不提交 |
| `.venv/`、缓存、`dist/`、`build/` | 可重新生成的环境与构建产物；不提交 |

已结束旧实验的产物保留与清理记录见[本地产物保留规则](operations/artifact-retention.md)。

wheel 提供模型代码、配置与参考入口，sdist 另保留开发/文档资源。正式策略训练要求 Git checkout、原策略和实际数据/前驱证据，完整范围见[分发说明](releases/v0.1.0.md)。
