# 架构与目录

MiniFrontier 按模型、数据、训练、评估和推理划分代码。当前六个模型版本由 [models/factory.py](../minifrontier/models/factory.py) 统一构造，分别放在六个独立模型目录中。MF1.0 与 MF1.1 各自维护配置、计算层、媒体处理和缓存。[configs/models.json](../configs/models.json) 记录入口与能力字段；实际训练进度见[预训练主计划](pretraining-plan.md)。

## 模型与流程

模型结构详解：[MF1.0](models/minifrontier1.md) · [MF1.1](models/minifrontier11.md) · [Kimi](models/minikimik3.md) · [Qwen](models/miniqwen4.md) · [DeepSeek-V4](models/minideepseekv4.md) · [DeepSeek-V4.1](models/minideepseekv41.md)。本页说明工程目录，层序、张量形状和模块计算见对应模型页。

| 路径 | 职责 |
|---|---|
| `minifrontier/models/minifrontier1/` | MF1.0 的 KDA / CSA / QSA-MLA、LatentMoE、lookup、ViT、四流 GR、MTP、媒体处理与缓存 |
| `minifrontier/models/minifrontier11/` | MF1.1 的独立计算副本，使用 Single-Pass mHC；保留媒体协议，关闭 MTP，尚无草稿实现 |
| `minifrontier/models/miniqwen4/` | 固定 Qwen 计算与原生视觉、PLE/GDN/QSA、四流 MTP 和缓存 |
| `minifrontier/models/minikimik3/` | 固定 Kimi decoder、KDA/MLA、LatentMoE、AttnRes、原生视觉与 MTP |
| `minifrontier/models/minideepseekv4/` | 固定 DeepSeek block、压缩注意力、MoE/mHC、视觉迁移、MTP 与 DSpark |
| `minifrontier/models/minideepseekv41/` | CED、CSA2、Single-Pass mHC、Engram 与前缀重算缓存；`attention.py` 管理注意力，`residual.py` 管理残差混合 |
| `minifrontier/data/` | 各模型的数据来源、清洗、分组、tokenizer、媒体和训练编码；`text.py` 保留旧文本流程，`corpus.py` 管理来源与去重，`pretraining.py` 构造/合并候选切片，`evaluation.py` 准备固定评测排除清单，`minifrontier1*.py` 执行融合数据规则 |
| `minifrontier/training/` | 各版本的训练、恢复与后训练入口；`runtime.py` 等共用基础能力，`minifrontier1*.py` 保存融合配方差异，`v41_optim.py` 提供 V4.1 / MF1.1 的 Muon / Sinkhorn 更新 |
| `minifrontier/evaluation/` | 独立生成评测、视觉/时序对照；训练中的验证损失仍由训练器调用 |
| `minifrontier/inference/` | `runtime.py` 统一加载六版本权重；`demo.py` 与 `web/` 提供六版本文本比较页，`minifrontier1*.py` 提供 MF1.0 / MF1.1 原生媒体生成、导出和独立 Demo |
| `minifrontier/commands/` | 命令编排和离线示例；通用 `quickstart` 支持 Kimi / Qwen / DeepSeek-V4，独立 `mf1 quickstart` 用 `--model-version 1.0` 或 `1.1` 选择融合版本 |
| `minifrontier/cli.py` | 顶层命令解析与委派，保留已有用户命令 |
| `configs/minifrontier1/` | 融合配置、各阶段预算、数据/教师/评测规则及来源映射 |
| `configs/strategies/` | 四个来源模型的机器计划与策略配置 |

训练器分别管理模型的阶段累计与恢复。MF1 两个版本均保留 dense → indexer → sparse 课程，在 indexer 阶段暂存主干优化器状态；来源模型通过显式 `pretraining-program` 按参数名继承状态，普通 `--init` 路径新建优化器与计数。旧三模型的接续实例见[状态继承表](experiments/2026-09-10-pretraining-cutover/working-recipes.md#显式阶段状态)，新版本安排见[V4.1 / MF1.1 方案](training-strategies/2026-09-14/07-v41-mf11-implementation-and-training.md)。操作命令见[指南](guides/README.md)。

MF1.1 保留 MF1.0 的图像、视频、控制 token 与媒体缓存协议。复用编码数据时，新视图绑定原始文件及两份配置，只允许版本与 MTP 字段变化；旧权重不能直接恢复到新 mHC 架构。

MiniDeepSeek-V4.1 的当前正式流程是纯文本稀疏预训练。视觉模块已有前后向测试，但数据编码、文本到视觉的权重迁移和图像推理入口尚未接通。其缓存保存完整前缀，每步重新计算，供一致性检查和参考生成使用；不代表增量 KV 加速。

## 扩展与迁移约定

每个模型目录拥有完整的模型计算实现，不导入或继承其他模型包的层。借用上游或已有模型的模块时，只复制必要实现，保留来源版本与许可，并用输出、梯度和权重加载对照验证。MF1.1 的 mHC 在自身 `residual.py` 中维护；V4.1 的 CSA2 位于自身 `attention.py`。

`models/` 顶层保留构造入口和跨模型基础能力：`factory.py`、`common.py`、`cache_utils.py`、`grouped_experts.py`。它们分别处理模型分派、统一输出与批次校验、缓存事务和通用专家打包运算；各模型的 `batched_experts.py` 维护自身的执行适配器。新增模型特有的注意力、路由或残差代码放回模型目录。

1. 独立模型及架构版本放入 `models/<完整模型名>/`，各自提供配置、模型和缓存入口。专用的数据、训练和推理逻辑放入对应功能目录。
2. `commands/` 和浏览器服务负责组合模块。模型计算、数据处理与训练算法保持独立，可直接调用和测试。
3. 复用检查点、随机状态、存储和分布计算模块；不同训练器的损失与阶段状态分别定义，共用实现前检查行为一致性。
4. 根包仅保留入口及硬件、存储、来源等跨功能模块。扩展既有功能目录，模型版本的副本限于 `models/`，不重复创建整套训练、数据和推理框架。
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
