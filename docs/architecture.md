# 架构与目录

MiniFrontier1.0 是独立融合主线，三个来源模型保留为独立实现与实验对照。统一模型清单位于 [configs/models.json](../configs/models.json)，四个模型通过 `models/factory.py` 构造；模型清单分别记录训练入口的可执行范围与权重能力状态。

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

MF1 的阶段累计和恢复由自己的训练器管理，主干 moments 跨 indexer 阶段保留。三个来源模型的显式 `pretraining-program` 路径已支持按参数名继承优化器状态与累计主 CE；普通 `--init` 仍新建优化运行。同阶段 `--resume` 与跨阶段迁移的条件不同，详见[状态继承表](experiments/2026-09-10-pretraining-cutover/working-recipes.md#显式阶段状态)。操作入口集中在 [guides](guides/README.md)。

## 扩展与迁移约定

1. 包目录按功能划分；新增模型的结构放在 `models/<完整模型名>/`，训练、数据或推理差异放入对应功能目录。专用模块使用 `minifrontier1_*` 等完整模型名前缀。
2. 模型结构、数据处理和训练计算不依赖命令解析或浏览器服务。`commands/` 组合各模块；Demo 调用共用推理，命令文件不新增训练算法。
3. 可共用的检查点保存、随机状态、存储预算、分布计算等继续复用已有模块。阶段状态和损失含义不同的训练器显式区分，整合前验证行为一致。
4. 根包仅保留入口、模型清单和硬件/存储/来源等跨功能模块。数据处理不再新增根层 `data_v3.py` 一类文件，也不再新增包含整套数据/训练/Demo 的模型专属根目录。
5. 实验统一登记在 `configs/experiments.json`，工作站命令实例保存在各自忽略的 `outputs/<cohort>/queue-plan.json`。新的独占任务复用同一个队列入口；历史 launcher 保留可追溯性，不继续增加版本化 Python 调度器。当前计划、通用管理与日期快照分别集中在[计划](experiments/current-plan.md)、[管理](operations/experiment-management.md)和[实验索引](experiments.md)。

MF1 与其他模型使用相同的目录规则；原根包 `mf1/` 中的数据、训练和推理模块已迁入相应功能目录。`minifrontier mf1 ...` 等 CLI 入口保持兼容；依赖旧 `minifrontier.mf1.*` 或 `data_v2` 内部路径的脚本需更新导入。

精确恢复会检查代码、数据、词表和运行配置是否与原检查点一致，代码目录迁移也可能影响这项检查。恢复历史实验时应使用原记录的代码版本与产物；将旧权重用于推理或新实验初始化则通过共用加载器完成。

## 来源与工程边界

`third_party/upstream/` 保存固定的官方源码和许可证；各模型的 `upstream_*.py` 是可读、可打包的派生文件，不在运行时下载或动态执行远程代码。提取规则见 [scripts](../scripts/README.md)，原始条款见 [LICENSES](../LICENSES)。原始策略正文和上游快照保留字节内容，以便复核 SHA。

融合方案中的组合、预算和媒体分段属于本项目设计。来源对照测试验证借用的原语；融合正确性、模型学习、硬件效率和最终能力各有独立验收，不能相互替代。

当前四个主训练各使用一张卡，空闲设备按主计划安排；教师训练和新的参数探索在基础模型完成后推进。三个来源模型保留 DDP，使用前需确认实际收益。MF1 的无缓存稠密注意力、KDA 片段和专家计算已加入合批优化，测量条件与结果见[执行性能报告](audits/minifrontier1-execution-performance.md)。稀疏、长上下文和不同模态阶段分别记录执行表现。

## 可分享材料与本地产物

| 路径 | 内容和分发约定 |
|---|---|
| `docs/guides/`、`docs/models/` | 通用操作、模型结构和当前能力状态 |
| `docs/training-strategies/` | 按日期冻结的方案，修订另建版本 |
| `docs/experiments/`、`docs/audits/` | 轻量配置/指标/曲线与有时间边界的审计 |
| `docs/operations/`、`docs/legacy/` | 本机操作记录与历史设计；不作为默认入门说明 |
| `data/`、`tokenizers/` | 本地数据与独立训练产物；不提交 |
| `outputs/` | 训练状态、检查点、完整日志、实验代码副本、TensorBoard 与本地构建结果；不提交 |
| `.venv/`、缓存、`dist/`、`build/` | 可重新生成的环境与构建产物；不提交 |

已结束旧实验的产物保留与清理记录见[本地产物保留规则](operations/artifact-retention.md)。

wheel 提供模型代码、配置与参考入口，sdist 另保留开发/文档资源。正式策略训练要求 Git checkout、原策略和实际数据/前驱证据，完整范围见[分发说明](releases/v0.1.0.md)。
