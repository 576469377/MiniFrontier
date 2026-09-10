# 架构与目录

MiniFrontier1.0 是独立融合主线，原三模型保留为来源实现与实验对照。统一模型清单位于 [configs/models.json](../configs/models.json)，四个模型通过 `models/factory.py` 构造；可执行入口和能力验收分别记录。

## 模型与流程

| 路径 | 职责 |
|---|---|
| `minifrontier/models/minifrontier1/` | KDA / CSA / QSA-MLA、四流 GR、LatentMoE、lookup、随机 ViT、MTP、缓存与草稿 |
| `minifrontier/models/miniqwen4/` | 固定 Qwen 计算与原生视觉、PLE/GDN/QSA、四流 MTP 和缓存 |
| `minifrontier/models/minikimik3/` | 固定 Kimi decoder、KDA/MLA、LatentMoE、AttnRes、原生视觉与 MTP |
| `minifrontier/models/minideepseekv4/` | 固定 DeepSeek block、压缩注意力、MoE/mHC、视觉迁移、MTP 与 DSpark |
| `minifrontier/data/` | 四个模型的数据来源、清洗、分组、tokenizer、媒体和训练编码；`text.py` 保留旧文本流程，`corpus.py` 执行 strategy-v2，`minifrontier1*.py` 执行融合数据规则 |
| `minifrontier/training/` | 四个模型的训练、恢复与后训练；`runtime.py` 等共用基础能力，`minifrontier1*.py` 保存融合配方差异，与已有模型专用优化器并列 |
| `minifrontier/evaluation/` | 独立生成评测、视觉/时序对照；训练中的验证损失仍由训练器调用 |
| `minifrontier/inference/` | `runtime.py` 统一加载四模型权重并生成；`demo.py` 负责来源模型实验页，`minifrontier1*.py` 负责融合媒体生成、导出和 Demo |
| `minifrontier/commands/` | 命令编排和离线示例；`minifrontier1.py` 承接现有 `mf1` 子命令，`quickstart.py` 承接原三模型最小示例 |
| `minifrontier/cli.py` | 顶层命令解析与委派，保留已有用户命令 |
| `configs/minifrontier1/` | 融合配置、各阶段预算、数据/教师/评测规则及来源映射 |
| `configs/strategies/` | 原三模型机器计划与策略配置 |

MF1 的阶段累计和恢复由自己的训练器管理，主干 moments 跨 indexer 阶段保留。原三模型的阶段迁移仍按各自入口与方案执行，不能把不同训练器的 `--init` / `--resume` 语义混用。操作入口集中在 [guides](guides/README.md)。

## 扩展与迁移约定

1. 包目录按功能划分；新增模型的结构放在 `models/<完整模型名>/`，训练、数据或推理差异放入对应功能目录。专用模块使用 `minifrontier1_*` 等完整模型名前缀。
2. 模型结构、数据处理和训练计算不依赖命令解析或浏览器服务。`commands/` 组合各模块；Demo 调用共用推理，命令文件不新增训练算法。
3. 可共用的检查点保存、随机状态、存储预算、分布计算等继续复用已有模块。阶段状态和损失含义不同的训练器显式区分，整合前验证行为一致。
4. 根包仅保留入口、模型清单和硬件/存储/来源等跨功能模块。数据处理不再新增根层 `data_v3.py` 一类文件，也不再新增包含整套数据/训练/Demo 的模型专属根目录。
5. 科学试验统一登记在 `configs/experiments.json`，工作站命令实例保存在各自忽略的 `outputs/<cohort>/queue-plan.json`。新的独占任务复用同一个队列入口；历史 launcher 保留可追溯性，不继续增加版本化 Python 调度器。当前计划、通用管理与日期快照分别集中在[计划](experiments/current-plan.md)、[管理](operations/experiment-management.md)和[实验索引](experiments.md)。

2026-09-09 的迁移移除了根包下的 `mf1/`：数据与编码进入 `data/minifrontier1*.py`，六个训练模块进入 `training/minifrontier1*.py`，评测进入 `evaluation/minifrontier1.py`，生成/导出/Demo 进入 `inference/minifrontier1*.py`，命令进入 `commands/minifrontier1.py`。旧数据模块归入 `data/`，原 `inference.py` 拆成运行时、浏览器和参数解析。

`minifrontier mf1 ...`、`train`、`generate`、`demo`、`quickstart` 命令及检查点存储字段保持原有语义；`minifrontier.data`、`minifrontier.inference` 的常用函数导入仍可用。原 `minifrontier.mf1.*`、`data_v2` 等内部 Python 路径已迁移，开发脚本应使用上表的新路径。

目录与源码改动会改变源码校验值。旧实验需要精确恢复时继续使用各自的冻结源码；新 checkout 不绕过来源绑定检查。已有检查点可通过共用加载器用于推理或作为新实验初始化。当前运行目录、tokenizer、数据清单和冻结源码保持原位。

## 来源与工程边界

`third_party/upstream/` 保存固定的官方源码和许可证；各模型的 `upstream_*.py` 是可读、可打包的派生文件，不在运行时下载或动态执行远程代码。提取规则见 [scripts](../scripts/README.md)，原始条款见 [LICENSES](../LICENSES)。原始策略正文和上游快照保留字节内容，以便复核 SHA。

融合方案中的组合、预算和媒体分段属于本项目设计。来源对照测试验证借用的原语；融合正确性、模型学习、硬件效率和最终能力各有独立验收，不能相互替代。

默认一个训练任务在一张卡执行，多卡用于独立试验或教师。原三模型保留 DDP；是否提速需按同一预算测量。MF1 的 reference 注意力包含显式投影与 Python 调度，3090 的长上下文性能仍待验收。

## 可分享材料与本地产物

| 路径 | 内容和分发约定 |
|---|---|
| `docs/guides/`、`docs/models/` | 通用操作、模型结构和当前能力状态 |
| `docs/training-strategies/` | 按日期冻结的方案，修订另建版本 |
| `docs/experiments/`、`docs/audits/` | 轻量配置/指标/曲线与有时间边界的审计 |
| `docs/operations/`、`docs/legacy/` | 本机操作记录与历史设计；不作为默认入门说明 |
| `data/`、`tokenizers/` | 本地数据与独立训练产物；不提交 |
| `outputs/` | 训练状态、检查点、完整日志、冻结源码、TensorBoard 与本地构建结果；不提交 |
| `.venv/`、缓存、`dist/`、`build/` | 可重新生成的环境与构建产物；不提交 |

已结束旧实验的产物保留与清理记录见[本地产物保留规则](operations/artifact-retention.md)。

wheel 提供模型代码、配置与参考入口，sdist 另保留开发/文档资源。正式策略训练要求 Git checkout、原策略和实际数据/前驱证据，完整范围见[分发说明](releases/v0.1.0.md)。
