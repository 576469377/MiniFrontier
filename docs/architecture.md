# 架构与目录

MiniFrontier1.0 是独立融合主线，原三模型保留为来源实现与实验对照。统一模型清单位于 [configs/models.json](../configs/models.json)，四个模型通过 `models/factory.py` 构造；可执行入口和能力验收分别记录。

## 模型与流程

| 路径 | 职责 |
|---|---|
| `minifrontier/models/minifrontier1/` | KDA / CSA / QSA-MLA、四流 GR、LatentMoE、lookup、随机 ViT、MTP、缓存与草稿 |
| `minifrontier/mf1/` | 融合模型独立的数据 contract、课程、优化器、阶段门禁、训练/恢复、评测、后训练、导出与 Demo |
| `minifrontier/models/miniqwen4/` | 固定 Qwen 计算与原生视觉、PLE/GDN/QSA、四流 MTP 和缓存 |
| `minifrontier/models/minikimik3/` | 固定 Kimi decoder、KDA/MLA、LatentMoE、AttnRes、原生视觉与 MTP |
| `minifrontier/models/minideepseekv4/` | 固定 DeepSeek block、压缩注意力、MoE/mHC、视觉迁移、MTP 与 DSpark |
| `minifrontier/training/` | 原三模型训练器，以及共用损失、路由/优化器、预算、工具环境和检查点工具 |
| `minifrontier/cli.py` | 顶层命令；`mf1` 委派给 `minifrontier/mf1/cli.py` |
| `minifrontier/inference.py` | 共用检查点加载/生成，以及原三模型的实验 Demo |
| `minifrontier/data*.py`、`native_data.py` | 原三模型的数据来源、清洗、分组、tokenizer 和媒体编码 |
| `configs/minifrontier1/` | 融合配置、各阶段预算、数据/教师/评测规则及来源映射 |
| `configs/strategies/` | 原三模型机器计划与策略配置 |

MF1 的阶段累计和恢复由自己的训练器管理，主干 moments 跨 indexer 阶段保留。原三模型的阶段迁移仍按各自入口与方案执行，不能把不同训练器的 `--init` / `--resume` 语义混用。操作入口集中在 [guides](guides/README.md)。

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

wheel 提供模型代码、配置与参考入口，sdist 另保留开发/文档资源。正式策略训练要求 Git checkout、原策略和实际数据/前驱证据，完整范围见[分发说明](releases/v0.1.0.md)。
