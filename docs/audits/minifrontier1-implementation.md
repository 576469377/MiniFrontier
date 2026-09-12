# MiniFrontier1.0 实现与证据边界

> 本页保留 2026-09-09 首轮融合实现与 CPU 验收的时间背景，以下测试数和资源状态不代表最新运行。P0 已于 2026-09-11 开始正式训练。后续 CUDA、语言诊断和性能证据见[实验索引](../experiments.md)，当前结构与阶段见[模型页](../models/minifrontier1.md)。

依据 [2026-09-09 融合方案](../training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md)实现 MF1 模型与训练入口；原方案正文保持不变。此页记录实现范围，不把 M1–M3 的工程测试当作 M4–M8 的训练能力结果。

| 工作 | 2026-09-09 的证据 / 状态 |
|---|---|
| 完整目标模块 | 12 KDA、2 CSA、2 QSA-MLA、四流 GR、LatentMoE、lookup、随机 native ViT、MTP 已接入 |
| 实际参数账本 | 228,235,809；主配置实例化计数，没有用解析估算替代 |
| CPU 正确性 | 模块/融合/媒体/cache/rollback/packing/MTP/optimizer 测试；见 `tests/test_minifrontier1.py` |
| 完整 16 层前后向 | 随机 224 图、49 视觉 token，loss/梯度有限，vision/projector 非零；缓存误差约 6.6e-7 |
| 训练小闭环 | 离线 pilot 暂停恢复、indexer、P2、SFT；精确恢复测试比较权重、优化器、QB、sampler、RNG |
| 有限学习证据 | 小配置 600 步、9,404 CE、约 242 秒；[原始分项与曲线](../experiments/mf1-reference-v2/README.md)，有视觉依赖，也有算术过拟合 |
| 原生媒体数据 | 本地 schema、hash/尺寸/时间戳校验、分组去重、独立 tokenizer、labels/positions 编码 |
| 正式数据 | 尚未准备；candidate/fixture 均不自动通过 admission，感知去重和污染/人工审计待完成 |
| 3090 | 尚未完成 MF1 全阶段 profiling；本次 CPU 数据不能替代单卡 24GB 准入 |
| 20M/200M/第二 seed | 尚未开始 MF1 架构比较；旧三模型试验不替代此证据 |
| 3B 主预训练 | 未开始；门禁要求真实数据、性能、阶段权重与评测绑定 |
| RL / OPD / draft | 数值与小规模执行路径可运行，教师资格/训练收益/推理加速待验收 |
| Demo / export | 显式诊断权重、输入媒体预算预览、独立导出；无已验收可用模型 |

## 原方案列出的后续工作（历史清单）

以下为首轮实现时的待办。后续已加入合批注意力、专家执行优化和紧凑分片加载，见[性能与实现更新](minifrontier1-execution-performance.md)。当前将全面架构消融与正式后训练放到基础模型完成后，执行顺序以[预训练主计划](../pretraining-plan.md)为准。

1. 用 5–10GB 已审计训练文本做 32K/64K tokenizer 的中文/OCR/代码切分、byte-NLL 与成本比较，冻结词表。
2. 正式来源许可与数据清洗审计、跨源感知/近重复、污染检查，准备足够 unique 图像/视频和分层评测。
3. 将 reference 调度优化为经过等价验证的高吞吐执行，测 512/2K/4K/8K、833-token OCR、800-token video、SFT/RL/OPD/draft 的 3090 真实峰值与吞吐。
4. 做方案列出的机制、20M、200M、参数匹配和第二 seed 消融，再冻结架构配方。lookup/CSA/GR 保留与否由实验决定。
5. 完成八教师领域/effort 数据、长时间 on-policy 采样预算/吞吐验收、草稿分布/加速验收和最终模型卡。当前八槽顺序诊断、领域/effort 选择和资格评测已实现，实际合格教师集合仍为空。
6. 大规模分片 loader、阶段边界多级 checkpoint 保留、全面路由/模块 RMS 诊断、量化校准/QAT、发布能力 gate 仍有工程完善空间。现有接口拒绝未获资格的正式阶段，不把这些条件写成“已训练”。

## 源码与旧实验隔离

当时三个来源模型沿用各自的实现版本。MF1 小规模过拟合在独立代码目录中用 CPU 执行；默认预留 50 GiB 磁盘。检查点与原始诊断媒体留在 Git 忽略的 `outputs/`；公开轻量报告不包含模型权重和大数据。

[本次验证清单](minifrontier1-validation.json)：293 项 CPU 回归通过，1 项跳过；Ruff、格式、Mypy 通过。wheel 与 sdist 构建/Twine 通过，独立环境在 checkout 外跑通四模型示例、原生图片 CLI 与包含许可正文的导出；Demo 的页面、预览和生成 API 已本地验证后关闭。未运行 MF1 CUDA 性能验收，也未声明远端 CI 已执行。

来源映射记录 Kimi c5d1dd4、Qwen Transformers 4177486、DeepSeek 60d8d70 的本地适配文件 SHA；QSA-MLA 与媒体分段等组合属于本项目新设计，不声明官方为组合效果背书。
