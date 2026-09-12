# MiniFrontier1.0 首轮实现检查（2026-09-09）

MF1 首轮实现接通了完整模型、原生媒体、训练与恢复。228M 配置完成 CPU 前后向检查；学习实验使用约 132K 参数的小配置，结果表现为训练题记忆和简单色块识别，留出算术失败。

设计依据为 [2026-09-09 融合方案](../training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md)。本页记录当时结果；后续结构见[模型介绍](../models/minifrontier1.md)，P0 正式训练与阶段安排见[预训练计划](../pretraining-plan.md)。

| 检查项 | 结果 |
|---|---|
| 模型模块 | 12 KDA、2 CSA、2 QSA-MLA、四流 GR、LatentMoE、lookup、随机初始化原生 ViT、MTP |
| 参数计数 | 主配置实际实例化：228,235,809 |
| 完整 16 层前后向 | 随机 224 图像、49 视觉 token；loss 与梯度有限，vision/projector 梯度非零 |
| 缓存对照 | 完整配置 CPU 缓存误差约 6.6e-7 |
| 小配置学习 | 600 步、9,404 CE、约 242 秒；[曲线与生成结果](../experiments/mf1-reference-v2/README.md) |
| 训练恢复 | pilot 暂停恢复、indexer、P2、SFT；比较权重、优化器、路由均衡、sampler 与 RNG |
| 媒体数据 | schema、文件 hash、尺寸、时间戳、分组去重、独立 tokenizer、labels 与 positions 编码 |
| RL / OPD / draft | 数值及小规模执行路径已接入，教师训练与推理收益尚无实验结果 |
| 导出 / Demo | 诊断权重导出、输入媒体预算预览和本地 API 已验证 |

## 原方案列出的后续工作（历史清单）

首轮检查时，正式数据、3090 全阶段测量、20M/200M 架构对照、第二 seed 和 3B 主训练均未开始。原方案的后续工作分为：

1. 用 5–10GB 审核文本比较 32K/64K tokenizer 的中文、OCR、代码切分、byte-NLL 与成本。
2. 完成来源许可、跨源近重复与污染检查，准备独立图像、视频和分层验证集。
3. 测量 512/2K/4K/8K、833-token OCR、800-token video 及后训练阶段的 3090 峰值与吞吐。
4. 完成机制、20M、200M、参数匹配与第二 seed 对照，检查 lookup、CSA、GR 的收益。
5. 训练八教师领域/effort 组合，评估在线采样成本、草稿接受率与加速；当时八槽诊断及教师选择接口已实现，合格教师集合为空。
6. 完善大规模分片加载、阶段检查点保留、路由/模块 RMS 诊断、量化校准与发布评估。

此后已完成紧凑加载、注意力和专家执行优化，见[性能记录](minifrontier1-execution-performance.md)。完整架构对照与后训练的执行顺序由现行预训练计划维护。

## 源码与旧实验隔离

检查使用独立代码目录、CPU 两线程及默认 50 GiB 磁盘预留。检查点和诊断媒体保存在 `outputs/`；三个来源模型的既有实验使用各自的实现版本。

[验证清单](minifrontier1-validation.json)记录 **293 项 CPU 通过、1 项跳过**，Ruff、格式与 Mypy 通过。wheel/sdist 构建和 Twine 检查通过；独立环境在 checkout 外完成四模型示例、原生图片 CLI 与含许可正文的导出。Demo 页面、预览和生成 API 验证后关闭。本轮没有 CUDA 性能测量或远端 CI 结果。

来源映射包含 Kimi `c5d1dd4`、Qwen Transformers `4177486`、DeepSeek `60d8d70` 的本地适配文件 SHA。QSA-MLA、媒体分段及其组合是本项目设计。
