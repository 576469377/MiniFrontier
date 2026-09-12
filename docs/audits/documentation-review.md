# 文档与仓库核对（2026-09-12）

逐篇阅读 **76 份项目 Markdown**，覆盖首页、模型、指南、实验、运维、发布、历史方案及社区模板。本轮整合已有未提交修订，统一内容与实际配置；未移动活跃训练的源码或数据。六份日期方案保留原文，其历史状态由方案索引解释。

## 修正内容

- 当前安排集中在预训练主计划；模型页解释架构与能力，指南提供操作，实验档案保留数值和结论。旧链接保留短入口。
- 四模型状态更新为首阶段正式训练已启动。区分现有首阶段库存、后续数据目标，以及“代码支持”“实际训练过”“能力通过”。
- 工作配方同步实际 microbatch、来源和数据绑定；更正 DeepSeek 未生效的 Muon 参数误述。旧 20M pilot、50+200 性能测量和完整人工内容审核不再作为重复前置任务。
- 合并重复叙述，将冗长启动流水放入按阶段展开的历史记录；保留失败结果、日期、配置和重画曲线所需数值。
- 当前进度命令默认读取四条正式训练，结合实际进程与最新 JSONL，避免把滞后的检查点状态当作当前进度。TensorBoard 分类与原日志含义统一。
- CI 和贡献者安装说明补齐 data、monitoring 测试依赖；补全一处数据类型标注，更新策略测试的授权状态模拟。
- 69 份非活跃编码完成远端校验归档，回收本地载荷约 8.14 GiB，编码缓存降至 14.79 GiB；活跃依赖及历史元数据保留。

主计划保持 2B / 3B / 2.5B / 3B 主 CE 预算、随机首阶段初始化和各模型日程。运行中的模型结构、损失、样本顺序、batch 和优化器计算未因文档整理改变。

## 验证

- 完整 CPU 回归：**618 通过、1 跳过**；54 个 CUDA 标记测试未运行。跳过项为 Qwen 未声明原生 MX QAT 配方。
- Ruff 检查与格式检查通过；Mypy 对 190 个源码文件通过。
- 三个来源模型的公开首阶段配方绑定与各自冻结运行逐项一致。六份日期方案完整复读，共 2811 行，原文字节保留。
- 791 个本地链接、52 个章节锚点、代码围栏及折叠结构检查通过；文档中的 181 份 JSON、10 份 SVG 及 36 份配置 JSON 可解析，3 份上游图示的文件校验值匹配。关键 CLI 使用参数帮助和示例语法核对，不以新增 GPU 训练验证文案。
- wheel 与 sdist 构建及 Twine 严格检查通过。解压 wheel 的独立前缀可读取模型配置，正式策略入口明确要求 Git checkout；没有修改训练共用环境。
- 打包内容不含训练权重、原始语料或数据库；组件许可元数据与导出许可文件一致。上游图示保留各自来源和权利说明。

首次 CPU 回归发现一个测试模拟遗漏授权字段，已修正；另一项恢复检查在并发编辑期间未通过源码身份校验，隔离复查和完整回归均通过。恢复保护未放宽。

## 尚未完成

完整基础预训练和可用聊天权重尚未交付。下一阶段仍缺部分领域文本、正式视频、阶段数据绑定与实际父检查点评估；已完成构造的代码/数学增量仍待合并去重和编码。DeepSeek 的 4000 步续写观察仍有重复，随后沿原配方恢复。

本次没有逐一重新访问全部外部链接，没有完成从零安装全部依赖或原始语料的外部重建，也没有重做全部上游论文实验。私密反馈渠道仍需维护者补充；来源声明不等同于对所有数据或图示授予统一许可。

[当前训练与曲线快照](../experiments/2026-09-10-pretraining-cutover/formal-progress.json)和[执行优化记录](training-infrastructure.md)分别保存训练观察与数值依据。

## 逐篇范围

以下各篇均已完整阅读。日期方案和历史失败数值保留，当前内容按职责整理。

| 文档 | 核对重点 |
|---|---|
| [.github/pull_request_template.md](../../.github/pull_request_template.md) | 导航、术语、维护信息与范围 |
| [CHANGELOG.md](../../CHANGELOG.md) | 导航、术语、维护信息与范围 |
| [CODE_OF_CONDUCT.md](../../CODE_OF_CONDUCT.md) | 导航、术语、维护信息与范围 |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | 开发依赖、验证命令与贡献范围 |
| [MODEL_CARD_TEMPLATE.md](../../MODEL_CARD_TEMPLATE.md) | 导航、术语、维护信息与范围 |
| [README.md](../../README.md) | 首次使用、项目定位、训练状态与数据/许可入口 |
| [SECURITY.md](../../SECURITY.md) | 导航、术语、维护信息与范围 |
| [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) | 组件许可、数据与图示的分发范围 |
| [docs/README.md](../README.md) | 导航、术语、维护信息与范围 |
| [docs/README.zh-CN.md](../README.zh-CN.md) | 短兼容入口及其目标 |
| [docs/architecture.md](../architecture.md) | 导航、术语、维护信息与范围 |
| [docs/assets/README.md](../assets/README.md) | 图示来源、模块对应关系与权利说明 |
| [docs/assets/upstream/README.md](../assets/upstream/README.md) | 图示来源、模块对应关系与权利说明 |
| [docs/audits/README.md](README.md) | 实现与数值证据，区分历史结果和当前结论 |
| [docs/audits/documentation-review.md](documentation-review.md) | 本轮范围、检查结果与限制 |
| [docs/audits/minifrontier1-execution-performance.md](minifrontier1-execution-performance.md) | 实现与数值证据，区分历史结果和当前结论 |
| [docs/audits/minifrontier1-implementation.md](minifrontier1-implementation.md) | 实现与数值证据，区分历史结果和当前结论 |
| [docs/audits/strategy-implementation-v2.md](strategy-implementation-v2.md) | 实现与数值证据，区分历史结果和当前结论 |
| [docs/audits/training-infrastructure.md](training-infrastructure.md) | 实现与数值证据，区分历史结果和当前结论 |
| [docs/demo-experiments.md](../demo-experiments.md) | 短兼容入口及其目标 |
| [docs/draft-adaptation.md](../draft-adaptation.md) | 短兼容入口及其目标 |
| [docs/experiments.md](../experiments.md) | 按问题导航、正式快照、状态命令与复现方法 |
| [docs/experiments/2026-09-09-preview/README.md](../experiments/2026-09-09-preview/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-batch-frontier/README.md](../experiments/2026-09-10-batch-frontier/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-batch-tuning/README.md](../experiments/2026-09-10-batch-tuning/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-exclusive-remote.md](../experiments/2026-09-10-exclusive-remote.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-mf1-update/README.md](../experiments/2026-09-10-mf1-update/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-pretraining-cutover/execution.md](../experiments/2026-09-10-pretraining-cutover/execution.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-pretraining-cutover/working-recipes.md](../experiments/2026-09-10-pretraining-cutover/working-recipes.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/2026-09-10-recipe-snapshot/README.md](../experiments/2026-09-10-recipe-snapshot/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/current-plan.md](../experiments/current-plan.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/mf1-gpu-mechanism-v1/README.md](../experiments/mf1-gpu-mechanism-v1/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/mf1-language-performance-v1/README.md](../experiments/mf1-language-performance-v1/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/mf1-reference-v1/README.md](../experiments/mf1-reference-v1/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/mf1-reference-v2/README.md](../experiments/mf1-reference-v2/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/experiments/preview-quickstart/README.md](../experiments/preview-quickstart/README.md) | 问题、变量、实际数值、时间边界和复现入口 |
| [docs/guides/README.md](../guides/README.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/data-sources.md](../guides/data-sources.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/demo-experiments.md](../guides/demo-experiments.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/draft-adaptation.md](../guides/draft-adaptation.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/minifrontier1.md](../guides/minifrontier1.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/posttraining-adaptation.md](../guides/posttraining-adaptation.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/quickstart.md](../guides/quickstart.md) | 命令、产物、依赖与适用范围 |
| [docs/guides/training.md](../guides/training.md) | 命令、产物、依赖与适用范围 |
| [docs/legacy/minideepseekv4-text-v1.md](../legacy/minideepseekv4-text-v1.md) | 保留历史设计/结果，标明现行入口 |
| [docs/legacy/minikimik3-text-v1.md](../legacy/minikimik3-text-v1.md) | 保留历史设计/结果，标明现行入口 |
| [docs/legacy/miniqwen4-text-v1.md](../legacy/miniqwen4-text-v1.md) | 保留历史设计/结果，标明现行入口 |
| [docs/legacy/project-review-educational-v1.md](../legacy/project-review-educational-v1.md) | 保留历史设计/结果，标明现行入口 |
| [docs/legacy/training-educational-v1.md](../legacy/training-educational-v1.md) | 保留历史设计/结果，标明现行入口 |
| [docs/minifrontier1.md](../minifrontier1.md) | 短兼容入口及其目标 |
| [docs/models/minideepseekv4.md](../models/minideepseekv4.md) | 结构、参数、来源、入口及实现/训练/能力状态 |
| [docs/models/minifrontier1.md](../models/minifrontier1.md) | 结构、参数、来源、入口及实现/训练/能力状态 |
| [docs/models/minikimik3.md](../models/minikimik3.md) | 结构、参数、来源、入口及实现/训练/能力状态 |
| [docs/models/miniqwen4.md](../models/miniqwen4.md) | 结构、参数、来源、入口及实现/训练/能力状态 |
| [docs/operations/artifact-retention.md](../operations/artifact-retention.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/exclusive-gpu-queue.md](../operations/exclusive-gpu-queue.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/experiment-management.md](../operations/experiment-management.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/local-training.md](../operations/local-training.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/mf1-language-performance.md](../operations/mf1-language-performance.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/mf1-mechanism-experiments.md](../operations/mf1-mechanism-experiments.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/operations/shared-gpu-experiments.md](../operations/shared-gpu-experiments.md) | 当前管理方法、资源边界和历史调度背景 |
| [docs/posttraining-adaptation.md](../posttraining-adaptation.md) | 短兼容入口及其目标 |
| [docs/pretraining-plan.md](../pretraining-plan.md) | 实际配置、阶段预算、当前数据缺口、恢复和资源限制 |
| [docs/project-review.md](../project-review.md) | 原审查的日期、历史状态和当前入口 |
| [docs/quickstart.md](../quickstart.md) | 短兼容入口及其目标 |
| [docs/releases/v0.1.0.md](../releases/v0.1.0.md) | 研究预览定位、安装包支持范围和未完成训练 |
| [docs/training-failure-v1.md](../training-failure-v1.md) | 失败指标、原因与历史边界 |
| [docs/training-strategies/2026-09-08/01-MiniKimi-K3-全流程训练与结构改造方案.md](../training-strategies/2026-09-08/01-MiniKimi-K3-全流程训练与结构改造方案.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/2026-09-08/02-MiniQwen4-全流程训练与结构改造方案.md](../training-strategies/2026-09-08/02-MiniQwen4-全流程训练与结构改造方案.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/2026-09-08/03-MiniDeepSeek-V4-全流程训练与结构改造方案.md](../training-strategies/2026-09-08/03-MiniDeepSeek-V4-全流程训练与结构改造方案.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md](../training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/2026-09-10/05-four-model-pretraining-execution-plan.md](../training-strategies/2026-09-10/05-four-model-pretraining-execution-plan.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/2026-09-10/06-recipe-initialization-review.md](../training-strategies/2026-09-10/06-recipe-initialization-review.md) | 完整复读，保留绑定原文；历史待办不作为现行前置任务 |
| [docs/training-strategies/README.md](../training-strategies/README.md) | 六份原文与现行实现/执行规则的差异 |
| [docs/training.md](../training.md) | 短兼容入口及其目标 |
| [scripts/README.md](../../scripts/README.md) | 实际脚本、正式状态入口与历史兼容命令 |
