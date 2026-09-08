# 项目审查与整理（2026-09-08）

当前实现以 [三份训练方案](training-strategies/2026-09-08/) 为准。
`educational-v1` 已确认没有达到基本对话目标；旧权重和
[初版审查](legacy/project-review-educational-v1.md) 保留用于复盘。
目前有可训练的模型和分阶段入口，还没有完成正式全流程或可用模型验收。

## 模型与命名

| 名称 | 当前 strategy 配置 | 核心适应 | 参数量（含 MTP） |
|---|---|---|---:|
| MiniKimi-K3 | 12 层、512 hidden、32 路由专家 top-2、2 shared、64K vocab | KDA/MLA、AttnRes、QB、Per-Head Muon、MoonViT、7-step draft | 204,526,216 |
| MiniQwen4 | 16 层、512 hidden、64 路由专家 top-4、64K vocab | PLE/GDN/GR/QSA、原生视觉、四流 MTP | 513,405,536 |
| MiniDeepSeek-V4 | 12 层、512 hidden、32 专家、64K vocab | SWA128/CSA-HCA、mHC、hash route、文本 MTP、后接 Vision-Exp、DSpark | 243,983,472（文本） |

名称采用 Mini + 模型/架构名。MiniQwen4 对应方案所固定的 Qwen3.8-Flash-Next
`qwen4_exp` 源码，不能把教学项目名称当作官方 Qwen4 产品声明。包名与 Python 类
去掉连字符，展示名保留 MiniKimi-K3 / MiniDeepSeek-V4。

## 目录与职责

| 位置 | 作用 |
|---|---|
| `third_party/upstream`、各模型 `upstream_*.py` | 固定官方来源及提取代码；提取脚本和来源测试保留 |
| `minifrontier/models/<family>` | 模型、原生视觉、MTP、缓存与独立草稿 |
| `configs/strategies` | 本轮结构配置与机器可读阶段预算；根目录旧配置保留兼容用途 |
| `data_v2.py`、`native_data.py`、`multimodal.py` | 来源/分组去重、整样本编码、原生媒体和标签对齐 |
| `chat_controls.py` | SFT / rollout / 推理共用控制模板 |
| `training/train.py`、`training/train_draft.py` | 主模型与独立草稿训练、DDP、账本和恢复 |
| `training/rollouts.py`、`tool_environment.py`、`trajectory_log.py` | 原生媒体与工具轨迹、教师路由、行为概率和审计 |
| `scripts/run_recipe_pilot.py` | 诊断通过后执行独立 20M Muon/AdamW 比较；完成不自动晋级 |
| `scripts/training_status.py` | 读取实际在跑阶段、token 和该阶段实测 ETA |
| `outputs/strategy-source-*` | 每批真实训练使用的冻结源码；开发不修改在跑实现 |
| `docs/audits` | 失败复盘、数值对照、代码验证与运行快照；不能混为能力报告 |

## 已处理的训练问题

预训练使用连续文档、真实 next-token 分母；SFT 保留完整答案且仅监督 assistant。
累计窗口和 DDP 按实际有效 token 汇总，视觉暴露、视频帧、response token 分别记录。
空 CE/零优势窗口不做衰减或路由更新。模型、优化器、数据游标、RNG、配置、数据和
源码身份一同恢复，写入遵守磁盘保留量。

Kimi/DeepSeek 的专用优化器、路由与精度边界、原生视觉迁移已接入。QAT 是明确的
MX 数值仿真；不宣称 3090 原生 FP4 加速。Kimi sampled-token MOPD 与 DeepSeek
full-vocabulary reverse KL 保持独立目标。9/12 个教师槽位要求独立权重与留出提升，
不能把同一模型复制登记。

草稿训练冻结主模型，导出绑定精确目标哈希；投机推理具有拒绝重采样和状态回滚。
原生媒体进入学生/reference/对应教师；工具操作在可重置本地环境中执行，观察
不计动作损失。GPU sampler 的强制 BF16 问题已修正，概率比超界在更新前失败。

## 尚未完成的工作

正式数据来源/许可、配比和独立留出规模还未达到方案要求；当前真实视觉池只有
96 张，不能支撑百万图片课程。20M 配方试验正在进行，还需要 LR、MTP、32K/64K
质量和补种子对照。三条正式主预算、长上下文课程、SFT/QAT 校准、教师培养、
正式 RL/草稿训练与生成质量验收均未完成。

浏览器目前仍是文本演示；CLI 已支持原生多图和控制模式。浏览器媒体/模式范围、
训练后草稿接受率、confidence calibration 和实际延迟仍待完成。实验性批量专家
GEMM 未通过完整 BF16 梯度验收，实际配方保持原有专家循环。

最近完整工程回归：229 项 CPU、41 项 CUDA；这些结果不能替代实际模型的语言、
视觉、模式或工具能力。细节与剩余门槛见 [方案执行记录](audits/strategy-implementation-v2.md)。
