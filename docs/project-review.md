# 三个来源模型的项目审查（2026-09-08）

本轮审查确认三个来源模型已有可训练主干、原生视觉和分阶段入口；`educational-v1` 的基本对话目标失败。以下为当时的实现与实验范围，当前四模型进度见[预训练计划](pretraining-plan.md)，早期版本见[初版审查](legacy/project-review-educational-v1.md)。

## 模型与命名

| 模型 | 当时的 strategy 配置 | 主要模块 | 浮点参数量（含 MTP） |
|---|---|---|---:|
| MiniKimi-K3 | 12 层、hidden 512、32 专家 top-2、2 shared、64K 词表 | KDA/MLA、AttnRes、QB、Per-Head Muon、MoonViT | 204,526,216 |
| MiniQwen4 | 16 层、hidden 512、64 专家 top-4、64K 词表 | PLE、GDN、GR、QSA、原生视觉、四流 MTP | 513,405,536 |
| MiniDeepSeek-V4 | 12 层、hidden 512、32 专家、64K 词表 | SWA128/CSA-HCA、mHC、hash routing、文本 MTP | 243,983,472（文本） |

MiniQwen4 的名称对应 Qwen3.8-Flash-Next 源码标识 `qwen4_exp`。Kimi 七步草稿、DeepSeek Vision-Exp 和 DSpark 是另外接入的模块，不计入表中主模型容量。

## 目录与职责

| 范围 | 职责 |
|---|---|
| `third_party/upstream`、模型内 `upstream_*.py` | 上游快照、提取代码与来源对照 |
| `minifrontier/models/<family>` | 主干、视觉、MTP、缓存和独立草稿 |
| `configs/strategies` | 结构配置与阶段预算 |
| 数据及 `multimodal` 模块 | 来源、分组去重、完整记录编码、媒体和标签对齐 |
| `chat_controls.py` | 训练、rollout 与推理共享控制模板 |
| 主模型与草稿训练入口 | DDP、实际 token 账本、恢复及目标权重绑定 |
| rollout、工具环境与轨迹模块 | 媒体、教师路由、行为概率、工具观察与奖励 |
| 实验脚本、`docs/audits` | 配方试验、状态观察、数值对照与失败记录 |

此表按职责保留旧审查范围；重构后的文件位置见[文档导航](README.md)。

## 已处理的训练问题

预训练使用连续文档和实际 next-token 分母；SFT 保留完整答案，仅监督 assistant。累积窗口与 DDP 汇总有效 token，视觉曝光、视频帧和 response 分别计数。空 CE、零优势窗口跳过优化器衰减及路由更新；检查点恢复模型、优化器、采样游标、RNG、配方与数据身份。

Kimi/DeepSeek 专用优化器、路由、精度边界和视觉迁移已接入。QAT 为 MX 数值仿真；Kimi sampled-token MOPD 与 DeepSeek full-vocabulary reverse KL 保留独立目标。教师槽位要求互异权重和留出提升。

草稿训练冻结目标，导出绑定目标 hash；投机推理包含拒绝重采样和状态回滚。媒体进入学生、参考模型与教师，工具观察屏蔽动作损失。GPU sampler 强制 BF16 的问题已修正，概率比超限在更新前报错。

## 尚未完成的工作

当时真实视觉池只有 96 张图，正式数据、配比和独立留出规模不足。20M 配方试验正在进行，LR、MTP、词表质量与补种子对照尚未齐备；正式主预训练、长上下文、SFT/QAT、教师培养、RL 和草稿收益也没有完成结果。

浏览器当时仅支持文本，CLI 已支持原生多图与控制模式。批量专家 GEMM 未通过完整 BF16 梯度检查，实际实验继续使用专家循环。

当次工程回归为 **229 项 CPU、41 项 CUDA**。具体配置、生成结果和未完成实验见[方案执行记录](audits/strategy-implementation-v2.md)。
