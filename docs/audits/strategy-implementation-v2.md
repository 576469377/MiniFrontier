# 2026-09-08 训练方案执行记录

本记录区分实现、数值验证与能力验收；旧 educational-v1 权重保留作失败对照。
方案原文在 `docs/training-strategies/2026-09-08`，机器可读预算位于
`configs/strategies/*-plan.json`。正式主预算、SFT、RL、草稿和发布均未完成。

已实现的基础包括三个模型的原生视觉与 processor、训练 MTP、实际 token 账本、
全 accumulation/DDP 分母、各自 Muon 与路由更新、Kimi QKClip、DeepSeek Text→Vision
迁移、不可变数据编码、分组去重、同字节 tokenizer 对比及磁盘写入保护。
新配置使用 64K 冻结词表；32K 尚未完成等墙钟质量对比，64K 是方案默认选择。

Kimi/DeepSeek 增量缓存与回滚已实现，Qwen 已增加原生视觉 prefill 位置与回滚。
FP32 保持严格全前向对照。BF16 的逐元素绝对误差测试最初失败；固定路由为 FP32
后仍有不同计算顺序的舍入误差。`cache-precision-v2.json` 保留三个种子的
FP32/BF16 对照：缓存 RMSE 小于完整 BF16 相对 FP32 的 RMSE，贪心一致率均为 1。
这不是训练后模型的生成质量或草稿接受率验收。
DeepSeek indexer 并列分数采用稳定的低索引优先顺序，以消除前缀长度改变时的选择漂移；
这是显式的本地 tie policy，不能称作原始 topk 未定义并列顺序的逐位复现。

首批公开试验数据共 44,695 条，编码后 PT train 为 34,576,919 CE token。
数学科学、真实视觉/视频、SFT 及独立验证规模仍不满足正式配比；禁止据此启动正式主预算。
独立生成的诊断集包含 256 张图像、2,048 条可核验算术文本，仅供 K0/Q0/D0 学习诊断。
相似图像被分在同一训练组，当前视觉诊断没有独立视觉留出集；反事实/记忆检查
必须标为训练内诊断，不能作为视觉泛化成绩。

待完成：正式数据准入与各域/模态采样、阶段门禁与质量选模、配方对照和实测 profile；
Kimi QAT 与九教师 MOPD、DeepSeek QAT 与十二教师全词表 reverse-KL OPD 的训练整合；
真实工具/代码/视觉奖励、草稿训练与投机解码端到端验收、通过质量门槛的 demo。
已有辅助损失函数或小测试不代表这些阶段已经完成。

训练安排为 Kimi GPU 0–1、Qwen GPU 2–3、DeepSeek GPU 4–5。
GPU 6、7 的既有任务不改动。工作盘写入保留 50 GiB；根分区不放数据和大缓存。
每个 run 绑定源码 commit/内容 hash、数据 hash、冻结 tokenizer、配置、实际 token 和
RNG/cursor/优化器状态；诊断 run 不进入默认 demo。
