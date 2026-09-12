# educational-v1 初版审查（2026-09-07）

本页记录从模块演示到文本训练入口的首次整理。后来的对话效果检查失败，见[复盘](../training-failure-v1.md)；当前使用入口见[训练指南](../guides/training.md)。

## 审查结论

起始版本中，MiniQwen4 已从固定 Transformers revision 提取 PLE、GDN、MoE、GR 与 QSA，并有源码、梯度、缓存、优化器和双 rank 恢复测试。Kimi 只有 AttnRes，DeepSeek 只有非量化专家；项目尚缺数据构建、连续训练、恢复游标、阶段命令与 Demo。初始 CPU 基线为 87 项通过。

## 命名

| 展示名 | 来源与代码名 |
|---|---|
| MiniQwen4 | Qwen3.8-Flash-Next 的 `qwen4_exp` 架构标识，本项目名称 |
| MiniKimi-K3 | 保留 Kimi-K3 系列名；Python 类 `MiniKimiK3ForCausalLM` |
| MiniDeepSeek-V4 | 对齐 V4-Flash 固定源码；Python 类 `MiniDeepSeekV4ForCausalLM` |

来源：[Qwen 介绍](https://qwen.ai/blog?id=qwen3.8-flash-next)、[Kimi-K3 仓库](https://github.com/MoonshotAI/Kimi-K3)、[DeepSeek-V4-Flash 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)。

## 已处理的问题

| 问题 | 处理 |
|---|---|
| Kimi MoE 拒绝训练，专家分发处于 `no_grad` | 接入可反传分发，与独立原始推理类比较前向 |
| Kimi `dt_bias`、路由修正偏置没有初始化 | 显式初始化，保留报告中的 A_log=0，单列本地取值 |
| DeepSeek 全局 TP、推理模式和原地缓存不适合 DDP | TP=1、函数式全序列注意力、完整序列 LM head |
| 整数 hash 路由表不能求梯度 | 仅对浮点参数设置梯度开关 |
| mHC 动态混合在 autocast 下可能降精度 | 明确 FP32 边界，保留 Sinkhorn 迭代与矩阵方向 |
| 路由偏置缺少更新 | 跨卡汇总，每优化步更新，排除梯度重计算的重复计数 |
| 缺少数据与 tokenizer | 固定公开 revision 前缀、阶段内去重、分组划分、assistant-only mask、自训 65,536 BPE |
| 中断后不能恢复 | 原子检查点、全局游标、优化器、各 rank RNG 与配方校验 |
| 训练后没有体验入口 | 权重导出、CLI 生成、本地浏览器 Demo |

## 数据与效果边界

数据取自 MiniMind 公开文件的确定性前缀。SFT/DPO 按相同用户问题分组划分，阶段内精确去重；tokenizer 只使用训练文本。来源 revision、文件 hash、过滤计数和监督 token 写入 manifest。当时没有跨库近重复或基准污染检查，前缀样本也不能代表全库分布。

GRPO 使用本项目生成的算术题和整数答案奖励。MOPD 功能检查以 SFT/DPO 的不同检查点作为本地教师，要求相同 tokenizer；这些权重没有领域教师训练结果。

数据、训练和 Demo 的组织参考 [MiniMind](https://github.com/jingyaogong/minimind) 与 [MiniMind-V](https://github.com/jingyaogong/minimind-v)，模型主干仍来自各自的上游代码。

## 仍需完成

当时的缺项包括三模型 MTP、Kimi/Qwen 原生视觉、Kimi/DeepSeek 整模型数值对照、长序列与精度验证、MXFP4/FP8 QAT、模型级缓存，以及更大语料和能力评估。

该版 `training_ready=true` 表示文本训练入口可执行；完整复现缺项由 `missing` 和 `complete_model_parameters=null` 保留。
