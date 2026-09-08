> 历史 educational-v1 文档，保留作失败对照。当前入口见 [训练指南](../training.md)。

# 项目审查与整理（2026-09-07）

## 审查结论

原有工作最扎实的部分是 MiniQwen4：已经从固定 Transformers revision 提取文本计算，保留 PLE、GDN、MoE、gated residual 和 QSA，拥有源码 oracle、梯度、缓存、优化器和双 rank 恢复测试。应继续建立在这些实现之上。

原有 MiniKimiK3 只有 AttnRes，MiniDeepSeekV4 只有非量化专家。项目没有数据构建器、连续训练循环、可恢复数据位置、完整阶段命令或可体验 demo。初始 CPU 基线为 87 项通过。这些缺口使模型名称和目录看上去比实际实现更完整。

## 命名

- `MiniQwen4`：官方 Qwen3.8-Flash-Next 介绍明确以 Qwen4 为探索架构，固定源码标识为 `qwen4_exp`；这是项目架构名，不冒充官方 Qwen4 产品。
- `MiniKimi-K3`：保留官方仓库 Kimi-K3 的连字符。
- `MiniDeepSeek-V4`：具体对齐 V4-Flash 的固定源码，保留系列版本名。
- Python 类仍为 `MiniKimiK3ForCausalLM` / `MiniDeepSeekV4ForCausalLM`，包名不含连字符。

来源：[Qwen 官方介绍](https://qwen.ai/blog?id=qwen3.8-flash-next)、[Kimi-K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3)、[DeepSeek-V4-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)。

## 已处理的问题

| 原问题 | 处理 |
|---|---|
| Kimi MoE 在训练模式抛错，专家分发被 `no_grad` 包裹 | 保留门控和专家结构，接入可反传 dispatch；与独立原始推理类比较前向 |
| Kimi `dt_bias` / 路由修正偏置没有随机初始化 | 显式初始化；K3 报告的 A_log=0 保留，其余本地选择单独标注 |
| DeepSeek 源码全局 TP 状态、推理模式和原地缓存不适合 DDP 训练 | 固定单副本 TP=1，函数式全序列 attention，完整序列 LM head |
| DeepSeek 整数哈希表不能打开梯度 | 保留整数路由表，梯度开关只作用于浮点参数 |
| mHC 压缩和动态混合在 autocast 下可能降精度 | 明确 FP32 计算边界，保留 Sinkhorn 迭代和矩阵方向 |
| 专家负载修正偏置缺训练更新 | 跨卡汇总路由计数，每优化步更新；activation checkpoint 重算不重复计数 |
| 没有数据和 tokenizer | 下载固定公开 revision 前缀、按阶段去重、固定划分、assistant-only mask、自训 65,536 BPE |
| 中断后无法恢复配方和数据位置 | 原子 checkpoint、全局取样游标、优化器/各 rank RNG、严格配方校验 |
| 训练完成无法试用 | 可导出权重、CLI 生成和本地浏览器 demo |

## 数据与效果边界

当前语料是 MiniMind 公开数据的确定性文件前缀，含公开整理及合成数据。前缀抽样不代表总体分布。SFT/DPO 按相同用户问题分组划分，阶段内精确去重；不声称进行了跨全网模糊去重或基准污染清理。tokenizer 只见训练集文本。来源 revision、文件哈希、过滤计数和实际监督 token 数保存在数据 manifest。

GRPO 的算术任务由本项目生成，奖励是整数字符串的精确匹配。MOPD 需要显式提供同 tokenizer 的两个以上本地教师。功能测试使用 SFT/DPO 的不同检查点充当教师，只证明计算链路；没有把它们描述为官方九个领域/推理档位专家。

参考 MiniMind 的低门槛实践路线和 MiniMind-V 的数据/训练/demo 组织方式，但本轮主干来自上述官方固定源码，不将 MiniMind 的常规 Transformer 替换进去。[MiniMind](https://github.com/jingyaogong/minimind)、[MiniMind-V](https://github.com/jingyaogong/minimind-v)。

## 仍需完成

1. 三模型 MTP 的训练接入与独立源码/报告对照；已有的 DeepSeek MTP 推理块提取不等于训练完成。
2. Qwen 和 Kimi 的原生视觉编码、融合和多模态数据/训练；目前 demo 为文本。
3. Kimi/DeepSeek 训练适配的整模型数值 oracle，以及更长序列和更多训练精度的系统验证。
4. 原生 MXFP4/FP8 QAT 与部署优化、Kimi/DeepSeek 模型级增量缓存。
5. 更大且更均衡的语料、领域教师、真实能力评测；旗舰未公开的数据与超参数不能编造。

因此，`training_ready=true` 的定义是“存在已验收的文本教学训练入口”；`missing` 和 `complete_model_parameters=null` 明确保留完整复现的边界。
