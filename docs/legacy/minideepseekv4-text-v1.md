# MiniDeepSeek-V4：初版文本实现（2026-09-07）

本页记录不含视觉和 MTP 的初版。当前配置与训练状态见[模型介绍](../models/minideepseekv4.md)。

来源为 [DeepSeek-V4-Flash `60d8d70`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1/inference)；原始 `model.py`、`kernel.py` 和 MIT 许可保存在[源码快照](../../third_party/upstream/deepseek-v4-60d8d70)。

## 当时容量

| 模块 | 配置 |
|---|---|
| 主干 | 12 层、hidden 512、65,536 词表、8 头 |
| 注意力投影 | head_dim 64、RoPE 16、Q/O 低秩 128、2 个输出组 |
| 注意力层序 | 前两层纯滑窗，之后压缩比 4/128 交替；窗口 64 |
| CSA indexer | 4 heads × 32、Top-8 压缩块 |
| MoE | 32 路由专家、Top-2、1 共享专家；前两层 token-ID hash routing |
| mHC | 4 streams、20 次 Sinkhorn |
| 参数 | 229,996,877 个浮点参数，另有 262,144 个整数哈希路由项；不含 MTP |

## 训练适配

上游快照提供推理代码。本版保留专家、门控、mHC 和输出头布局，将 TP 固定为单副本，去除推理模式限制，输出完整序列 logits；压缩注意力使用可反向传播的函数式张量。

Compressor 保留 gated pooling、比例 4 的前窗重叠、位置偏置、RMSNorm 和 RoPE。注意力保留共享 KV、sink、Q normalization、输出 inverse RoPE 与分组低秩投影；mHC 动态混合为 FP32，并沿用快照中的 Sinkhorn 次序。

参数为 FP32，前向使用 BF16 autocast，未模拟 FP8/FP4。共同 Hadamard 旋转在未量化的点积中抵消，因此该后端省略旋转和量化，与官方量化推理存在数值差异。

从零训练采用本地初始化：投影 Normal(.02)、修正偏置为零、mHC dynamic scale .01、静态混合含对角偏置。默认 AdamW，路由计数平衡更新率 .001。

## 阶段与证据

- `pretrain`：使用全部可见压缩块，冻结 indexer。
- `dense_distill`：冻结主干，以压缩块注意力分布训练 indexer；头汇总、重新归一化和层平均为本地集成规则。
- `sparse_cpt`：主干与 indexer 联合训练。
- SFT/DPO/GRPO/MOPD：保留稀疏选择并冻结 indexer。

短双卡流程完成文本基础阶段。独立源码对照覆盖非量化专家及比例 4/128 Compressor；其他检查包括 mHC 归一化、因果性、梯度、阶段冻结、恢复与双 rank 参数一致。

本版尚无整模型官方数值对照、MTP、QAT、长上下文测量或模型级缓存。生成重算完整前缀，注意力会物化 O(T²) 张量。
