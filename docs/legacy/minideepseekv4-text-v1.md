> 历史文本版本说明，保留早期容量、测试与未实现项的时间背景。当前状态见[模型页](../models/minideepseekv4.md)。

# MiniDeepSeek-V4

具体来源为 [DeepSeek-V4-Flash 固定 revision 60d8d70](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1/inference)。原始 model.py、kernel.py 及 MIT 许可位于[源码快照](../../third_party/upstream/deepseek-v4-60d8d70)。

## 当前容量

12 层、hidden 512、65,536 词表、8 attention heads；head_dim 64、RoPE 16、Q/O 低秩 128、2 个输出组。32 路由专家、Top-2、1 个共享专家，前两层保留 token-ID hash routing。

前两层使用纯滑窗，之后压缩比 4 / 128 交替；窗口 64。CSA indexer 为 4 heads × 32，Top-8 压缩块；mHC 为 4 streams、20 次 Sinkhorn。共 **229,996,877** 个浮点参数，另有 **262,144** 个整数哈希路由项；不含 MTP。

## 训练适配

固定源码提供的是推理实现。保留专家、门控、mHC block 和 head 的计算布局，关闭 TP 分片（每 rank 是完整副本），去掉推理模式限制，输出完整序列 logits。压缩注意力用函数式 PyTorch 张量替换可变 KV cache，确保池化、压缩 KV、RoPE 和 attention 可反传。

Compressor 保留学习式 gated pooling、比例 4 的前一窗口重叠、位置偏置、RMSNorm、RoPE；attention 保留共享 KV、attention sink、Q normalization、输出 inverse RoPE 与分组低秩输出。mHC 保持 FP32 动态混合和固定源码 Sinkhorn 次序。

训练使用 FP32 master weights / BF16 autocast，不模拟 FP8/FP4。indexer 的共同 Hadamard 旋转在无量化 dot product 中抵消，因此该后端不执行旋转/量化。它不等价于官方量化推理数值。

随机初始化是本地选择：投影 Normal(.02)，修正偏置置零，mHC dynamic scale .01，静态混合使用对角偏置；官方加载检查点的代码未提供这些从零训练数值。默认 AdamW、路由计数平衡更新率 .001 也属于本地配方。

## 阶段与证据

`pretrain` 使用全部可见压缩块，冻结 indexer；`dense_distill` 冻结主干，以 attention 的压缩块分布指导 indexer；`sparse_cpt` 联合训练。KL 的头汇总、压缩块重新归一化及层平均是显式本地集成，不能标为已公开的官方训练代码。

SFT/DPO/GRPO/MOPD 保留稀疏选择并冻结 indexer。短双卡流程已完成全部文本基础阶段。独立原始源码对照包括非量化专家和比例 4/128 Compressor；另有 mHC 随机矩阵归一化、因果性、梯度、索引器阶段冻结、断点恢复和双 rank 参数一致测试。

尚未完成整模型官方数值 oracle、MTP 接入、QAT、长上下文系统验收和模型级缓存。生成采用完整前缀重算；PyTorch 教学注意力的 O(T²) 张量物化不能外推至官方百万上下文能力。
