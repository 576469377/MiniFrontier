> 历史文本版本说明，保留早期容量、测试与未实现项的时间背景。当前状态见[模型页](../models/miniqwen4.md)。

# MiniQwen4

MiniQwen4 是从固定官方实现派生的缩小文本模型，不是官方发布的模型名称。
目前完成文本主干、无 MTP 的语言模型适配器、基础优化与推理缓存验收；
**已提供文本教学训练管线并启动持续预训练；完整旗舰复现仍未完成。**

## 来源与容量

计算源码固定为 Hugging Face Transformers
[`4177486a9f199bd7be520eff14431071d5d41ec5`](https://github.com/huggingface/transformers/tree/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/qwen4_exp)。
发布配置核对的是 Qwen3.8-Flash-Next
[`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)。
名称中的 Qwen4 对应该源码的 `qwen4_exp` 架构标识，不代表声称复刻其他未核验版本。

| 项目 | 当前缩小配置 |
|---|---|
| 层数 / 隐藏宽度 / 词表 | 16 / 512 / 65,536 |
| 混合层 | 3 层 GDN : 1 层注意力 |
| 注意力 Q / KV / head_dim | 8 / 2 / 64 |
| GDN key / value 头数 | 4 / 12，各头维度 64 |
| MoE | 64 专家、Top-4、专家中间宽度 192 |
| 共享专家 | 中间宽度 192，保留与路由专家 1:1 的宽度关系 |
| 多路残差 | 4 路、门控低秩宽度 128 |
| PLE | 第 2 层，2/3-gram，每种 2 头，完整投影/门控/归一化/膨胀卷积 |
| QSA indexer | 4 头 × 32，压缩比 4，token budget 512 |
| 输出门控 / RoPE theta | sigmoid / 10,000,000 |
| 配置最大长度 | 4,096；不是已通过该长度的显存验收 |
| 文本主干参数 | 398,055,200 |
| 主干 + 未绑定 LM 头 | 431,609,632；**不含 MTP** |

本次核对纠正了早期新配置的 GDN 头比例、共享专家宽度、indexer 头数/维度/
压缩比、输出门控和 RoPE theta。此前记录的 408,283,072 是修正前的主干计数，
已被上述数值取代；不删除原审计记录。

## 训练与缓存实现边界

训练适配器使用官方文本层、未绑定 LM 头、next-token CE 和直接提取的官方
router auxiliary loss。基础预训练使用 full attention，indexer 保留原参数布局但冻结。
QSA 已接入两个训练目标：冻结主干的 indexer 蒸馏，以及主干/indexer 联合稀疏 CPT。
KL 按报告完成跨头求和/L1、完整块 max-pooling/L1；第二阶段只在已选块上重新归一化。
分母为全部有效 query，无完整块的早期 query 贡献零；padding 和 incomplete tail 不进入教师块目标。
教师概率和 indexer 输入 stop-gradient、跨层取平均、KL 系数属于显式本地集成选择。
评分保留固定 HF 源码的 1/sqrt(head_dim)，该尺度没有写在报告公式 15 中。

根据[固定报告 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)
实现语义分块 Muon：8 步 Polar Express、0.95 Nesterov 动量、按 Q/K/V 头拆分、
按专家拆分 gate/up；融合 Q/output-gate 中的门控行走 AdamW。PLE key/value 投影走
Muon，embedding、LM 头、router、门控等走 AdamW，n-gram 表不做 weight decay。
LR、Adam 参数和其他 decay 是明确的本地验收设置，不声称官方未公开值；
当前为复制式 DDP，未实现 Canzona/ZeRO/TP 的完整官方工程栈。

缓存覆盖 GDN 卷积/递归状态、PLE 词组/膨胀卷积历史、KV 与 QSA 索引历史。
状态更新与同一 HF revision 的原始 `cache_utils.py` 对照。
只支持 eval + no-grad 的无填充文本批次；不声称支持 beam search、rollback、
offloading、缓存序列化或训练反传。部分 forward 失败后缓存必须 reset，不能继续复用。

## 工程验收，不是训练曲线

原始官方 TextModel 与本地同权重的整栈前向、梯度已经测试；LM/router loss、
融合优化器分块以及微型双卡更新/恢复另有独立测试。缓存比较同时覆盖 CPU/CUDA、
FP32/BF16、逐 token 与分段输入、跨 EOS，以及非零 PLE 卷积权重。

实际 431,609,632 参数配置也执行了双卡一次更新，使用合成 token，**不用于评估效果**：

| 条件 / 结果 | 数值 |
|---|---|
| 设备 | 2 张 RTX 3090，物理 GPU 0 / 1 |
| 参数 / 计算 | FP32 参数，BF16 autocast，activation checkpointing |
| 每卡 batch / 序列长度 | 1 / 128 |
| 更新次数 | 1，仅验收前向、反向与优化器 |
| 单步时间 | 约 7.7 秒，不是稳定吞吐基准 |
| 每卡峰值 allocated / reserved | 5,785.76 / 5,870 MiB，不含全部驱动/NCCL 显存 |
| 更新后两卡参数 | 全部逐项精确一致 |
| 大检查点 / TensorBoard 写入 | 无 / 无 |

数值保存在[验收记录](../audits/miniqwen4-acceptance.json)，
测量发生在本轮目录整理前。长上下文、更大 batch、MTP 和其他训练阶段
必须单独测显存，不可外推此结果。

当前教学训练及 TensorBoard 位于 `outputs/miniqwen4/educational-v1/`。
较早的合成 token 单步记录作为历史审计保留；本轮真实语料的双卡阶段验收单独汇总，
不会混入教学训练曲线。见[本轮验收](../audits/training-acceptance.json)。

## 尚未完成

- MTP 的完整官方训练依据、训练接入与对照验证。
- MTP、原生视觉和量化感知后训练仍待接入。
- 更大语料、领域后训练和真实能力评测；当前配方仅为文本教学规模。

公开代码没有披露的训练细节将单独记录，不能用历史模型的简化实现替代。


## 当前可执行流程

统一入口提供 `pretrain → dense_distill → sparse_cpt → sft → dpo`，以及可选 GRPO/MOPD。
每阶段独立重建优化器和 DDP，检查点包含数据游标、调度配方及各 rank RNG；精确恢复
拒绝更改总步数、batch、卡数或数据。SFT/DPO 保留 QSA 的稀疏模式，冻结离散 indexer。
CLI 和浏览器生成使用已有 Qwen cache。[操作指南](../guides/training.md)。

本轮真实语料上完成双 3090、每阶段两步的基础流程；另外在单 3090 上验证了长度
1,024、batch 2、累积 2 的 sparse CPT 更新，峰值 allocated 约 5,785 MiB。
该序列长度超过 512 token 的 indexer budget，会实际触发稀疏选择。
这项显存实测只覆盖所述配置，不代表更长上下文或加入 MTP 后的开销。
