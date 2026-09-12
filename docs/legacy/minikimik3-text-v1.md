# MiniKimi-K3：初版文本实现（2026-09-07）

本页记录不含视觉和 MTP 的初版。当前配置与训练状态见[模型介绍](../models/minikimik3.md)。

实现来自 [Kimi-K3 `c5d1dd4`](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721)，原始代码和许可位于[源码快照](../../third_party/upstream/kimi-k3-c5d1dd4)。

## 当时容量与结构

| 模块 | 配置 |
|---|---|
| 主干 | 12 层、hidden 512、65,536 词表 |
| 注意力 | 9 KDA＋3 gated MLA；MLA 使用 NoPE 和 sigmoid 输出门 |
| MoE | 32 路由专家、Top-2、2 共享专家；潜在宽度和专家中间宽度均为 256 |
| 激活 | SiTU(beta=4, linear_beta=25)，保留 latent normalization 与投影 |
| 层间连接 | 第一层 dense MLP；AttnRes 每 4 层一个块 |
| 参数 | 166,109,480 个浮点参数，不含视觉和 MTP |

## 源码适配

[提取脚本](../../scripts/extract_training_sources.py)保留 KDA/MLA、专家、门控、decoder 与 AttnRes；专家分发改用可反向传播的 `index_add`。CUDA KDA 使用 FLA 0.5.2，将旧 `transpose_state_layout` 适配为 `state_v_first`；CPU 使用逐 token delta recurrence。测试比较两条路径的前向与输入、门控梯度。

线性层和 embedding 沿用 Normal(0, .02)，A_log=0 依据 [K3 报告 §2.1](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)。推理代码未提供的 `dt_bias` 采用本地 log-uniform dt [.001,.1] / inverse-softplus 初始化；路由修正偏置和 AttnRes 查询置零。第一层没有前序残差块，相应未使用参数冻结。

默认优化器为 AdamW。路由修正偏置按跨 rank 专家计数更新，更新率 .001 为本地配方。派生代码保留 Kimi K3 自定义许可，见[第三方说明](../../THIRD_PARTY_NOTICES.md)。

## 训练与证据

预训练、SFT、DPO、算术 GRPO 和多教师 MOPD 已有入口。双卡短流程完成预训练/SFT/DPO 各两步；持续训练另行登记。

独立原始代码对照覆盖 AttnRes、MLA 和 LatentMoE 前向，主干通过因果性、梯度与双 rank 参数一致检查；没有整模型官方训练数值对照。

本版尚未接入 MoonViT-V2、多模态联合训练、MTP、MXFP4/MXFP8 QAT 或模型级增量缓存。Demo 只处理文本并重算完整前缀；MOPD 仅有功能测试，没有已训练的领域教师。
