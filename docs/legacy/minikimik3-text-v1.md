> 2026-09-07 文本版本档案。下文容量、接口、训练状态和未实现项仅适用于该版本。当前状态见[模型页](../models/minikimik3.md)。

# MiniKimi-K3

基于 [moonshotai/Kimi-K3 固定 revision c5d1dd4](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721) 的缩小文本训练实现。原始代码与许可位于[源码快照](../../third_party/upstream/kimi-k3-c5d1dd4)。

## 当时容量与结构

12 层、hidden 512、词表 65,536；9 层 KDA 与 3 层 gated MLA。MLA 保留官方 NoPE 行为与输出 sigmoid 门；不会因为字段叫 `qk_rope_head_dim` 就擅自加入 RoPE。

32 个路由专家、Top-2、2 个共享专家；路由潜在宽度 256，专家中间宽度 256；保留 SiTU(beta=4, linear_beta=25)、latent normalization 和投影。第一层是 dense MLP，AttnRes 每 4 层一个块。共 **166,109,480** 个浮点参数，不含视觉和 MTP。

## 源码适配

[提取脚本](../../scripts/extract_training_sources.py)保留 KDA/MLA、专家、门控、decoder 与 AttnRes 的结构，移除 MoE 的推理模式限制，将专家分发改为可反传的 `index_add`。不把 Qwen/GDN 当作 KDA 替代。

CUDA KDA 使用官方依赖 FLA 0.5.2；旧 `transpose_state_layout` 适配为当前 `state_v_first`。CPU 使用逐 token delta recurrence 作为教学参考。测试比较 CUDA 与参考递推的前向及各输入/门控梯度。

线性层与 embedding 使用源码 Normal(0, .02)；A_log=0 来自 [K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) §2.1。发布推理代码没有初始化的 dt_bias 使用本地 log-uniform dt [.001,.1] / inverse-softplus 初始化；路由修正偏置和 AttnRes 查询置零属于明确本地选择。第一层没有前序残差块，对应的未使用 AttnRes 参数冻结。

路由修正偏置按跨 rank 的专家计数更新；更新率 .001 是本地配方。默认优化器为 AdamW，不能声称完整复现官方优化栈。

## 训练与证据

预训练、SFT、DPO、可验证算术 GRPO 和多教师 MOPD 均有可执行入口。真实双卡短流程完成预训练/SFT/DPO，每阶段两步；完整持续训练单独登记。MOPD 的功能测试不代表已经拥有官方领域教师。

独立原始代码 oracle 覆盖 AttnRes、MLA 与 LatentMoE 前向；新增主干通过因果性、训练梯度和双 rank 参数一致验证。**没有声明与官方训练整模型数值完全等价。**

原生 MoonViT-V2、多模态从零联合训练、MTP、MXFP4/MXFP8 QAT 和模型级增量缓存尚未接入。当前 demo 为文本，采用完整前缀重算。

派生源码保留 Kimi K3 自定义许可，见[第三方说明](../../THIRD_PARTY_NOTICES.md)。
