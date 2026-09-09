# MiniKimi-K3

v0.1.0 研究预览实现。固定来源与逐组件许可见[第三方说明](../../THIRD_PARTY_NOTICES.md)，[历史文本版本](../legacy/minikimik3-text-v1.md)只保留作演进记录。

## 当前研究配置

[configs/strategies/minikimik3-v2.json](../../configs/strategies/minikimik3-v2.json)：12 层、hidden 512、64K 词表，9 KDA + 3 gated MLA；32 路由专家、Top-2、2 共享专家，latent width 256；AttnRes、MoonViT-V2 与 MTP。

当前配置浮点参数 **204,526,216**；Kimi/Qwen 包含视觉与 MTP，DeepSeek 此数为 Text-v2 与 MTP，后接 Vision-v1 需重新计量。根目录文本兼容配置和[离线极小示例](../quickstart.md)的参数量不同。默认单卡独立训练；长上下文与新模态阶段必须重新测显存和吞吐。

## 实现、验证与训练状态

| 状态 | 范围 |
|---|---|
| 已实现 | 文本主干、原生视觉适配、MTP、对应 Muon/路由更新、增量缓存；QAT 仿真和后训练/草稿入口 |
| 已验证 | KDA CPU 参考递推与 CUDA FLA 的前向/梯度，AttnRes、MLA、LatentMoE 原始源码对照，原生视觉/MTP、路由、Muon、增量缓存及短训练/恢复。MLA 保留 NoPE 和 sigmoid 输出门。 |
| 已训练 | 可学习性诊断与 20M-token 配方试验，详见[带时间边界的实验档案](../experiments.md) |
| 待完成 | 完整主预算、正式数据准入、配方与教师资格、独立语言/视觉能力、量化部署与草稿加速验收 |

联合 PT → SFT/QAT → 9 教师 → sampled-token MOPD → Kimi 草稿 → 能力验收。

公开源码没有披露的初始化、LR、loss 聚合和容量比例属于显式 mini 适配，不声称完整复现官方训练栈。PyTorch 参考后端的长上下文内存与速度不代表官方融合内核表现。

## 使用与边界

[最小示例](../quickstart.md)覆盖离线数据、PT、暂停恢复、SFT、验证和 CLI 生成；[通用训练指南](../training.md)说明策略门槛和阶段迁移。[原生视觉/MTP与方案审计](../audits/strategy-implementation-v2.md)、[后训练适应](../posttraining-adaptation.md)、[草稿适应](../draft-adaptation.md)记录更细的实现与测试范围。

当前没有通过能力验收的可用聊天权重。训练集算术记忆、loss 下降、工程测试成功均不能代表泛化或对话能力。旧 educational-v1 失败见[复盘](../training-failure-v1.md)。

Kimi 派生组件保留 [Kimi K3 自定义许可](../../LICENSES/LicenseRef-Kimi-K3.txt)。
