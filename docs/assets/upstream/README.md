# 上游技术报告结构图

这些图片是官方技术报告中的局部图示摘录，用于模型架构讲解与本仓库实现的对照。仅保留对应结构图，未收录整份报告；图内标签和结构未改写。作者与权利归属保留在上游，不因放入本仓库而改为 Apache-2.0。源码组件许可与报告图示的权利范围应分别理解，见[第三方说明](../../../THIRD_PARTY_NOTICES.md)。

| 本地图片 | 作者与报告 | 图号 / PDF 页码 | 使用位置 |
|---|---|---|---|
| [kimi-k3-architecture.png](kimi-k3-architecture.png) | Moonshot AI / Kimi Team，*Kimi K3: Open Frontier Intelligence*，[官方报告](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf#page=3) | Figure 2 / 第 3 页 | [MiniKimi-K3](../../models/minikimik3.md#官方报告结构图) |
| [qwen3.8-next-architecture.png](qwen3.8-next-architecture.png) | Qwen Team，*On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*，[官方报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf#page=2) | Figure 1 / 第 2 页 | [MiniQwen4](../../models/miniqwen4.md#官方报告结构图) |
| [deepseek-v4-architecture.png](deepseek-v4-architecture.png) | DeepSeek-AI，*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，[官方报告 v1](https://arxiv.org/pdf/2606.19348v1#page=6) | Figure 2 / 第 6 页 | [MiniDeepSeek-V4](../../models/minideepseekv4.md#官方报告结构图) |

上游图解释原始架构，不代表本项目的层数、维度、权重或能力。每个模型页在原图之后给出本项目配置的 Mermaid 图及逐模块说明；MF1 的融合结构使用[本仓库单独绘制的 SVG](../README.md)。

图示引用保留原作者权利；此处不声明图示获得了与项目代码相同的再许可，也不将源码仓库的 Apache-2.0 或 MIT 标识扩展到独立报告。Kimi 官方仓库另有 [Kimi K3 许可](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/LICENSE)；Qwen 报告与其 Transformers 源码是不同产物；DeepSeek 报告的发布条款见 [arXiv 记录](https://arxiv.org/abs/2606.19348v1)。

提取记录见 [manifest.json](manifest.json)，包括报告版本、下载文件 SHA256、裁剪坐标和输出图片 SHA256。坐标基于 `pdftoppm -scale-to 3300` 的整页栅格，以左上角为原点，单位为像素；页码从 1 开始。提取日期为 2026-09-10。
