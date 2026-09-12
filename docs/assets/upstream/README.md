# 上游技术报告结构图

本目录保存三份官方技术报告的结构图摘录，供模型页对照。查看本项目的层数、维度与适配说明，请进入表中的模型页；核对图片来源时查看报告和提取记录。

[返回素材导航](../README.md) · [返回文档导航](../../README.md)

## 报告与使用位置

| 本地图片 | 作者与报告 | 图号 / PDF 页码 | 使用位置 |
|---|---|---|---|
| [kimi-k3-architecture.png](kimi-k3-architecture.png) | Moonshot AI / Kimi Team，*Kimi K3: Open Frontier Intelligence*，[官方报告](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf#page=3) | Figure 2 / 第 3 页 | [MiniKimi-K3](../../models/minikimik3.md#官方报告结构图) |
| [qwen3.8-next-architecture.png](qwen3.8-next-architecture.png) | Qwen Team，*On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*，[官方报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf#page=2) | Figure 1 / 第 2 页 | [MiniQwen4](../../models/miniqwen4.md#官方报告结构图) |
| [deepseek-v4-architecture.png](deepseek-v4-architecture.png) | DeepSeek-AI，*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，[官方报告 v1](https://arxiv.org/pdf/2606.19348v1#page=6) | Figure 2 / 第 6 页 | [MiniDeepSeek-V4](../../models/minideepseekv4.md#官方报告结构图) |

图内标签和结构保持原样，未收录整份报告。每个模型页在原图之后给出本项目配置的 Mermaid 图及逐模块说明；MF1 使用[本仓库绘制的结构图](../README.md#mf1-架构图)。上游图中的规模、权重与能力不等同于本项目。

## 权利与引用

图片保留原作者权利，不适用 MiniFrontier 原创代码的 Apache-2.0 声明，见[第三方说明](../../../THIRD_PARTY_NOTICES.md)。

各报告的使用条款需单独核对：Kimi 官方仓库附有 [Kimi K3 许可](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/LICENSE)；Qwen 报告与其 Transformers 源码属于不同产物；DeepSeek 报告见 [arXiv 发布记录](https://arxiv.org/abs/2606.19348v1)。上述源码组件的许可不能代替报告图示的使用条款。

## 提取记录

[manifest.json](manifest.json) 记录报告版本、下载文件 SHA256、裁剪坐标和输出图片 SHA256。提取日期为 2026-09-10。

复核裁剪时，使用 `pdftoppm -scale-to 3300` 的整页栅格；坐标以左上角为原点、单位为像素，页码从 1 开始。
