# MiniFrontier 视觉素材

这里保存项目封面、图标和按本地配置绘制的架构图。模型结构见 [MF1.1](../models/minifrontier11.md)、[MF1.0](../models/minifrontier1.md) 与 [MiniDeepSeek-V4.1](../models/minideepseekv41.md)；原始报告图示见[上游素材](upstream/README.md)。

[返回文档导航](../README.md) · [项目首页](../../README.md)

## 模型架构图

| 图示 | 适合查看 |
|---|---|
| [minifrontier11-architecture.svg](minifrontier11-architecture.svg) | MF1.1 的 211M 结构：统一图文输入、16 层注意力、Single-Pass mHC、LatentMoE 与 lookup；无 MTP |
| [minideepseekv41-architecture.svg](minideepseekv41-architecture.svg) | V4.1 的 242M 文本配置：6+6 CED、CSA2 共享与读取、Single-Pass mHC、Engram |
| [minifrontier-architecture.svg](minifrontier-architecture.svg) | MF1.0 模块概览；参数与层序以[模型说明](../models/minifrontier1.md)为准 |
| [minifrontier1-detail.svg](minifrontier1-detail.svg) | MF1.0 的 228M 完整结构：输入汇合、16 层顺序、GR 子层读写、lookup 与 MTP |
| [minifrontier1-decoder.svg](minifrontier1-decoder.svg) | MF1.0 单层展开：GR 读写、残差旁路、潜在空间路由专家和全宽共享分支 |
| [minifrontier1-attention.svg](minifrontier1-attention.svg) | MF1.0 三种历史读取方式：KDA 递推、CSA 压缩 KV、QSA-MLA 原始 token 检索及缓存差异 |

SVG 可直接下载并缩放。Kimi、Qwen、DeepSeek-V4 模型页以 Mermaid 展示本地配置，并附官方报告图作对照；MF1.0 的 MTP 数据流也有单独的 Mermaid 图。V4.1 图按文本配置绘制。

## 封面与图标

[首页实验图](minifrontier-cover.png)并排显示了一对 224 × 224 的合成留出图片。MiniDeepSeek-V4.1 新数据分支在第 8,665 步对两图均回答「红色」。左图参考「黄色」，右图参考「红色」。两张图片在评测面板中的样本 ID 以 `cbf20f87`、`87762691` 开头。归一化计分、样本量与换图结果见[学习路线](../guides/learning-path.md#看图时模型用了什么)。

[项目标识](minifrontier-mark.svg)可用于引用。项目原创图示随仓库[许可证](../../LICENSE)发布，转载的报告图见[上游素材与来源](upstream/README.md)。
