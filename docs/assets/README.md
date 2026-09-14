# MiniFrontier 视觉素材

这里保存项目封面、图标和按本地配置绘制的架构图。模型结构见 [MF1.1](../models/minifrontier11.md)、[MF1.0](../models/minifrontier1.md) 与 [MiniDeepSeek-V4.1](../models/minideepseekv41.md)；原始报告图示见[上游素材](upstream/README.md)。

[返回文档导航](../README.md) · [项目首页](../../README.md)

## 模型架构图

| 图示 | 适合查看 |
|---|---|
| [minifrontier11-architecture.svg](minifrontier11-architecture.svg) | MF1.1 的 211M 结构：统一图文输入、16 层注意力、Single-Pass mHC、LatentMoE 与 lookup；无 MTP |
| [minideepseekv41-architecture.svg](minideepseekv41-architecture.svg) | V4.1 的 242M 文本配置：6+6 CED、CSA2 共享与读取、Single-Pass mHC、Engram |
| [minifrontier-architecture.svg](minifrontier-architecture.svg) | MF1 模块概览；参数与层序以[模型说明](../models/minifrontier1.md)为准 |
| [minifrontier1-detail.svg](minifrontier1-detail.svg) | 首页与 MF1 模型页的完整结构图：输入汇合、16 层顺序、GR 子层读写、lookup 与 MTP；按 228M 配置绘制 |
| [minifrontier1-decoder.svg](minifrontier1-decoder.svg) | MF1 单层展开：GR 读写、残差旁路、潜在空间路由专家和全宽共享分支 |
| [minifrontier1-attention.svg](minifrontier1-attention.svg) | MF1 三种历史读取方式：KDA 递推、CSA 压缩 KV、QSA-MLA 原始 token 检索及缓存差异 |

SVG 可直接下载并缩放。Kimi、Qwen、DeepSeek-V4 模型页同时展示官方报告图和本地配置图；MF1.0 的 MTP 数据流另有 Mermaid 图。V4.1 的本地结构图与其来源说明均在对应模型页。

## 封面与图标

| 文件 | 用途 |
|---|---|
| [minifrontier-cover.png](minifrontier-cover.png) | README 封面，2172 × 724，浅色不透明背景，可用于深浅两种页面主题 |
| [minifrontier-mark.svg](minifrontier-mark.svg) | 可缩放的独立图标，适合头像、文档和小尺寸入口 |
| [icon-source.svg](icon-source.svg) · [icon-experiment.svg](icon-experiment.svg) · [icon-multimodal.svg](icon-multimodal.svg) | 源码、实验与多模态的配套图标；保留供文档使用 |

## 修改与引用

架构图保留 SVG 源文件。修改时按 [228M 配置](../../configs/minifrontier1/model_228m_native.json)、[Decoder](../../minifrontier/models/minifrontier1/modeling.py)、[视觉编码器](../../minifrontier/models/minifrontier1/vision.py)、[LatentMoE](../../minifrontier/models/minifrontier1/moe.py) 与 [MTP](../../minifrontier/models/minifrontier1/mtp.py) 核对层序、维度和分支，并同步图注。图使用不透明背景与可缩放文字；引用时提供文字替代说明。

新版结构图分别以 [MF1.1 配置](../../configs/minifrontier11.json)、[V4.1 配置](../../configs/minideepseekv41.json)和各模型页列出的实现为准。旧版图保留对应版本名称，不将新模块画入旧结构。

封面由 imagegen 生成并调整背景；SVG 在本仓库绘制。这些原创素材沿用仓库原创内容的 [Apache-2.0](../../LICENSE) 声明。`upstream/` 下的报告图示保留原作者权利，采用独立的[来源与权利说明](upstream/README.md)，不纳入上述原创素材许可。
