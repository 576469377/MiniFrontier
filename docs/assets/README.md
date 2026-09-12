# MiniFrontier 视觉素材

这里保存项目封面、图标和 MF1 架构图。阅读模型结构请从[MF1 模型页](../models/minifrontier1.md)进入；查找原始报告图示见[上游素材](upstream/README.md)。

[返回文档导航](../README.md) · [项目首页](../../README.md)

## MF1 架构图

| 图示 | 适合查看 |
|---|---|
| [minifrontier-architecture.svg](minifrontier-architecture.svg) | MF1 模块概览；参数与层序以[模型说明](../models/minifrontier1.md)为准 |
| [minifrontier1-detail.svg](minifrontier1-detail.svg) | 首页与 MF1 模型页的完整结构图：输入汇合、16 层顺序、GR 子层读写、lookup 与 MTP；按 228M 配置绘制 |
| [minifrontier1-decoder.svg](minifrontier1-decoder.svg) | MF1 单层展开：GR 读写、残差旁路、潜在空间路由专家和全宽共享分支 |
| [minifrontier1-attention.svg](minifrontier1-attention.svg) | MF1 三种历史读取方式：KDA 递推、CSA 压缩 KV、QSA-MLA 原始 token 检索及缓存差异 |

SVG 可直接下载并缩放。三个来源模型页同时展示官方报告图和表示本项目配置的 Mermaid 图；MF1 的 MTP 数据流也使用 Mermaid。

## 封面与图标

| 文件 | 用途 |
|---|---|
| [minifrontier-cover.png](minifrontier-cover.png) | README 封面，2172 × 724，浅色不透明背景，可用于深浅两种页面主题 |
| [minifrontier-mark.svg](minifrontier-mark.svg) | 可缩放的独立图标，适合头像、文档和小尺寸入口 |
| [icon-source.svg](icon-source.svg) · [icon-experiment.svg](icon-experiment.svg) · [icon-multimodal.svg](icon-multimodal.svg) | 源码、实验与多模态的配套图标；保留供文档使用 |

## 修改与引用

架构图保留 SVG 源文件。修改时按 [228M 配置](../../configs/minifrontier1/model_228m_native.json)、[Decoder](../../minifrontier/models/minifrontier1/modeling.py)、[视觉编码器](../../minifrontier/models/minifrontier1/vision.py)、[LatentMoE](../../minifrontier/models/minifrontier1/moe.py) 与 [MTP](../../minifrontier/models/minifrontier1/mtp.py) 核对层序、维度和分支，并同步图注。图使用不透明背景与可缩放文字；引用时提供文字替代说明。

封面由 imagegen 生成并调整背景；SVG 在本仓库绘制。这些原创素材沿用仓库原创内容的 [Apache-2.0](../../LICENSE) 声明。`upstream/` 下的报告图示保留原作者权利，采用独立的[来源与权利说明](upstream/README.md)，不纳入上述原创素材许可。
