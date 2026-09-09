# MiniFrontier 视觉素材

| 文件 | 用途 |
|---|---|
| [minifrontier-cover.png](minifrontier-cover.png) | README 封面，2172 × 724，浅色不透明背景，可用于深浅两种页面主题 |
| [minifrontier-mark.svg](minifrontier-mark.svg) | 可缩放的独立图标，适合头像、文档和小尺寸入口 |
| [minifrontier-architecture.svg](minifrontier-architecture.svg) | MF1 模块概览；参数与层序以[模型说明](../models/minifrontier1.md)为准 |
| [icon-source.svg](icon-source.svg) · [icon-experiment.svg](icon-experiment.svg) · [icon-multimodal.svg](icon-multimodal.svg) | README 三个项目特色入口的配套图标 |

标识以折叠山峰和字母 M 表达项目名称。蓝色、暖金色与中性色贯穿封面、图标和架构图。架构图保留文本与路径源文件，更新模型时应同步检查；图片在 README 中提供文字替代说明。

封面由内置 **imagegen** 工具生成并进行一次背景调整；SVG 在本仓库绘制。未复制上游项目的标识或图片。此处素材沿用仓库原创内容的 [Apache-2.0](../../LICENSE) 声明。

README 编排参考了 [MiniMind](https://github.com/jingyaogong/minimind) 的品牌页头与入门路径、[SmolLM](https://github.com/huggingface/smollm) 的模型导航和 [nanochat](https://github.com/karpathy/nanochat) 的实践导向。项目介绍、视觉素材及实验表述按 MiniFrontier 的实际情况编写。

<details>
<summary>封面生成提示词</summary>

初始生成：

```text
Use case: logo-brand.
Asset type: polished wide GitHub README brand cover for the open source project MiniFrontier.
Create one finished horizontal cover, approximately 3:1 aspect ratio, with spacious editorial composition suitable for a technical research repository. The concept is several small architectural paths joining into a new frontier: a sculptural, folded, ribbon-like monogram evoking an M and a horizon, rendered with subtle depth and fine material texture, beside a beautifully typeset large wordmark.
The only text in the image must be exactly "MiniFrontier" (M i n i F r o n t i e r), very crisp and easy to read at thumbnail size. No tagline, version, numbers or tiny labels. The monogram and wordmark should feel like one coherent identity. Use a restrained contemporary research-lab aesthetic, rich ink typography on a very light neutral background, a few harmonious color accents within the sculptural mark, plenty of breathing room and meticulous alignment. Fill the horizontal composition well with a generous safe margin on every edge. Original artwork only; no existing company logos, no robots, brains, circuit-board clip art, stock illustrations, starbursts, fake charts, badges, interface mockups or watermark. This is the final cover graphic, not a photograph of a printed sign or a layout sheet.
```

背景调整（以上一步生成图为输入）：

```text
Use case: background-extraction (background replacement).
Image 1 is the edit target: the MiniFrontier brand cover. Preserve the existing sculptural folded M, its colors and texture, and the exact MiniFrontier wordmark with its existing typography. Change only the background and framing: this finished GitHub cover MUST have a solid, fully opaque, very light warm neutral background (#f7f8fa or close), including behind the wordmark. No transparent pixels anywhere, no alpha cutout. Ensure the entire mark, its shadow and the entire wordmark fit inside the canvas with a clean safe margin, keeping a 3:1 wide aspect ratio. Keep all the current artwork and no additional text. The light panel must remain readable when displayed on either a dark or a light webpage.
```

</details>
