# 数据来源与处理

本页记录正式预训练、历史实验与离线示例实际使用的数据。语料来自公开数据集或本项目生成，不使用旗舰模型的官方训练集；数据、tokenizer 和权重的许可独立于代码许可证。

## 首阶段正式预训练

四个模型的首阶段已绑定数据和冻结 tokenizer，共用同一文本划分。来源模型使用 64K 词表，MF1 使用 32K；视觉模型另行绑定图像组件。各阶段的数据供给与后续补充见[预训练计划](../pretraining-plan.md#data)。

首阶段冻结文本包含 **396,842 篇训练文档、3,737 篇验证文档、4,432 篇测试文档**。训练划分在 64K 词表下约为 **5.33 亿 CE token**，在 MF1 词表下约为 **5.70 亿**；两者是同一语料的不同编码库存，不能相加或视为训练进度。重复采样会消耗训练预算，但不会增加独立样本数。

| 来源 | 用途 | 本地来源记录中的条款与限制 |
|---|---|---|
| [Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | 中文教育文本，`4_5` 子集 | 数据卡标注 Apache-2.0；保留原始子来源 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 英文教育文本，`sample/10BT` | ODC-BY；原网页内容的权利单独保留 |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 数学网页文本 | ODC-BY、Common Crawl 条款及原网页权利；领域名不表示逐题正确性已核验 |
| [CodeParrot train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Python 代码 | 逐文件筛选 Apache-2.0、MIT、BSD 声明；部分原仓库 commit 信息缺失 |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 对话文本，`train_sft` 分片 | 数据卡标注 MIT |
| [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)，通过 FineVision 固定版本读取 | 自然图像描述与问答 | CC-BY-NC-4.0，并保留底层图片权利；后续权重发布需考虑非商业限制 |
| [CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K)，通过 FineVision 固定版本读取 | 文档与图表 | ODC-BY、生成内容条款及 Ai2 使用说明分别记录 |
| 本项目生成的 OCR | 图中文字识别 | 生成过程、文本来源与渲染资源随组件记录 |

读取与筛选规则见[文本](../../minifrontier/data/public_sources.py)、[数学与合并](../../minifrontier/data/pretraining.py)、[代码](../../minifrontier/data/code_sources.py)和[视觉](../../minifrontier/data/visual_sources.py)实现；组件构造记录见[执行档案](../experiments/2026-09-10-pretraining-cutover/execution.md)。

## 已做的处理

1. **固定来源。** 记录数据集版本、原始项目或网页、采样种子、读取位置和内容校验值，限制下载与本地存储规模。
2. **格式与质量筛选。** 检查文本长度、编码、字段和来源评分；代码单独检查许可列、文件头声明和 Python 3 语法，不执行源码，也不做破坏缩进的文本归一化。
3. **去重与分组。** 文本进行精确和近重复检查；同源文档、同图问答及 OCR 派生记录按关联组管理，避免跨训练／验证／测试泄漏。
4. **评测排除。** 对固定版本的 HumanEval、MBPP、GSM8K 和已登记视觉评测内容执行重叠检查；发现与 TextVQA 评测图片重合的训练组后已排除。匹配规则不保证发现所有改写、翻译或语义等价内容。
5. **冻结与编码。** tokenizer 仅用训练划分构建，编码记录其 SHA256、处理器配置及各领域 CE 计数。视觉模型消费原始像素，视觉编码器参与训练，不用离线视觉特征代替。
6. **校验训练入口。** 检查组件 hash、监督掩码、样本边界和固定验证库存。每个模型按自己的监督规则计数，媒体占位、padding 和辅助 MTP 不计入主 CE 预算。

**系统性的逐来源人工质量复核尚未完成。** 首阶段按主计划记录的条件开展研究训练，机械检查不替代人工复核，也不构成数据或权重的发布验收。后续变更及已知限制保存在各组件清单中。

## 固定版本

读取器固定以下版本；上游更新不会改变已有实验。每次构造的 manifest 另行记录实际样本库存。

| 来源 | 版本 |
|---|---|
| Fineweb-Edu-Chinese-V2.1 | `a5b574efa48beb3a8f6887ef0b093becf004328b` |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| OpenWebMath | `fde8ef8de2300f5e778f56261843dab89f230815` |
| CodeParrot train | `3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4` |
| UltraChat 200k | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| FineVision | `3c380a731a3429c1d04693d6ec16d7e683def84c` |
| ALLaVA-4V 来源 | `0fd42fce5c047d387a4bb5318d588eae9a9797f0` |
| CoSyn-400K 来源 | `86e46e1fd5e754d056169f0fb38f06c6997ff7de` |

文本与 tokenizer 校验值见[训练执行记录](../audits/training-infrastructure.json)。

## 历史实验与离线示例

| 数据 | 用途与记录 |
|---|---|
| 本项目生成的算术、色块图像和视频帧 | 离线示例验证训练、恢复和媒体输入；MF1 默认生成 32 条算术、32 条图像和 8 条视频训练记录。[小配置学习实验](../experiments/mf1-reference-v2/README.md)单独记录其留出结果。 |
| 中英文教育文本与生成媒体的小切片 | MF1 两组各 500K CE 的 228M 机制实验；[manifest](../experiments/mf1-gpu-mechanism-v1/data-manifest.json)保存抽样与划分。后续基础问答诊断见[语言实验档案](../experiments/mf1-language-performance-v1/README.md)。 |
| [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset)，`312afb4f76391145c6902f765bb51691c09a12f5` | 早期文本预训练、SFT 和偏好训练，结果见[失败复盘](../training-failure-v1.md)。数据卡列出 Apache-2.0 和 CC-BY-NC-2.0，需按文件核对条款。 |
| [SmolLM-Corpus / Python-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md) | 早期代码配方试验。上游要求参照 The Stack v2；本地索引缺逐仓库许可，记录为 `original-license-unresolved`，未用于本轮正式代码训练。 |
| 96 张 ALLaVA 图片 | 早期 Kimi/Qwen 图文流程诊断，不代表首阶段正式视觉库存。 |

## 在本地构造数据

离线试用可运行[来源模型示例](quickstart.md)或 [MF1 示例](minifrontier1.md)，由命令生成数据。准备公开语料需要网络和 `data` extra；下面演示固定评测集准备与已有候选的合并：

```bash
uv sync --locked --extra data
uv run python -m minifrontier.data.evaluation --output data/base-evaluation-v1
uv run python -m minifrontier.data.pretraining merge \
  --inputs data/text-candidate-v1 data/code-candidate-v1 \
  --evaluation data/base-evaluation-v1 --output data/base-candidate-v1 \
  --max-gib 12
```

`--inputs` 需要预先构造、使用同一参考 tokenizer 的候选目录，每次输出到新目录。合并保留已有留出组，同源组连接时 test 优先于 val；得到候选库存后，再绑定正式 tokenizer 和阶段编码。`data/` 不随 Git 仓库或安装包分发。
