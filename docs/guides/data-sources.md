# 数据来源与处理

本页说明截至 **2026-09-10** 已准备或实际用过的数据，以及仍需完成的工作。不同实验使用的数据不同，不能把三个来源模型的语料直接视为 MiniFrontier1.0 的正式训练集。

本轮正式预训练的数据构造见[主计划](../pretraining-plan.md#data)和[执行记录](../experiments/2026-09-10-pretraining-cutover/execution.md)。首批文本候选按参考 tokenizer 设中文 225M、英文 150M、数学 50M、对话 40M 目标，另需补足可核验来源的代码域；共享视觉候选正在构造。这些候选尚未完成正式准入。下文保留既往实验的数据说明，不能用诊断集代替本轮库存。

补充代码候选已接入固定版本的 [CodeParrot train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train)，目标 100M 参考 token。筛选同时检查逐文件许可字段、文件开头的明确声明和 Python 3 语法，保留 repo/path、内容和许可头校验值；原仓库 commit 的缺项单列。源码不执行，也不使用普通文本的 NFKC/空白归一化。它与原许可未核验的 Python-Edu 行分别管理；来源筛选通过不等于正式数据准入，具体限制见执行记录。

## MiniFrontier1.0

离线示例和已完成的小配置学习实验使用程序生成的数据：整数算术、纯色色块图片，以及带时间戳的色块视频帧。训练部分包含 32 条算术、32 条图片和 8 条视频记录，另设独立的验证、测试和演示分组。生成方法见 [data/minifrontier1.py](../../minifrontier/data/minifrontier1.py)，实际结果见[学习实验](../experiments/mf1-reference-v2/README.md)。

2026-09-09 启动的完整 228M GPU 机制实验增加了已有的 Fineweb-Edu-Chinese-V2.1、FineWeb-Edu 和本项目生成数学，各抽取 2,000 个训练文档。继承原语料分组，仅用训练部分训练独立的 32K 候选 BPE，按字符边界分块并删除 38 条新增重复片段。加上各 128 条新生成图像、视频记录，训练集共 12,659 条、约 5.21M 可用 CE；验证集 232 条，未读取封存测试样本。每组实际训练预算为 500K CE，领域采样会重复暴露生成媒体，不能把暴露次数当成独立媒体数量。生成媒体与文本保留独立来源、处理记录和校验值，详见 [manifest](../experiments/mf1-gpu-mechanism-v1/data-manifest.json) 和[构造脚本](../../scripts/prepare_mf1_mechanism_data.py)。

这些数据用于检查模型能否学习、视觉输入是否参与预测，以及训练恢复是否正确。32K 候选尚未完成方案要求的词表对照，机制数据也未完成正式来源及质量验收；[正式数据计划](../../configs/minifrontier1/data_manifest.json)中的文本、图像和视频数量仍是目标值，来源名单尚待填写。

2026-09-10 的语言 SFT 诊断另用 **32 条本项目编写的基础问答、32 张生成图片和 8 段生成视频**，不沿用此前算术训练集作为语言问答训练集。留出集包含 106 条算术／复制任务和 32 条媒体记录，共 138 条；它们按题目与媒体身份分组。词表沿用上述 32K 候选，不重新训练。另已准备 886 条训练记录的扩展指令集，但截至最新公开快照尚未开训。数据清单、构造过程和启用条件见[语言诊断档案](../experiments/mf1-language-performance-v1/README.md)与[07:51 UTC 更新](../experiments/2026-09-10-mf1-update/README.md)。

## 三个来源模型的实验数据

| 来源 | 在本项目中的用途 | 数据使用条款与核查状态 |
|---|---|---|
| [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset) | 早期文本预训练、监督微调和偏好训练；结果保留在失败复盘中 | 数据卡列出 Apache-2.0 和 CC-BY-NC-2.0；按实际使用文件核对来源与条款 |
| [Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | 中文文本配方试验，使用 `4_5` 子集 | 数据卡标注 Apache-2.0，保留原始子来源信息 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 英文文本配方试验，抽取 `sample/10BT` 中的数据 | 数据卡标注 ODC-BY；原网页内容的权利仍需按来源处理 |
| [SmolLM-Corpus / Python-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md) | Python 代码配方试验 | 上游要求参照 The Stack v2 的数据许可。本地样本索引没有逐仓库许可证，已记录此缺口，尚未作为正式代码训练集通过审核 |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 对话数据准备，使用 `train_sft` 分片 | 数据卡标注 MIT |
| [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) 中的 [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) | Kimi/Qwen 小规模图文试验，共 96 张独立图片 | ALLaVA 数据卡标注 CC-BY-NC-4.0；本项目记录为试验数据，正式来源和划分审核尚未完成 |
| 本项目生成的算术与简单代数题 | 学习诊断和配方试验中的数学部分 | 生成器和随机种子随代码提供；任务范围限于所实现的规则 |

这里的许可证名称用于定位上游条款；训练数据按各自来源管理，项目代码的 Apache-2.0 许可不替代数据许可。

## 版本与处理方法

实验固定了以下来源版本，便于查找当时使用的文件。公开数据集后续更新不会自动改变已有实验。

| 来源 | 固定版本 |
|---|---|
| MiniMind | `312afb4f76391145c6902f765bb51691c09a12f5` |
| Fineweb-Edu-Chinese-V2.1 | `a5b574efa48beb3a8f6887ef0b093becf004328b` |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| SmolLM-Corpus | `3ba9d605774198c5868892d7a8deda78031a781f` |
| UltraChat 200k | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| FineVision | `3c380a731a3429c1d04693d6ec16d7e683def84c` |

早期 MiniMind 实验使用文件前缀采样，发现重复问法和覆盖不足的问题，详见[失败复盘](../training-failure-v1.md)。后续公开文本试验会按种子打乱文件和行组读取顺序，执行格式检查、去重、关联样本分组及训练/验证/测试划分。Python-Edu 通过上游索引获取代码，并核对内容校验值。

图文试验会核对媒体文件和解码后的图片校验值，按图片分组，避免同一图片的相关问答分入不同集合。96 张图片足以进行工程试验，尚不足以评估通用视觉能力。

实现入口为 [public_sources.py](../../minifrontier/data/public_sources.py)、[corpus.py](../../minifrontier/data/corpus.py)、[visual_sources.py](../../minifrontier/data/visual_sources.py) 和 [recipe.py](../../minifrontier/data/recipe.py)。每次构造记录来源版本、采样种子、接受/拒绝数量、划分和文件校验值。

## 如何使用这些材料

首次使用建议运行[离线最小示例](quickstart.md)或 [MiniFrontier1.0 示例](minifrontier1.md)，它们会自行生成所需数据。使用公开语料时，先阅读来源说明，再通过准备脚本构造本地数据；根目录 `data/` 不包含在 Git 仓库中。

小规模试验的来源记录、去重和划分不等于完成了正式数据审核。更大规模训练仍需补齐来源核查、近重复与评测污染检查、分层人工抽查，以及独立验证集。实验快照和现有复现范围见[实验档案](../experiments.md)。
