# 数据来源与处理

本页说明截至 **2026-09-09** 实际用过的数据，以及仍需完成的工作。不同实验使用的数据不同，不能把三个来源模型的语料直接视为 MiniFrontier1.0 的正式训练集。

## MiniFrontier1.0

当前离线示例和小规模学习实验使用程序生成的数据：整数算术、纯色色块图片，以及带时间戳的色块视频帧。训练部分包含 32 条算术、32 条图片和 8 条视频记录，另设独立的验证、测试和演示分组。生成方法见 [data/minifrontier1.py](../../minifrontier/data/minifrontier1.py)，实际结果见[学习实验](../experiments/mf1-reference-v2/README.md)。

这些数据用于检查模型能否学习、视觉输入是否参与预测，以及训练恢复是否正确。完整模型的正式数据尚未准备完成；[数据计划](../../configs/minifrontier1/data_manifest.json)中的文本、图像和视频数量是目标值，来源名单仍为空。

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
