# 从一条样本追到一次回答

从第一个 CPU 实验开始，依次看输入、训练目标、检查点和模型的回答。全程用同一个问题和同一个 tokenizer，容易找到两次运行真正不同的地方。

[项目首页](../../README.md) · [操作指南](README.md) · [按模型查看原版视觉配方](visual-recipes.md)

## 先得到一个能加载的检查点

运行[首页的 CPU 示例](../../README.md#快速开始)。打开 `outputs/mf1-quickstart/report.json` 的 `results`，找到第 4 步暂停、第 8 步恢复训练完成的记录。沿着 `checkpoint` 找到 SFT 权重，并用首页的生成命令读入同一文件。输出中的 `termination` 标记实际停止原因；如果第一个 token 就是结束符，`text` 会是空字符串。

| 你想知道什么 | 打开哪个产物 |
|---|---|
| 输入问题、目标答案和媒体在哪里 | `data/train.jsonl`、`data/val.jsonl` 与 `data/media/` |
| 训练用了哪个词表和配置 | `sft/tokenizer.json`、`sft/resolved_config.json` |
| 本次训练实际更新了几次 | `sft/status.json` 和 `sft/metrics.jsonl` |
| 检查点在验证集上的指标怎样 | `sft/evaluation.json` |

这些路径都相对 `outputs/mf1-quickstart/`。这个微型权重适合追踪数据流；要研究训练预算和能力的关系，可以接着看同架构[固定留出集实验](../experiments/mf1-reference-v2/README.md)。

## 训练时究竟要预测哪些 token

先拿最小对话举例。用户问「2 加 2 等于几？」，助手答「4」。预训练可以把整段文档作为续写目标。指令微调通常只对助手回答及回合结束符计算损失。两者可以使用同一段文字，却产生不同的梯度。

在来源模型的[文本编码](../../minifrontier/data/text.py)中找到 `chat_tokens`，与[生成入口](../../minifrontier/inference/runtime.py)中构建提示词的位置对读。MF1 有独立的[数据编码](../../minifrontier/data/minifrontier1_encoding.py)和[推理实现](../../minifrontier/inference/minifrontier1.py)。用同一家族的两条路径做下面的比较。

1. 给训练编码器一轮完整问答，记录输入 ID、标签和第一个有效助手标签的位置。检查问题和填充位的标签是否被忽略，回答末尾是否保留结束符。
2. 仅给推理入口同一个问题，核对生成第一个回答 token 前收到的 ID。它们应逐个等于完整样本在助手答案之前的前缀，包括角色标记。图文输入还要核对图像占位符的位置、数量和图片顺序。
3. 加上第二轮用户提问，重做检查。上一轮助手答完时，也应有明确的结束边界。记录超过长度预算的样本如何处理，避免用残缺回答训练提前停句。

选择一个断点，先固定提示、解码方式和输出上限，再改变一个条件。记录原始回答、是否主动结束及是否重复。训练损失回答「给定正确前文，下一个 token 有多可预测」；自由生成可以观察模型读到自己生成的文字后会走向哪里。

## 用文本底座训练双语助手

下面可复现 MiniMind 与 V4.1 的同数据对照。准备 [固定版本语料](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5) 的 `sft_t2t_mini.jsonl`、它的数据集 revision，以及父权重目录里的 `tokenizer.json`。同一份会话可按两个模型各自的分词器和聊天模板训练。

```bash
uv run python -m scripts.prepare_chat_sft \
  --source "$SFT_JSONL" --revision "$DATASET_REVISION" \
  --tokenizer "$PARENT_DIR/tokenizer.json" --output outputs/chat-data
```

打开 `outputs/chat-data/chat-audit.json`。每个助手回合都对照实际的生成前缀，只有最终回答标记、正文和该回合的 EOS 参与监督。完整回答超出 384 token 的样本会计数并跳过。已有固定评测题时，用 `--exclude-prompts` 传入问题字符串的 JSON 数组。

```bash
updates=$(python -c 'import json; n=json.load(open("outputs/chat-data/chat-audit.json"))["splits"]["train"]["examples"]; print((n+7)//8)')
uv run minifrontier train --device cuda --model minideepseekv41 --stage sft \
  --init "$PARENT_DIR/model.pt" --data outputs/chat-data/encoded \
  --output outputs/chat-sft --steps "$updates" --sequence-length 384 \
  --batch-size 8 --grad-accum 1 --optimizer adamw --lr 0.00005 \
  --weight-decay 0.01 --warmup-steps 51 --seed 20261109
```

这里按一遍训练集向上凑齐最后一批。用相同留出题分别测试父权重和新权重，同时记下短答准确率、模型自行结束的比例和重复片段。这样能区分回答越来越短、学会按格式回答以及回答内容变正确三种变化。

实际抽出的 14,723 个训练答案中，仅 181 个不超过 4 个正文 token。按用户文本识别为英文的 7,525 个回答中位长度为 144 token，中文的 7,198 个中位长度为 43 token。只按字符长度选择候选，会高估这份语料对一词回答的覆盖。

## 看图时模型用了什么

封面的两张合成图来自 MiniDeepSeek-V4.1 新数据分支在第 8,665 步的真实生成。问题完全相同，标注分别为「黄色」和「红色」，模型均生成「红色」。这个例子展示了一种成对检验。

1. 从留出集中选一对问题相同、答案不同的图片。固定权重、提示和解码，让模型自己答题。
2. 对照正确图计算准确率，然后把同一问题交给配对图，仍对原图参考答案评分。正确图片上的优势反映模型在这组题上是否使用了图像信息。
3. OCR 同时记录整行完全正确数和字符错误率。字符错误率是插入、删除、替换的最少总次数除以参考总字符数，模型多写时可以超过 100%。

2026 年 11 月的这次实验有 180 道中英文短答（颜色、计数、位置各 60 道）、80 道中英文逐行 OCR，另有 16 道整段 OCR。两支均从第 8,250 步的模型、优化器和采样进度出发。

| 第 8,665 步 | 旧数据 | 换入可核验短答与逐行 OCR |
|---|---:|---:|
| 本窗口监督 token | 7,358,169 | 7,361,599 |
| 本窗口图像曝光 | 4,415 | 4,416 |
| 短答，正确图 | 0/180 | 14/180 |
| 短答，换图后仍按原答案评分 | 0/180 | 18/180 |
| 逐行 OCR 字符错误率，964 个参考字符 | 645.95% | 153.84% |
| 逐行 OCR 完全正确 | 0/80 | 0/80 |

短答按规范化后的完整答案匹配，OCR 按字符编辑距离计分，两者均从自由生成取得。旧数据分支最后训练到第 14,034 步，累计 250,007,334 个监督 token、150,000 次图像曝光；同一面板的短答仍为 0/180，逐行 OCR 为 7,696 次编辑 / 964 字符，80 行均未完全匹配。这里测的是指定提示下的合成场景与渲染文档。

### 四色对照实验

另一次教学实验固定问题，只改变图形颜色。生成器把红、黄、蓝、绿在训练与测试中等量采样，测试用 96 张留出图。19.6M 参数的紧凑 V4.1 在种子 42 和 73 上，正确图都为 96/96，换成另一种颜色后也都 96/96 跟随新图。当训练时就把图替换为统一的灰色，两次实验各得 24/96。正确图训练所得模型在测试时看灰图，也各得 24/96。

这一实验只检验颜色。逐行 OCR、数数以及换种问法的泛化需要各自的留出集。可用 [数据生成](../../scripts/prepare_v41_learning.py) 和 [换图评测](../../scripts/evaluate_v41_learning.py) 复查。

## 选一个问题继续读

| 下一步想研究什么 | 入口 |
|---|---|
| 比较六个模型的层序、视觉初始化与训练阶段 | [模型目录](../README.md#模型结构与阅读顺序) · [视觉配方](visual-recipes.md) |
| 接入自己的训练数据并划分训练与验证集 | [数据来源](data-sources.md) · [MF1 全流程](minifrontier1.md) |
| 复查真实实验中失效的设计和对照条件 | [实验记录](../experiments.md) · [生成诊断](../experiments/checkpoint-diagnostics-2026-09-14/README.md) |
