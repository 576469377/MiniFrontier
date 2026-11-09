# MiniDeepSeek-V4.1 的视觉塔怎样接到语言主干

## 原报告

DeepSeek-V4.1-Flash 有原生视觉编码器。官方先用约 470 亿 alt-text 图文对训练 DeepSeek-ViT，使用 SigLIP 的逐对 sigmoid 对比损失，保持图像宽高比，最大 224 × 224。随后临时接入 4B MoE 解码器，在 236B tokens 的图像描述、alt text、图表、OCR 上做下一词预测，此时图像尺度为 544–1344。完成后丢弃临时语言模型，只保留视觉编码器。正式主干从开头便接受图文混合预训练。早期冻结 ViT 主体，末端归一化和视觉投影继续训练。全局学习率在 28T / 45T tokens 处开始衰减，此时以更小学习率解冻视觉塔。官方模型代码提供了视觉运算和张量名称。报告没有给出每一对原始图文数据和对比文本塔的完整可执行训练代码。[官方报告，§3 与 §4.2.2](https://arxiv.org/pdf/2609.19969) · [官方视觉实现，固定版本](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/vision.py)

## 教学运行

| 阶段 | 训练信号和预算 | 交给下阶段 |
| --- | --- | --- |
| V-C 图文对比 | 4 层 192 宽原生 ViT，224² 等比填充。每批 32 个不同图像，4,000 次更新共 128,000 对曝光。英文描述取前 96 个 BPE，文本编码器取轻量词向量均值。相同截断 token 序列标为多正例。 | `tower.pt` 包含 35 个 ViT 张量和来源清单。文本词向量以及两端的对比投影到此结束。 |
| V-AR 有监督视觉训练 | 接入约 35.46M 参数、随机初始化的临时 MoE 解码器。8M 实际受监督 CE 位置，caption/OCR/chart/VQA 抽样权重为 45/25/15/15%。动态处理器使用 64 个视觉特征的预算，最低像素预算 3,136，据实际宽高规划图像网格。 | 导出更新后的 1,887,360 个 ViT 参数。检查来源阶段、tokenizer、视觉拓扑，以及确有优化更新、图像输入和 CE 标签。 |
| 主干联合预训练 | 512 宽、12 层主干和视觉投影重新随机初始化。预算 2.5B 实际受监督 CE 位置、1M 张图曝光。2,048 上下文，开启层次候选。首 1.55B 冻结 ViT 主体，此后解冻并采用主干十分之一的视觉学习率。 | 同一次训练跨过 1.55B 边界。文本优化器动量、游标、已消费 CE 计数和学习率进度连续，解冻后的 ViT 主体才开始积累自己的动量。 |

V-C 使用 FineVision 中的 ALLaVA-LAION 英文生成描述。固定来源的 43,510 条原生记录按原有 train/val/test 检查 group 与 RGB 身份。提取出 21,409 / 103 / 199 个不同图像，约 5.98 遍训练曝光。冻结后的文件各自绑定 SHA-256，来源质量字段仍是候选级。训练期间用 103 对验证图文计算双向检索 R@1，199 对测试样本留出。真实答题另测短答准确率、逐行 OCR CER 和换图对照。

V-AR 和主干共享原生单图来源、解码器格式及数据隔离规则。按 seed 42、每批 2 行、累积 4 批重放完整采样，V-AR 实际访问 63,432 张不同图片，每张一次，共 8,000,460 个 CE 标签。主干首个更新包含 6 条文本和 2 条图片，预计全程抽取 3,239,008 行，同一行使用次数最多为 6。以上是确定性采样回放，训练消耗以各阶段实际账本为准。

## 复用命令

将原始组件、图像根目录、图文输出、训练输出、纯视觉库和联合库分别绑定到下方环境变量。两个原生库的 `media-mixture.json` 固定抽样配比与实际监督预算。

```bash
python -m scripts.prepare_v41_caption_pairs --source "$NATIVE_COMPONENT" \
  --media-root "$MEDIA_ROOT" --output "$PAIRS"
python -m scripts.v41_vision_curriculum contrastive --data "$PAIRS" \
  --config configs/strategies/minideepseekv41-vision-main.json --output "$RUN/V-C" \
  --steps 4000 --batch-size 32 --side 224 --text-length 96 --eval-limit 103
python -m scripts.v41_vision_curriculum attach --role temporary --tower "$RUN/V-C/tower.pt" \
  --config configs/strategies/minideepseekv41-vision-temporary.json \
  --tokenizer "$PAIRS/tokenizer.json" --output "$RUN/temporary-init.pt"
python -m minifrontier.training.train --model minideepseekv41 --stage pretrain \
  --data "$VISUAL" --media-mixture "$VISUAL/media-mixture.json" --ce-tokens 8000000 \
  --sequence-length 2048 --batch-size 2 --grad-accum 4 --seed 42 --max-data-epochs 1 \
  --init "$RUN/temporary-init.pt" --output "$RUN/V-AR"
python -m scripts.v41_vision_curriculum extract --checkpoint "$RUN/V-AR/model.pt" \
  --output "$RUN/generative-vit.pt"
python -m scripts.v41_vision_curriculum attach --role main --tower "$RUN/generative-vit.pt" \
  --config configs/strategies/minideepseekv41-vision-main.json --seed 20261111 \
  --tokenizer "$PAIRS/tokenizer.json" --output "$RUN/main-init.pt"
python -m minifrontier.training.train --model minideepseekv41 --stage pretrain \
  --data "$MIXED" --media-mixture "$MIXED/media-mixture.json" --ce-tokens 2500000000 \
  --sequence-length 2048 --batch-size 2 --grad-accum 4 --seed 42 --max-data-epochs 6 \
  --schedule wsd --v41-thaw-ce 1550000000 --vision-lr 3e-5 --lr 3e-4 --projector-lr 3e-4 \
  --init "$RUN/main-init.pt" --output "$RUN/joint"
```

对比阶段加 `--resume` 会从同一目录的 checkpoint 恢复优化器和由步数决定的样本顺序。通用训练阶段用 `--resume path/to/checkpoint.pt` 替换 `--init`。传入 `--next-plan` 可在 V-C 完成后校验塔的 SHA、初末留出检索和各阶段输入，顺序执行同一设备的后续命令，各自保存进度。主干的动态解冻当前支持单设备。

这里的学习范围包括原生 2D RoPE ViT、3×3 合并投影、阶段隔离和层次索引。图文来源、文本编码器、分辨率、训练量和上下文均有缩减。长上下文增量推理会重新运算已有前缀。2.5B 的计数是实际受监督 CE 位置，原报告 45T 的分母是预训练 tokens。

## 同一张设备自动衔接

先生成可检查的运行计划。计划记录本机数据、配比、模型配置的 SHA-256 以及每个阶段的参数和输出位置。以下命令替代上面的分段启动。

```bash
python -m scripts.prepare_v41_continuation --pairs "$PAIRS" --visual "$VISUAL" \
  --mixed "$MIXED" --run "$RUN" --output "$PLAN_DIR/continuation.json"
python -m scripts.v41_vision_curriculum contrastive --data "$PAIRS" \
  --config configs/strategies/minideepseekv41-vision-main.json --output "$RUN/V-C" \
  --next-plan "$PLAN_DIR/continuation.json"
```

完成 V-C 后，进程会释放自己的 CUDA 上下文、进入控制器，并在原设备上依次运行后续阶段。若下游训练中断，确保旧进程已退出，沿用同一设备和计划恢复，控制器会读取已完成的收据并从中断阶段最近一次完整 checkpoint 接续。

```bash
python -m scripts.run_v41_visual_stages --plan "$PLAN_DIR/continuation.json" \
  --contrastive-output "$RUN/V-C"
```
