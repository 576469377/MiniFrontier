# MF1 语言诊断与单卡性能对照

2026-09-10 开始。本轮针对两个实际问题：500K CE 后，MF1 已能利用简单色块媒体，但仍不能正常回答基础文字问题；单卡训练的显存和算力也未充分使用。这里记录学习证据、实现检查和性能测量，各项按自己的完成状态解释。

## 已完成的 500K CE 实验

两组均完成 477 次更新、500,853 CE；参数量为 228,235,809，seed 42。冻结源码为 `f2e96ca22fa5122d6520d932a1e70e2eee00fef6`。完整配置、初始化和产物校验值见 [完成记录](completed-500k.json)，逐步曲线见 [LR 1.5e-4](adamw-lr1.5e-4.completed.metrics.jsonl)、[LR 3e-4](adamw-lr3e-4.completed.metrics.jsonl)。

| 项目 | LR 1.5e-4 | LR 3e-4 |
|---|---:|---:|
| 原验证前缀 LM NLL | 6.3779 | 6.4048 |
| 留出图片：首个生成颜色 token 正确 | 16/16 | 16/16 |
| 留出视频：首个生成颜色 token 正确 | 16/16 | 14/16 |
| 图片置黑后正确 | 4/16 | 5/16 |
| 视频置黑后正确 | 5/16 | 8/16 |
| 自我介绍、7+5 的生成抽查 | 重复、错误、未输出 EOS | 重复、错误、未输出 EOS |

视觉报告见 [1.5e-4](adamw-lr1.5e-4.visual.json)、[3e-4](adamw-lr3e-4.visual.json)。低学习率组换入不同答案的媒体后，32 个输出全部跟随替换媒体的颜色。这支持模型确实使用了色块媒体；不能据此推断通用视觉、视频理解或对话能力。

原 NLL 仅覆盖验证集前 12 条，含 4 条英文、2 条中文、2 条数学、2 张图片、2 段视频；图片和视频子集分别只有一种答案。该 NLL 不足以完成配方选择。本轮改为默认遍历完整验证集，同时把 TensorBoard 验证分母放入 `validation/` 命名空间，避免与累计训练 CE 混淆。新的诊断验证集与旧语料不同，两个 NLL 不能直接比较。

## 语言学习实验：运行中

两组均已完成第 2 次更新后的 checkpoint 重载，并在新的 CUDA 进程从第 3 步继续；[运行快照](sft-running-snapshot.json)及 [2e-5 曲线](adamw-lr2e-5.running.metrics.jsonl)、[1e-4 曲线](adamw-lr1e-4.running.metrics.jsonl)保留当时进度。

新运行冻结源码为 `d77c51c4ba074ba2d325f0b52e2ecceca392972e`。两组从上述 LR 1.5e-4 的完整检查点初始化，使用同一候选 32K tokenizer、数据和 seed，仅改变主干及标量参数的学习率。视觉学习率均为 5e-6。

| 控制项 | 设置 |
|---|---|
| 模型 | 完整 MF1 228M，视觉、MTP、lookup 保留 |
| 主干及标量 LR | 2e-5 / 1e-4，AdamW |
| 预算 | 每组 30,000 CE，按完整更新可能略超出；最多 10,000 updates |
| 每次更新输入预算 | 512 token，使用完整记录，不截断答案 |
| CE 比例 | 基础问答 90%、图片回放 7.5%、视频回放 2.5% |
| 训练集 | 项目编写的 32 条基础问答、32 张图片、8 段视频 |
| 留出集 | 106 条算术/复制任务、16 张图片、16 段视频，共 138 条 |
| 恢复与评估 | 第 2 次更新后退出并重载；每 25 次更新完整验证；训练后执行 greedy 生成、精确答案匹配及 EOS 统计 |

这是可控的 SFT 学习诊断。32 条问答被反复学习，训练题的精确匹配衡量记忆；独立题目才能衡量对应任务的泛化。该数据不是通用聊天语料，不承诺完成后具备聊天能力。

现有初始化权重尚未训练稀疏索引器，因此本轮显式使用 `--diagnostic-attention dense_pretrain`，保持原 attention 路径。这个覆盖只允许 `acceptance` 的 SFT 诊断；正式 SFT 仍需 P3 前置阶段和固定方案的准入证据。

数据生成不下载额外语料，不读取封存测试集；复制、加法的全部语言和改写形式按同一题目分组切分。tokenizer 按字节复制，媒体复用已有带校验值的生成资源。见 [控制集清单](control-data.json)和[扩展指令集清单](instruction-data.json)。后者已有 886 条训练记录，尚未开始。条件队列已就绪：每组须在 32 条训练问答中精确匹配至少 29 条、至少 30 条正常输出 EOS，且 32 条视觉留出样本至少答对 28 条，才从该组完成权重继续 100K CE 扩展指令诊断。未满足则停止扩展，保留结果。阈值是本轮操作条件，不能作为通用能力准入；见[续训条件](continuation-policy.json)。

```bash
# 已准备原机制数据，并有其 LR 1.5e-4 的完成检查点。
uv run python scripts/prepare_mf1_language_data.py \
  --source data/mf1-mechanism-public-v3 --output data/mf1-language-v1 \
  --config configs/minifrontier1/model_228m_native.json

git worktree add --detach outputs/mf1-source-language-v1 d77c51c
uv run python scripts/run_mf1_trial.py \
  --source-checkout outputs/mf1-source-language-v1 \
  --data data/mf1-language-v1/control --output outputs/mf1-language-sft-v1/adamw-lr2e-5 \
  --task language-sft --init outputs/mf1-mechanism-v1/adamw-lr1.5e-4/checkpoint.pt \
  --gpu 0 --lr 2e-5 --token-budget 30000 --input-batch-tokens 512 \
  --memory-gib 6 --reserve-gib 5
```

设备编号为复现示例，运行前按实际空闲设备选择。完整流程要求 Git checkout 和相应的本地产物；离线首次体验见[微型示例](../../guides/minifrontier1.md)。

## 缓存与 lookup 检查

同一 14-token 输入，关闭 TF32，对比完整前向和“7-token 前缀 + 逐 token 解码”：[原始数值](cache-checks.json)。

| KDA 路径 | FP32 最大绝对误差 | BF16 最大绝对误差 | 该样本各位置 greedy 一致率 |
|---|---:|---:|---:|
| 自动融合内核 | 0.006961 | 0.609375 | 100% |
| PyTorch 参考递推 | 0.00000286 | 0.0625 | 100% |

参考 FP32 通过 `atol=rtol=2e-4`，融合内核未通过这一严格阈值。这把该样本的主要差异定位到内核计算路径，仍不能证明任意输入的生成等价；也不能把单条样本的 greedy 一致当成完整 CUDA 缓存验收。

lookup 预填充原本逐 token 执行投影和卷积，现改为批量投影与有重置掩码的卷积窗口，参数及检查点键保持一致。旧实现保留为流式路径和测试参照。CPU 测试覆盖控制 token、媒体、padding、样本边界、梯度和拆分缓存；CUDA 228M 配置模块对照的 FP32 输出最大误差约 1.86e-9、梯度最大误差约 2.91e-11，BF16 输出最大误差约 4.99e-6，状态一致。见[模块检查结果](lookup-correctness.json)、[当次执行脚本](lookup-cuda-runner.txt)。脚本公开副本把本机绝对根目录改为当前目录，其余计算保持不变。

实现提交前，全量 CPU 检查为 302 passed、1 skipped；后续定向回归 53 passed，包含共享显存准入与 lookup autocast 状态修正。Ruff、格式和 mypy 通过。41 项既有 CUDA 测试未计入该全量 CPU 检查；这里的 CUDA 证据来自实际运行的专项对照。

## 性能：短筛完成基线，优化版正在测量

原实现的 [算子表](baseline-profile.json)有大量小算子调用。下面是独占单张 3090、完整配置、纯文本 512、每次实际输入 4096 token 的短筛。各组使用相同随机输入、seed、AdamW 和辅助目标，预热 1 次后测量 2 次；这些数字只用于筛选。

| microbatch | 梯度累积 | 原实现 CE/s | lookup 改造后 CE/s |
|---:|---:|---:|---:|
| 1 | 8 | 56.50 | 59.91 |
| 2 | 4 | 61.48 | 未测 |
| 4 | 2 | 63.85 | 68.26 |
| 8 | 1 | 63.92 | 运行中 |

数据见[基线逐次测量](baseline-batches.json)。4 增到 8 几乎没有进一步提升，说明增加显存占用本身不会保证提速。lookup 改造后的[运行中测量](optimized-batches-running.json)已完成 microbatch 1 和 4，其余短筛与持续测量仍在进行。相比原实现同项，短测差异约 6%–7%，尚不足以确认稳定加速；主机上并行任务也与基线采集时不同。

已启动的优化版队列依次测量 512 的 microbatch 1/4/8、2048 文本、196 特征图片，再对短筛选出的配置执行 **20 次预热 + 100 次无 profiler 更新**。结果只适用于测过的长度、batch 和模态；8K、视频矩阵及真实混合记录训练仍需单独测量。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_mf1.py \
  --config configs/minifrontier1/model_228m_native.json --output outputs/mf1-batch-screen \
  --length 512 --input-tokens 4096 --batches 1 4 8 --warmup 1 --updates 2
# 按短筛结果设置 --batches，并改用 --warmup 20 --updates 100 进行持续测量。
```

原始数据、完整检查点与设备身份信息不随本档案发布。运行中的学习曲线见本地 TensorBoard `mf1-language/`；性能、缓存和正确性检查不添加到学习曲线。
