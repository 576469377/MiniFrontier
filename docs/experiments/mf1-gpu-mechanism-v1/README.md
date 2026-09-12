# MF1 228M 首轮 GPU 实验（2026-09-09）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**完整 228,235,809 参数配置在五种输入下均完成前后向和 AdamW 更新。** 这是 2026-09-09 的 RTX 3090 资源探测与两组 500K CE 学习实验启动记录。学习结果后来收录于[9 月 10 日档案](../mf1-language-performance-v1/README.md)；当前正式训练见[预训练计划](../../pretraining-plan.md)。

## 已完成的资源探测

入口为[profile_mf1.py](../../../scripts/profile_mf1.py)，seed 42。完整配置、环境和逐步数值见[资源报告](resource-probe.json)。当时 GPU 与两个 DeepSeek 实验共享。下表给出每场景第 2 次更新耗时；显存列是本进程截至该场景的累计峰值，不能据此比较两种图像预算的独立峰值。

| 场景 | 输入 token | 第 2 次更新时间 | 累计峰值 reserved |
|---|---:|---:|---:|
| 文本 | 128 | 3.94 s | 3.88 GiB |
| 文本 | 512 | 10.69 s | 3.90 GiB |
| 图像 49 特征位置 | 512 | 11.14 s | 4.19 GiB |
| 图像 196 特征位置 | 512 | 10.75 s | 4.19 GiB |
| 视频 128 特征位置 | 512 | 11.19 s | 4.20 GiB |

五种输入均完成前后向和 AdamW 更新，梯度有限，媒体场景视觉梯度非零；整卡最低空闲约 **8.34 GiB**。每场景只有两次更新，第一步预热，reserved 是同一进程累计峰值。这个 10 次更新探测不等同于 20＋100 持续测量。

## 有界学习实验配方

资源探测后的学习实验用于检查完整模型能否在固定数据上获得学习信号；本页保留启动和恢复条件，完成时的文字与视觉结果在后续档案中报告。

两组从零初始化完整模型，主干 LR 分别为 **3e-4 / 1.5e-4**，各 500K 有效 CE，MTP 目标另计。数据为教育文本、生成数学、色块图像/视频，使用独立 32K 候选词表；来源见[数据说明](../../guides/data-sources.md)，设备与预算见[调度记录](../../operations/mf1-mechanism-experiments.md)。

第 2 步后的恢复沿用原 token 账本，每 25 步验证并保存滚动检查点。初期验证只读取覆盖五个领域的短前缀，用于发现异常；完整生成与视觉对照在完成档案中另报。

[学习快照](learning-snapshot.json)保留采集时间、源码、更新和恢复点 hash，曲线见[LR3e-4](adamw-lr3e-4.metrics.jsonl)、[LR1.5e-4](adamw-lr1.5e-4.metrics.jsonl)。两组在第 2 步退出，再由新 CUDA 进程重载全部状态并续跑；此 GPU 试验没有另做不间断轨迹的逐位比较。

## 复现入口

以下为历史实验重建入口，需要已准备的 `data/strategy-recipe-public-v2/corpus.sqlite` 及校验清单。首次离线体验见[入门指南](../../guides/minifrontier1.md)。

<details>
<summary>已有数据上的构造与运行命令</summary>

```bash
MINIFRONTIER_MIN_FREE_GIB=50 uv run python scripts/prepare_mf1_mechanism_data.py \
  --database data/strategy-recipe-public-v2/corpus.sqlite \
  --output data/mf1-mechanism-public-v3 --seed 42

# 创建实验所用版本的独立代码目录；数据和输出放在该目录之外。
git worktree add --detach outputs/mf1-source-mechanism-v1 f2e96ca
uv run python scripts/run_mf1_trial.py \
  --source-checkout outputs/mf1-source-mechanism-v1 \
  --data data/mf1-mechanism-public-v3 \
  --output outputs/mf1-mechanism-v1/adamw-lr3e-4 \
  --gpu 0 --lr 0.0003 --token-budget 500000 \
  --input-batch-tokens 1024 --memory-gib 6
```

第二组改为 `--lr 0.00015`，使用独立输出 `adamw-lr1.5e-4`。两组可顺序执行或使用不同空闲 GPU，监督进程需保持运行；每组的代码、数据和预算按原记录绑定。

</details>

启动前回归为 **298 CPU passed、1 skipped**，41 项 CUDA 未选入；Ruff、格式和 mypy 通过。数据检查覆盖 Unicode 分块、测试隔离、分组泄漏与分块去重。本目录保留数值与来源校验值，不分发权重或原始媒体。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [500K 完成结果](../mf1-language-performance-v1/README.md)
