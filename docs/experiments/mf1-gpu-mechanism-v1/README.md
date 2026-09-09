# MF1 228M GPU 资源与学习实验

2026-09-09，完整 228,235,809 参数配置开始进入 RTX 3090 实验。此档案将已完成的短程资源探测与后续有界学习实验分开记录。正式数据审核、32K/64K 词表比较、完整性能矩阵及 3B 主预训练尚未完成。

## 已完成的资源探测

训练器源码为 `f226845da4079a641a9ac69ed1536ba54751752c`；执行脚本与当前 [`profile_mf1.py`](../../../scripts/profile_mf1.py) 对应，SHA256 写在 [resource-probe.json](resource-probe.json)。完整配置、seed42、环境和逐次更新数据也在该文件内。GPU 当时与两个 DeepSeek 实验共享。

| 场景 | 输入 token | 第 2 次更新时间 | 累计峰值 reserved |
|---|---:|---:|---:|
| 文本 | 128 | 3.94 s | 3.88 GiB |
| 文本 | 512 | 10.69 s | 3.90 GiB |
| 图像 49 特征位置 | 512 | 11.14 s | 4.19 GiB |
| 图像 196 特征位置 | 512 | 10.75 s | 4.19 GiB |
| 视频 128 特征位置 | 512 | 11.19 s | 4.20 GiB |

五种输入均完成前向、反向和 AdamW 更新，梯度有限，媒体场景的视觉梯度非零。整卡最低空闲约 8.34 GiB。每场景仅两次更新，第一步用于预热；reserved 为同一进程的累计峰值。随机输入上的 loss 不用于评价可学习性，这 10 次更新也不满足方案 §17.3 的 20 次预热和至少 100 次测量要求。

## 有界学习实验配方

两组均从零初始化完整模型，比较主干矩阵 LR 3e-4 与 1.5e-4，各 500K 有效 CE；辅助 MTP 目标单独计数。数据为已清洗教育文本、生成数学和生成图像/视频，采用独立 32K 候选词表。来源版本和处理范围见[数据说明](../../guides/data-sources.md)，设备分配、显存/磁盘预算、自动停止及与正式配置的区别见[调度记录](../../operations/mf1-mechanism-experiments.md)。

前两次更新后强制退出，再从 checkpoint 恢复全部训练状态。这两步属于同一次 run 的预算，恢复不会重置 token 账本。每 25 次更新记录验证损失并保存一个滚动恢复点。验证器的短前缀覆盖五个数据领域，但样本量只适合早期异常检查，不能据此确定最终配方或声明通用语言、视觉能力。

## 复现入口

以下命令要求 GPU 环境和 Git checkout，并要求已通过原数据流程构造 `data/strategy-recipe-public-v2/corpus.sqlite` 及其校验清单。此处提供的是该冻结语料上的二次构造入口；首次安装和完全离线流程见[入门指南](../../guides/minifrontier1.md)。准备时不下载额外语料，原始记录和生成媒体不随源码分发。

```bash
MINIFRONTIER_MIN_FREE_GIB=50 uv run python scripts/prepare_mf1_mechanism_data.py \
  --database data/strategy-recipe-public-v2/corpus.sqlite \
  --output data/mf1-mechanism-public-v3 --seed 42

# 先提交实现，使用确定的提交创建冻结源码；输出、数据均在工作树外。
git worktree add --detach outputs/mf1-source-mechanism-v1 HEAD
uv run python scripts/run_mf1_trial.py \
  --source-checkout outputs/mf1-source-mechanism-v1 \
  --data data/mf1-mechanism-public-v3 \
  --output outputs/mf1-mechanism-v1/adamw-lr3e-4 \
  --gpu 4 --lr 0.0003 --token-budget 500000 \
  --input-batch-tokens 1024 --memory-gib 6
```

第二组使用 GPU 5、`--lr 0.00015` 和独立输出 `adamw-lr1.5e-4`。设备编号属于本机调度示例，执行前按自己的设备选择；每组是独立单卡训练。脚本等待资源满足条件后启动，监督进程需保持运行。不要让训练在同一个被修改的源码目录中继续，也不要手工更改已经绑定的预算或数据。

原始报告和曲线保存在各 run。公开资源报告删除了 GPU UUID、其他卡的进程信息；保留原报告校验值，指标数值未改动。学习结果将在实际启动后另附注明时间的快照，未完成实验不写作成功结果。
