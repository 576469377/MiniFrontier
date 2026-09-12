# 离线最小示例

本示例依次完成数据生成、预训练、暂停恢复、监督微调、评估和生成，适合首次检查安装与训练流程。数据为程序生成的整数加法题，模型与词表也采用微型配置。安装依赖需要网络或本地缓存；后续运行无需下载语料或模型。

## CPU

在 Linux 的 Git checkout 中使用 Python 3.11+（Windows 请使用 Linux/WSL2 环境）：

```bash
uv sync --locked --extra dev
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 uv run minifrontier quickstart \
  --model all --device cpu --output outputs/quickstart-cpu
uv run minifrontier generate \
  --checkpoint outputs/quickstart-cpu/minideepseekv4/sft/model.pt \
  --prompt 'What is 10 + 2?' --max-new-tokens 12 --temperature 0 --device cpu
```

每次选一个新输出目录；已有目录会被拒绝覆盖。也可只指定 `--model minikimik3`、`miniqwen4` 或 `minideepseekv4`。

## 单张 RTX 3090

确认所选 GPU 可用，Kimi CUDA 需要 training extra 中固定的 FLA：

```bash
uv sync --locked --extra dev --extra training
CUDA_VISIBLE_DEVICES=0 MINIFRONTIER_MIN_FREE_GIB=1 uv run minifrontier quickstart \
  --model all --device cuda:0 --output outputs/quickstart-cuda
```

这里只将微型示例的磁盘保留量设为 1 GiB；训练器默认保留 50 GiB，正式运行按[存储计划](../pretraining-plan.md#resources)设置更高余量。示例使用真正的三个模型构造入口、数据编码器和训练器，单进程依次执行三组，不需要第二张卡。

## 实际执行与预期结果

每个模型生成 112 道加法题，各生成 PT 和 SFT 记录；每阶段 train/val/test 为 80/16/16。按左操作数划分组，两阶段共享分组；tokenizer 只读取训练集，目标词表 512，当前生成 313 个 token。test 不参与训练和本示例验证。

PT 总预算 8 次更新，在第 4 次暂停并保存优化器、游标和 RNG，再用同一预算恢复到第 8 次。SFT 从 PT 权重初始化另训 8 次；每 2 次更新对整个微型验证集评估，最后显式加载 SFT 权重生成。序列长 64、batch 2、累积 1、seed 42、AdamW LR 0.001。`report.json` 保存完整命令、配置与数据校验值、验证指标、阶段状态、生成文本和耗时。

预期文件为各模型目录中的 `config.json`、`data/`、`pretrain/checkpoint.pt`、`pretrain/model.pt`、`sft/model.pt`、`report.json`。报告应显示 `resume_executed: true`、`pause_step: 4`，两个最终阶段均为 `complete`。训练器测试另外比较暂停恢复与连续训练的参数、token 账本和游标。

**生成可能为空、错误数字或乱码。** 下方 2026-09-09 的 CPU 记录中，DeepSeek/Kimi 输出为空，Qwen 输出 `8`；这些输出不表示模型学会加法或语言。本示例不启用视觉和 MTP，64-token 序列也不覆盖全部长压缩块路径。示例单独记账，不出现在浏览器默认的能力合格模型列表。

## 2026-09-09 实测

| 模型 | 参数项计数（报告口径） | CPU，2 线程 | 单张 RTX 3090 |
|---|---:|---:|---:|
| MiniDeepSeek-V4 | 79,255 | 5.68 秒 | 10.62 秒 |
| MiniKimi-K3 | 74,066 | 7.79 秒 | 108.17 秒 |
| MiniQwen4 | 32,596 | 3.72 秒 | 9.29 秒 |

这里沿用当时 `report.json` 对全部 Parameter 元素求和的记录，DeepSeek 的计数包含固定整数 hash 路由项；首页的研究配置表则只统计浮点参数。两处容量与统计范围均不同。

测量为 PyTorch 2.13 环境中每模型的数据生成到生成结束，不含安装、命令进程导入时间；GPU 测量与既有训练共享一张 3090，并包括当次内核初始化/编译影响。它是端到端可执行性记录，不是独占 GPU 吞吐基准；尤其不能据此推断正式容量的 CPU/GPU 速度关系。完整数值报告位于[预览验收档案](../experiments/preview-quickstart)。

## 安装包与浏览器

从本地构建的 wheel 安装后，在 checkout 外运行同样的 `minifrontier quickstart` 和显式 checkpoint CLI 生成，无需源码树；生成的数据仍完全离线。正式 `--run-kind strategy` 暂要求 Git checkout 和完整策略文档，见[支持范围](../releases/v0.1.0.md)。

浏览器命令为 `minifrontier demo --root outputs --device cpu`，默认仅展示与实际权重 hash 绑定、通过能力验收的模型。当前没有这样的可用聊天权重；完成 quickstart 不会自动产生浏览器可选模型。
