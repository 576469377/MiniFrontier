# 离线最小示例

用程序生成的整数加法题，依次运行数据编码、预训练、暂停恢复、监督微调、评估和生成。模型与词表采用微型配置；安装完成后，示例无需下载语料或权重。MF1 的图像和视频示例见[独立指南](minifrontier1.md#离线最小示例)。

## CPU

在 Linux 的 Git checkout 中使用 Python 3.11+：

```bash
uv sync --locked --extra dev
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 uv run minifrontier quickstart \
  --model all --device cpu --output outputs/quickstart-cpu
uv run minifrontier generate \
  --checkpoint outputs/quickstart-cpu/minideepseekv4/sft/model.pt \
  --prompt 'What is 10 + 2?' --max-new-tokens 12 --temperature 0 --device cpu
```

每次使用新输出目录，已有目录会被拒绝覆盖。`--model` 也可单独指定 `minikimik3`、`miniqwen4` 或 `minideepseekv4`。

## 单张 RTX 3090

安装 GPU 依赖，包括 Kimi KDA 使用的固定版本 FLA，并选择空闲设备：

```bash
uv sync --locked --extra dev --extra training
CUDA_VISIBLE_DEVICES=0 MINIFRONTIER_MIN_FREE_GIB=1 uv run minifrontier quickstart \
  --model all --device cuda:0 --output outputs/quickstart-cuda
```

三个模型按顺序运行。这里仅为微型示例将磁盘保留量设为 1 GiB；训练器默认保留 50 GiB，正式训练使用[存储计划](../pretraining-plan.md#resources)规定的余量。

## 实际执行与预期结果

| 项目 | 示例设置 |
|---|---|
| 数据 | 112 道加法题，各生成 PT 和 SFT 记录；每阶段 train/val/test 为 80/16/16 |
| 划分 | 按左操作数分组，PT 与 SFT 共用分组；test 不参与训练和示例验证 |
| tokenizer | 只读取训练集；目标词表 512，当前生成 313 个 token |
| 预训练 | 总计 8 次更新；第 4 次保存后暂停，再以同一预算恢复至第 8 次 |
| SFT | 从预训练权重初始化，再训练 8 次 |
| 优化 | 序列长 64、batch 2、累积 1、seed 42、AdamW LR 0.001 |
| 验证 | 每 2 次更新评估完整微型验证集；最后加载 SFT 权重生成 |

每个模型目录包含 `config.json`、`data/`、`pretrain/checkpoint.pt`、`pretrain/model.pt`、`sft/model.pt` 和 `report.json`。报告保存命令、配置与数据校验值、验证指标、生成文本和耗时；应显示 `resume_executed: true`、`pause_step: 4`，两个最终阶段均为 `complete`。恢复的状态包括优化器、数据游标和 RNG。

生成可能为空、错误数字或乱码。此示例检查训练流程，不评定语言能力；它不启用视觉和 MTP，64-token 序列也未覆盖全部长压缩块路径。

## 2026-09-09 实测

| 模型 | 参数项计数（报告口径） | CPU，2 线程 | 单张 RTX 3090 |
|---|---:|---:|---:|
| MiniDeepSeek-V4 | 79,255 | 5.68 秒 | 10.62 秒 |
| MiniKimi-K3 | 74,066 | 7.79 秒 | 108.17 秒 |
| MiniQwen4 | 32,596 | 3.72 秒 | 9.29 秒 |

该次 CPU 生成中，DeepSeek、Kimi 输出为空，Qwen 输出 `8`。参数计数沿用当时报告对全部 Parameter 元素求和的口径，DeepSeek 包含固定整数 hash 路由项；首页研究配置则统计浮点参数，容量与统计范围均不同。

测量环境为 PyTorch 2.13，耗时从数据生成计至生成结束，不含安装和进程导入。GPU 当时与其他训练共享，且包含内核初始化/编译，不能作为独占吞吐基准或正式容量的速度估计。完整记录见[预览验收档案](../experiments/preview-quickstart)。

## 安装包与浏览器

安装本地构建的 wheel 后，可在 checkout 外运行 `minifrontier quickstart`，并通过 CLI 显式加载其检查点。正式 `--run-kind strategy` 仍要求 Git checkout 和策略文档，详见[支持范围](../releases/v0.1.0.md)。

`minifrontier demo --root outputs --device cpu` 默认只展示与权重 hash 绑定、通过能力验收的模型，当前没有这样的聊天权重。查看 quickstart 等实验检查点时使用[实验视图](demo-experiments.md)。
