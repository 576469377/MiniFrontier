# MiniFrontier

在固定官方源码的基础上缩小旗舰模型，学习架构，亲手构造数据、训练、评估和推理。

[中文训练指南](docs/training.md) · [项目梳理](docs/project-review.md) · [架构与目录](docs/architecture.md) · [来源与许可证](THIRD_PARTY_NOTICES.md)

| 模型 | 当前文本实现 | 浮点参数（不含 MTP） |
|---|---|---:|
| [MiniQwen4](docs/models/miniqwen4.md) | GDN / gated residual / PLE / MoE / QSA，源码语义分块 Muon | 431,609,632 |
| [MiniKimi-K3](docs/models/minikimik3.md) | KDA / gated MLA / LatentMoE / SiTU / AttnRes | 166,109,480 |
| [MiniDeepSeek-V4](docs/models/minideepseekv4.md) | 滑窗 / CSA-HCA / indexer / hash-MoE / mHC | 229,996,877 |

名称采用 `Mini` + 官方模型/架构名；包名为 `miniqwen4`、`minikimik3`、`minideepseekv4`。MiniQwen4 对应 Qwen3.8-Flash-Next 发布所用的 `qwen4_exp` 源码，属于独立教学项目。DeepSeek 以 V4-Flash 为具体来源。

**现已提供可执行的文本训练链路，不代表旗舰模型完整复现。** 原生视觉、MTP 训练接入、量化感知后训练和完整官方规模评测仍未完成。公开推理源码没有披露的初始化、数据和训练超参数均标记为本地选择。Kimi/DeepSeek 的源码对照目前覆盖关键组件，不能等同于整模型与官方训练数值等价。

**2026-09-08 效果审计：`educational-v1` 未达到基本对话目标。** 阶段完成和损失下降不能作为模型可用的证据；SFT 已出现重复、答非所问，DPO 也未修复。见[失败复盘与纠正措施](docs/training-failure-v1.md)。当前权重用于排查与学习，不标记为可用对话模型。

## 开始使用

Python 3.11+；当前锁定环境使用 PyTorch 2.13。3090 上的 Kimi CUDA 训练使用 FLA 0.5.2。

```bash
uv sync --locked --extra dev --extra training --extra monitoring
uv run minifrontier models
uv run minifrontier doctor

# 对固定 revision 的完整源文件做均匀抽样，再清洗、划分和训练 tokenizer
# 必须读完整源文件；快速工程检查可显式指定 --sampling prefix
uv run minifrontier prepare-data --output data/educational-v1

# 单卡训练；也可用 torchrun 启动同一个入口
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run minifrontier train \
  --model minikimik3 --data data/educational-v1 \
  --output outputs/minikimik3/my-run/pretrain --steps 1000

# 显式指定数据覆盖量；这里一轮仅演示预算方式，不保证对话能力
uv run python scripts/launch_training.py --run coverage-example \
  --gpu-groups 0,1 2,3 4,5 --batch-size 2 --grad-accum 2 \
  --pretrain-epochs 1 --sft-epochs 1 --save-every 100

uv run python scripts/training_status.py
uv run minifrontier demo --root outputs --device cpu
```

浏览器 demo 默认地址为 `http://127.0.0.1:7860`，默认展示 SFT，可切换 DPO、预训练和最新阶段进行比较，并显示对话验收状态。GPU 推理可指定 `--device cuda:0`。通过 SSH 使用时转发对应端口。

## 训练流程

```text
公开文本 → 清洗 / 去重 / 固定划分 → 自训 ByteLevel BPE → 预训练
                                                     ↓
Qwen / DeepSeek：dense indexer distillation → sparse CPT
                                                     ↓
                                             SFT → 对话效果评估 → 可选 DPO
                                              └→ GRPO（可验证任务）
                                              └→ MOPD（显式提供多教师）
```

自动管线现在要求显式指定预训练和 SFT 的步数或 epoch 预算，并在 recipe 中记录样本覆盖量。DPO 默认关闭。原先 1,000 步预训练、500 步 SFT、100 步 DPO 的 `educational-v1` 仅完成执行链路，基本对话效果验收失败，不能作为复现可用模型的推荐配方。Qwen/DeepSeek 保留索引器蒸馏与稀疏 CPT；GRPO/MOPD 是独立可选入口，详见[训练指南](docs/training.md)。

检查点包含模型、优化器、阶段、完整配方、tokenizer 校验和、数据游标以及各 rank 随机状态；支持相同配方的精确恢复。训练和验证记录位于每模型独立目录，TensorBoard 与 JSONL 同步保存。DDP 完成阶段前逐项核验所有 rank 的参数一致。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest -q -m 'not cuda'
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 uv run pytest -q -m cuda
uv run ruff check .
uv run ruff format --check .
uv run mypy minifrontier scripts --ignore-missing-imports
```

实际双卡短流程的日志和显存记录汇总在[运行验收记录](docs/audits/training-acceptance.json)。验收用真实处理语料，每阶段两步；用于验证工程链路，不用于声明模型能力。较早的 [Qwen 单步源码验收](docs/audits/miniqwen4-acceptance.json)作为历史记录保留。

## 数据与许可

公开语料来自 [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset)，本地重新处理并训练 tokenizer；它不是任何旗舰模型的官方训练数据。数据卡标注 Apache-2.0 / CC-BY-NC-2.0，不能随项目代码统一标成 Apache-2.0。原始语料、权重和训练产物不进入源码或 wheel。

项目原创代码为 Apache-2.0。Qwen Transformers 源码、Kimi 自定义许可源码、DeepSeek MIT 源码分别保留原始条款；见[第三方说明](THIRD_PARTY_NOTICES.md)。
