# MiniFrontier

在固定官方源码的基础上缩小旗舰模型，学习架构，亲手构造数据、训练、评估和推理。

[中文训练指南](docs/training.md) · [项目梳理](docs/project-review.md) · [架构与目录](docs/architecture.md) · [来源与许可证](THIRD_PARTY_NOTICES.md)

| 模型 | strategy-v2 实现 | 当前诊断配置参数量 |
|---|---|---:|
| [MiniQwen4](docs/models/miniqwen4.md) | GDN / GR / PLE / MoE / QSA + 原生视觉 + 四流 MTP | 513,405,536 |
| [MiniKimi-K3](docs/models/minikimik3.md) | KDA / MLA / LatentMoE / AttnRes + MoonViT + MTP | 204,526,216 |
| [MiniDeepSeek-V4](docs/models/minideepseekv4.md) | SWA128 / CSA-HCA / hash-MoE / mHC + 文本 MTP | 243,983,472 |

名称采用 `Mini` + 官方模型/架构名；包名为 `miniqwen4`、`minikimik3`、`minideepseekv4`。MiniQwen4 对应 Qwen3.8-Flash-Next 发布所用的 `qwen4_exp` 源码，属于独立教学项目。DeepSeek 以 V4-Flash 为具体来源。

新训练配置位于 `configs/strategies`，上表包含 MTP；DeepSeek 按方案先训练文本，再迁移原生 Vision-Exp。根目录三个配置保留文本兼容用途。原生视觉、MTP、各自 Muon/路由更新和增量缓存已接入；QAT 仿真、Kimi sampled-token MOPD、DeepSeek full-vocabulary reverse-KL OPD 有独立代码路径，仍需配方和能力验收。

**完整训练与可用模型尚未完成。** 三个模型正在执行独立 20M-token 配方试验。Qwen 首轮 Q0 失败后，累计约 1M CE 的续诊断达到训练内算术 16/16，留出仍为 0/9；诊断与配方试验均不计正式主预算。正式数据准入、配方选择、教师资格、草稿训练和发布门槛仍有待完成项，详见[方案执行记录](docs/audits/strategy-implementation-v2.md)。公开材料未披露的 mini 配方明确属于本地选择。

**2026-09-08 效果审计：`educational-v1` 未达到基本对话目标。** 阶段完成和损失下降不能作为模型可用的证据；SFT 已出现重复、答非所问，DPO 也未修复。见[失败复盘与纠正措施](docs/training-failure-v1.md)。当前权重用于排查与学习，不标记为可用对话模型。

## 开始使用

Python 3.11+；当前锁定环境使用 PyTorch 2.13。3090 上的 Kimi CUDA 训练使用 FLA 0.5.2。

```bash
uv sync --locked --extra dev --extra training --extra monitoring --extra data
uv run minifrontier models
uv run minifrontier doctor

# 受磁盘预算约束的公开来源试验池，不代表正式数据规模
uv run minifrontier prepare-public-data --output data/public-pilot \
  --limits-mib '{"zh_edu":96,"en_edu":64,"python_edu":16,"ultrachat":32}' --max-gib 8
uv run minifrontier prepare-tokenizers --corpus-root data/public-pilot \
  --output data/tokenizers-v2
uv run minifrontier encode-data --corpus-root data/public-pilot \
  --tokenizer-path data/tokenizers-v2/tokenizer-65536.json --output data/text-v2

# 单卡训练；也可用 torchrun 启动同一个入口
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run minifrontier train \
  --model minideepseekv4 --config configs/strategies/minideepseekv4-v2.json \
  --data data/text-v2 --output outputs/text-diagnostic \
  --ce-tokens 500000 --sequence-length 64 --run-kind acceptance

# 正式方案使用 configs/strategies/*-plan.json；先完成诊断与配方比较
# 实际输入 batch：--input-batch-tokens 16384（按各卡非 padding 输入累计）
# 显式阶段变更：--init-transition text-to-vision / qat / mtp-weight
# DeepSeek V1 另加 --visual-warmup；视觉与投影 LR 分别设置

uv run python scripts/training_status.py
uv run minifrontier demo --root outputs --device cpu
```

浏览器 demo 默认地址为 `http://127.0.0.1:7860`，默认只展示通过能力验收的检查点；可切换历史 SFT/DPO/预训练作对照。诊断 run 不进入默认列表。GPU 推理可指定 `--device cuda:0`。

工作盘写入默认预留 50 GiB，数据和权重使用独立目录并校验 hash；不要把大型缓存放在空间紧张的根分区。当前双卡诊断使用独立源码 checkout，GPU 分配为 Kimi 0–1、Qwen 2–3、DeepSeek 4–5。

## 训练流程

```text
固定源码与词表 → 数值/原生视觉/MTP诊断 → 等预算配方比较 → 正式数据准入
Kimi：联合 PT → SFT/QAT → 9 位领域/模式教师 → MOPD → 7-step draft → demo
Qwen：dense 联合 PT → indexer → sparse CPT → SFT/GRPO → MTP draft → demo
DeepSeek：Text-v2 PT/indexer/CPT → Vision-v1接入/CPT → SFT/QAT → 12 教师 → OPD → DSpark → demo
```

上图为目标依赖，仍有未实现和未验收的阶段，不能作为完成清单。新入口按 CE/input/response 实际 token 计费；`--run-kind strategy` 检查源码、数据、配置、依赖证据与实测性能。`scripts/run_recipe_pilot.py` 仅执行两组独立 20M-token Muon/AdamW 试验，之后仍需 LR、MTP、tokenizer 和补种子对照，不自动进入主训练。旧 `launch_training.py` 保留作历史对照，不执行新方案。DPO 默认关闭。

检查点包含模型、优化器、阶段、完整配方、tokenizer 校验和、数据游标以及各 rank 随机状态；支持相同配方的精确恢复。文本域按 CE token 采样；视觉域按样本采样，同时单列图像/视频暴露预算。PT/SFT 的 `best-model.pt` 按同一验证集 LM NLL 保存，仍须生成能力验收。训练和验证记录位于每模型独立目录，TensorBoard 与 JSONL 同步保存。DDP 完成阶段前逐项核验所有 rank 的参数一致。

`train-draft` 提供独立的 Kimi LK、Qwen 四流 CE 和 DeepSeek DSpark 训练/恢复入口，
目标冻结且导出绑定精确目标 hash。`generate --draft` 提供接受/拒绝与原生缓存回滚，
目前仅作正确性路径，尚无训练后加速结论，见[草稿适应与推理](docs/draft-adaptation.md)。
新的 control-v1 模板贯通 SFT、原生视觉/工具 rollout 和 CLI；实际行为概率、策略
权重 hash、工具观察掩码与终态奖励分别保留。此接口尚未完成正式后训练，见
[后训练适应与范围](docs/posttraining-adaptation.md)。
`expert_execution="batched"` 是尚未通过完整 BF16 梯度比较的实验选项，三个方案配置
及正在运行的训练均保持 `loop`；微基准提速不构成正式配方准入。

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

旧语料来自 [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。strategy-v2 使用固定版本的中文教育文本、FineWeb-Edu、SmolLM Python-Edu、UltraChat 及逐来源审核的视觉候选池；来源许可分别记录，Python-Edu 原代码许可未解决的行明确标记为试验数据。它们不是旗舰官方训练数据，不能随项目代码统一标成 Apache-2.0。原始语料、权重和训练产物不进入源码或 wheel。

项目原创代码为 Apache-2.0。Qwen Transformers 源码、Kimi 自定义许可源码、DeepSeek MIT 源码分别保留原始条款；见[第三方说明](THIRD_PARTY_NOTICES.md)。
