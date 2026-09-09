# MiniFrontier

从旗舰开源模型的固定源码中学习并复用计算原语，构建自己的小型原生多模态模型 **MiniFrontier1.0**，亲手完成数据构造、训练、评估和推理。原来的 MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4 保留为学习与实验对照。

[文档导航](docs/README.md) · [MiniFrontier1.0 使用指南](docs/guides/minifrontier1.md) · [融合模型与状态](docs/models/minifrontier1.md) · [实现验收记录](docs/audits/minifrontier1-implementation.md) · [实验记录](docs/experiments.md) · [来源与许可证](THIRD_PARTY_NOTICES.md)

| 模型 | 架构 / 定位 | 当前配置参数量 |
|---|---|---:|
| **[MiniFrontier1.0](docs/models/minifrontier1.md)** | **主线：12 KDA + 2 CSA-4 + 2 QSA-MLA / 四流 GR / LatentMoE / 单点 lookup / 原生 ViT / MTP** | **228,235,809** |
| [MiniQwen4](docs/models/miniqwen4.md) | GDN / GR / PLE / MoE / QSA + 原生视觉 + 四流 MTP | 513,405,536 |
| [MiniKimi-K3](docs/models/minikimik3.md) | KDA / MLA / LatentMoE / AttnRes + MoonViT + MTP | 204,526,216 |
| [MiniDeepSeek-V4](docs/models/minideepseekv4.md) | SWA128 / CSA-HCA / hash-MoE / mHC + 文本 MTP | 243,983,472 |

MiniFrontier1.0 是本项目提出的融合架构，不是任何一家官方模型的缩小复刻或官方认可配方。主配置为 16 层、hidden 512、32K 词表、32 个 routed experts / top-4、latent 256、一个全宽 shared expert；视觉塔与主干均随机初始化。输入支持文本、图片、多图与带时间戳的视频帧，输出为文本/结构化调用。完整配置和来源映射位于 [configs/minifrontier1](configs/minifrontier1)。

原三个对照模型采用 `Mini` + 官方模型/架构名；MiniQwen4 对应发布源码中的 `qwen4_exp`，DeepSeek 以 V4-Flash 为具体来源。它们的默认配置和正在运行的冻结源码实验不因新增融合模型而迁移。

**MiniFrontier1.0 当前是可运行、已做 CPU 正确性验证的研究实现。** 三通路、媒体边界、缓存回滚、MTP、视觉梯度和精确恢复已有测试；完整 228M 配置已完成 CPU 随机图像前后向。离线小配置跑通了训练、恢复、indexer、稀疏续训与 SFT。**尚未完成 20M/200M 架构比较、3090 各阶段性能验收、3B 主预训练或对话能力验收。** reference 注意力包含 Python 调度与显式投影，不能据参数大小承诺 8K 多模态训练速度或显存。

小配置另完成了 600 步学习探测：消费 9,404 CE token、CPU 两线程约 242 秒。留出色块图像/视频题分别答对 4/4、2/2，置黑后降为 1/4、0/2；留出算术为 0/6，说明存在明显过拟合。这些结果仅验证视觉路径可以参与学习，详见[实验报告与可重画曲线](docs/experiments/mf1-reference-v2/README.md)。

原三模型的策略配置位于 `configs/strategies`，表中均包含 MTP；DeepSeek 对照按旧方案先训练文本，再迁移原生 Vision-Exp。

原三个模型仍在进行各自的 20M-token 配方试验，诊断不计入正式主预算，见[原方案执行记录](docs/audits/strategy-implementation-v2.md)。MiniFrontier1.0 另建训练系列、tokenizer、数据清单及阶段门禁；旧模型的试验通过不能替代融合架构验收。

**2026-09-08 效果审计：`educational-v1` 未达到基本对话目标。** 阶段完成和损失下降不能作为模型可用的证据；SFT 已出现重复、答非所问，DPO 也未修复。见[失败复盘与纠正措施](docs/training-failure-v1.md)。当前权重用于排查与学习，不标记为可用对话模型。

## 开始使用

**v0.1.0 研究预览版尚未公开发布。** “MiniFrontier1.0”是模型系列名称，不表示软件已发布 1.0 正式版。

Python 3.11+。在 Git checkout 中安装依赖后，运行完全离线的微型示例：

```bash
uv sync --locked --extra dev
uv run minifrontier models
uv run minifrontier mf1 params
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 quickstart --device cpu --output outputs/mf1-quickstart
uv run minifrontier mf1 generate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --prompt '1+2=' --max-new-tokens 12 --temperature 0 --device cpu
```

示例离线生成文本、图片与时序色块视频，使用约 132K 参数、四层且包含全部通路的 CPU 小配置。它验证训练与推理链路，**不产出可用聊天模型，也不计入 3B 主预算**。数据来源、阶段命令、显式诊断 Demo、恢复及导出见 [MiniFrontier1.0 指南](docs/guides/minifrontier1.md)。原三个模型的离线示例仍可用 `minifrontier quickstart --model all` 运行，见[原最小示例](docs/guides/quickstart.md)。

wheel 支持最小示例、模型清单、显式配置 acceptance 训练与 CLI 推理；正式策略训练首版要求 Git checkout。浏览器默认只展示能力验收通过的检查点，当前没有合格聊天权重，quickstart 不会产生默认可选模型。见[版本范围、状态与路线图](docs/releases/v0.1.0.md)。

当前 checkout 尚未配置正式仓库地址、文档/Issues URL 和私下反馈渠道；维护者需在公开发布前补齐。

## 项目目录

```text
minifrontier/models/       融合模型与三条来源架构；每个模型独立子目录
minifrontier/mf1/          MF1 数据、训练、评估、后训练、导出和 Demo
minifrontier/training/     来源模型训练器及共用损失、优化器、工具环境
configs/                  模型清单、容量配置；mf1 配方在 minifrontier1/
scripts/                  数据提取、实验调度、状态查询与分发验收
tests/                    数值、训练、缓存、来源与安装回归
docs/guides/              通用操作指南
docs/models/              结构与能力状态
docs/training-strategies/  按日期冻结的原始方案
docs/experiments/          可分享的轻量实验档案和曲线
docs/audits/               验收及历史检查记录
docs/operations/           本机调度记录
docs/releases/            分发范围和发布准备
docs/legacy/              历史设计与失败对照
third_party/upstream/     固定的官方源码快照
LICENSES/                 各上游组件许可正文
```

详细职责见[架构与目录](docs/architecture.md)、[脚本导航](scripts/README.md)。`data/`、`outputs/`、环境、缓存和构建文件是本地产物，不进入 Git；新增可分享结果应整理到 `docs/experiments/`。历史文档保留原日期与证据边界。

## 训练流程

MiniFrontier1.0 按[融合架构方案](docs/training-strategies/2026-09-09/04-MiniFrontier1.0-原生多模态融合架构与全流程实现方案.md)独立推进：

```text
词表/数据与数值验证 → 20M / 200M 消融
→ P0 200M CE → P1 800M CE → I 40M input（只训练 indexer）
→ P2 1.4B CE → P3 600M CE → SFT 120M assistant CE
→ 可验证 RL → 最多 8 个合格领域/模式教师 → 全词表 reverse-KL OPD
→ 可选偏好 → MTP 派生 draft → 可选量化 → 导出与 Demo 验收
```

主 CE 合计 **3B**，其中视觉文本监督目标 **630M**；indexer、MTP、rollout、SFT 另记账。默认单任务单卡，其他 GPU 用于独立消融/教师。AdamW 为 reference，Muon 为待比较候选。阶段启动核对实际初始化权重、词表、处理器、数据和评测文件的 hash；精确恢复保留优化器、路由统计、数据游标和 RNG。尚未达标的阶段不能仅凭更新次数进入下一阶段。

以下是保留的三个来源模型的独立路线：

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
目前仅作正确性路径，尚无训练后加速结论，见[草稿适应与推理](docs/guides/draft-adaptation.md)。
新的 control-v1 模板贯通 SFT、原生视觉/工具 rollout 和 CLI；实际行为概率、策略
权重 hash、工具观察掩码与终态奖励分别保留。此接口尚未完成正式后训练，见
[后训练适应与范围](docs/guides/posttraining-adaptation.md)。
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
