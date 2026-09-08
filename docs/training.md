# 训练与 demo

`educational-v1` 的基本对话效果验收失败，见[复盘](training-failure-v1.md)。本文的短步数命令演示接口，不能用于估算学会语言所需的训练量。

## 环境与模型

```bash
uv sync --locked --extra dev --extra training --extra monitoring
uv run minifrontier doctor
uv run minifrontier models --count-backbones
```

安装 CPU 开发依赖不强制需要 FLA；Kimi CUDA KDA 使用 `training` extra 中固定的 FLA 0.5.2。运行时不下载官方权重。首次 CUDA KDA 调用可能需要编译内核。

## 构造数据

```bash
uv run minifrontier prepare-data --output data/educational-v1 \
  --pretrain-rows 60000 --sft-rows 30000 --dpo-rows 10000 \
  --vocab-size 65536 --sequence-length 256 --seed 42
```

处理器记录 Hub revision，默认 `--sampling reservoir` 读取完整源文件并做确定性的均匀蓄水池抽样，避免只取文件开头的题材偏置；记录完整文件 hash、原始行数、样本 hash 和抽样种子。它仍然是样本集，不是全量语料训练。读取完整文件有相应下载耗时。复现旧工程实验可显式使用 `--sampling prefix`，不应将前缀视为代表性语料。最终目录已完成时拒绝覆盖，扩大语料请使用新目录。也可在 Python 中调用 `prepare_data(..., local_sources={"pretrain": ..., "sft": ..., "dpo": ...})` 接入自己的 JSONL；本地来源仍按指定行数读取前缀。

预训练格式为 `{"text":"..."}`；SFT 为 `{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`；偏好数据为 `chosen` / `rejected` 两份完整消息列表。SFT 仅监督 assistant 内容与结束标记，问题和 padding 标签为 -100。DPO 分别求 chosen/rejected completion 的 log probability 总和。

产物包括 `manifest.json`、`tokenizer.json`、清洗后的 JSONL、预训练 int32 token 流和 SFT/DPO mmap 数组。预训练流用相邻块重叠一个 token，避免块边界漏掉 next-token 目标；不同文档以 EOS 分隔，未施加文档间隔离注意力。固定长度截断会丢失长回复尾部，manifest 记录保留下来的监督量。

当前首批产物：预训练训练集 7,627,201 token；验证集 158,065 token；SFT 29,449 条训练对话，3,906,250 个有效监督 token；DPO 8,735 对训练偏好。源数据卡许可是 Apache-2.0 / CC-BY-NC-2.0，保持其使用与署名条件。

## 单卡与双卡

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run minifrontier train \
  --model minikimik3 --data data/educational-v1 \
  --output outputs/minikimik3/example/pretrain --steps 1000 \
  --batch-size 2 --grad-accum 2 --sequence-length 256

# 两卡运行相同模型
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=2 uv run torchrun --standalone \
  --nproc_per_node=2 -m minifrontier.training.train \
  --model minikimik3 --data data/educational-v1 \
  --output outputs/minikimik3/two-gpu/pretrain --steps 1000
```

模型/优化器为 FP32，CUDA 前向采用 BF16 autocast 与 activation checkpointing。Kimi/DeepSeek 当前默认 AdamW 是本地教学优化选择；Qwen 默认走已有语义分块 Muon/AdamW。学习率、warmup、负载平衡更新率等不声称来自未公开官方配方。

## 自动运行三个模型

```bash
uv run python scripts/launch_training.py --run coverage-example \
  --gpu-groups 0,1 2,3 4,5 --batch-size 2 --grad-accum 2 \
  --pretrain-epochs 1 --sft-epochs 1 \
  --save-every 100 --eval-every 100 --eval-batches 64
uv run python scripts/training_status.py --run coverage-example
```

控制器脱离当前终端运行。预训练和 SFT 分别必须显式指定 `--*-steps` 或 `--*-epochs`，二者互斥；epoch 根据当前长度、数据量和全局 batch 换算为向上取整的更新次数，并将预计样本数、覆盖轮次写入 recipe。一轮命令仅演示预算方式，不保证模型可用。DPO 默认 0 步；应先检查 SFT 的留出集和真实生成，再单独安排偏好训练。Qwen/DeepSeek 中间增加 100 步索引器蒸馏和 200 步稀疏 CPT。基础/SFT/DPO 长度 256，索引器阶段长度 1,024，超过 Qwen 的 512 token 索引预算，实际触发稀疏选择。每步全局样本数 = 单卡 batch × 卡数 × 梯度累积。

训练期验证从整个留出集均匀抽取固定样本，不再固定取前几行；`--eval-batches 64` 为每 rank 最多 64 个样本，`--eval-batches 0` 为完整留出集。偏好阶段额外记录 `preference_accuracy`。样本抽取不会消耗训练 RNG。

对已完成的 SFT/DPO 做完整验证集和固定 greedy 提示对比：

```bash
CUDA_VISIBLE_DEVICES=2 uv run python scripts/evaluate_checkpoints.py \
  --model minikimik3 --device cuda:0 \
  --output docs/audits/minikimik3-capability-v1.json
```

报告包含 551 条 SFT 验证样本、172 对偏好验证样本（取决于实际数据集）、token 加权 NLL、偏好指标、真实生成和权重 SHA256。脚本不会按 loss 自动判定“能正常对话”。

启动前检查 GPU 是否被占用，不干预其他任务。每模型控制器持有文件锁，阶段失败立即停止。目录结构为：

```text
outputs/<model>/<run>/
  recipe.json                   固定阶段计划与 GPU 组
  pipeline.json                 控制器状态、当前阶段与 PID
  controller.log
  pretrain/                     其余阶段结构相同
    run.json / status.json       配方、进度
    train.log / metrics.jsonl   文本日志、结构化指标
    tensorboard/                训练与验证曲线
    checkpoint.pt               可精确恢复的训练状态
    model.pt                    阶段完成后导出的纯模型载荷
    tokenizer.json
```

短验收目录为 `acceptance-v1`；它们不出现在浏览器模型列表和教学 TensorBoard 中。

## 阶段与恢复

```bash
# 阶段切换：只载入权重，重新创建优化器和调度
uv run minifrontier train --model minikimik3 --data data/educational-v1 \
  --stage sft --init outputs/minikimik3/example/pretrain/model.pt \
  --output outputs/minikimik3/example/sft --steps 500

# 控制器中断后，以原 recipe 继续；已完成阶段跳过
uv run python scripts/launch_training.py \
  --controller outputs/minikimik3/educational-v1/recipe.json
```

手工恢复时使用原命令并加 `--resume <该阶段/checkpoint.pt>`，移除 `--init`。总步数、学习率、长度、batch、累积、卡数、阶段和语料必须与检查点一致；改变配方请用新的输出目录和 `--init`，不将其称为精确恢复。`--resume` 包含各 rank RNG 与全局样本游标。阶段最终保存前逐项核验 DDP 参数一致。

Qwen/DeepSeek 阶段顺序为 `pretrain → dense_distill → sparse_cpt → sft → dpo`。indexer-only 蒸馏冻结主干；SFT/DPO 保留已学到的稀疏选择，冻结离散索引器，不再混入索引 KL。

## GRPO 与 MOPD

```bash
uv run minifrontier prepare-rl-data --output data/arithmetic-v1 --count 10000
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m minifrontier.training.train --model minikimik3 --data data/educational-v1 \
  --stage grpo --rl-data data/arithmetic-v1 --group-size 4 --rollout-tokens 32 \
  --init outputs/minikimik3/example/sft/model.pt \
  --output outputs/minikimik3/example/grpo --steps 100
```

GRPO 在线采样，每题多回复，用可验证整数结果给奖励，组内标准化 advantage，使用 clipped policy ratio 和冻结参考策略 KL。全部回复都错误时，组内奖励无法提供改善信号；日志中的奖励需结合模型已有能力判断。此入口不声称复现工具代理环境或完整官方 RL 系统。

MOPD 将 `--stage` 改为 `mopd`，加 `--teacher-map path/to/teachers.json`。映射例如：

```json
{
  "arithmetic:low": "/absolute/path/to/teacher-low/model.pt",
  "arithmetic:high": "/absolute/path/to/teacher-high/model.pt"
}
```

要求至少两个不同本地检查点、相同 tokenizer，覆盖所有任务的 domain/effort。教师文件 SHA256 纳入恢复配方，替换教师后不能继续声称精确恢复。学生在线生成，教师在同一前缀评估 token，使用 Kimi K3 报告公式 15 的 stop-gradient、裁剪 log-ratio 奖励。没有教师时不会伪造教师或调用付费 API。真实领域教师质量和训练投入需要自行提供。

## 监控与 demo

```bash
uv run tensorboard --logdir_spec \
  MiniQwen4:outputs/miniqwen4/educational-v1,MiniKimi-K3:outputs/minikimik3/educational-v1,MiniDeepSeek-V4:outputs/minideepseekv4/educational-v1 \
  --host 127.0.0.1 --port 6006
uv run minifrontier demo --root outputs --device cuda:0 --port 7860

uv run minifrontier generate \
  --checkpoint outputs/minikimik3/example/sft/model.pt \
  --device cuda:0 --prompt '请简单解释什么是大语言模型。'
```

demo 发现每个模型最新的教学检查点。预训练/稀疏 CPT 采用续写，SFT 后采用对话模板。训练尚不足时输出可能乱码、重复或不遵循指令；工程验收不等于已训练出可用聊天能力。远程服务器可用 SSH 转发 6006/7860 端口。
