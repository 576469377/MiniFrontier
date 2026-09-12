# 三个来源模型的草稿训练与投机推理

草稿模型先提出候选 token，再由主模型验证。本页介绍 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4 的草稿训练、目标绑定与接受/拒绝采样；MF1 见[独立指南](minifrontier1.md#后训练与草稿)。当前已验证训练更新和采样逻辑，尚无完整草稿训练或实际加速结果。

## 结构与训练目标

| 模型 | 冻结目标后的损失 | 结构与初始化 |
|---|---|---|
| MiniKimi-K3 | 温度 1 的七步自回馈 LK，无附加 CE | 取第 4/8/12 层 AttnRes 输出（0 起算索引 3/7/11）；融合初始化 `[0,0,I]`；复制已训练 MTP，改为草稿局部残差 |
| MiniQwen4 | 目标回答的未来 token CE，加三步自回馈 CE | 真实四流 hidden、共享 H→H 投影、GR/MoE/QSA 和独立 MTP head；无 PLE |
| MiniDeepSeek-V4 | 0.1 CE + 0.9 全词表 L1 + confidence BCE，gamma=4 | 三阶段、block 5、末三层 mHC 均值 taps、rank 64 Markov、noise id 20；每阶段从已训练文本 MTP 独立初始化 |

以上转换和课程为本地配方。Kimi LK 使用 log-space overlap，避免概率下溢。DSpark confidence 使用分布 overlap 软标签；损失除以有效位置的衰减权重和，训练预算另按整数有效位置累计，microbatch 和 DDP 分别汇总两者。

`train-draft` 读取同一冻结词表的 SFT 编码，只取首个回答前的 prompt 及完整原生媒体。冻结目标重新生成回答，原语料答案不进入草稿目标。Kimi/Qwen 自回馈未来特征；DSpark 噪声 backbone 不读未来 token，Markov head 只接触前一个 token。EOS 终止监督，预算截断不添加 EOS。

## 训练与恢复

先选定目标检查点，并准备覆盖来源、领域、模式和视觉任务的 prompt 池。下面是单卡小规模运行示例，需替换本地路径：

```bash
CUDA_VISIBLE_DEVICES=0 uv run minifrontier train-draft \
  --target /path/to/selected-final/model.pt \
  --data /path/to/audited-native-sft \
  --output outputs/draft-pilot \
  --draft-positions 10000 --sequence-length 1024 \
  --rollout-tokens 64 --grad-accum 4 --run-kind acceptance
```

双卡入口为 `CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 -m minifrontier train-draft`。默认 AdamW LR=1e-4，norm/bias 不做 weight decay，按有效位置执行 warmup/cosine。

正式训练还需 `--run-kind strategy`、`--strategy-plan`、`--strategy-phase`、`--strategy-evidence` 和 `--config`，绑定前驱权重、预算、词表、数据与执行配置。当前首阶段预训练尚未提供通过资格评估的最终目标；后续状态见[预训练计划](../pretraining-plan.md)。

| 产物 | 内容 |
|---|---|
| `checkpoint.pt` | 草稿、优化器、各 rank RNG、数据游标、独立位置账本及未完成的性能采样 |
| `draft.pt` / `best-draft.pt` | 最后草稿 / 同一留出集上草稿损失最低的版本；均不重复保存冻结目标 |
| `run.json` / `metrics.jsonl` | 目标、词表、数据与源码 hash，实际有效位置及损失分母 |
| `performance.json` | 完成配置窗口后的性能统计；默认排除前 50 次更新，再采样 200 次 |

`--resume` 复用同一输出目录和配方。`--steps` 是更新上限；先达到上限时阶段为 `incomplete`。训练不修改目标权重，阶段完成时检查 DDP 参数一致性。写入使用磁盘锁、原子替换和默认 50 GiB 保留线；短诊断无需为填满性能窗口继续训练。

## 投机采样

```bash
uv run minifrontier generate \
  --checkpoint /path/to/selected-final/model.pt \
  --draft outputs/draft-pilot/best-draft.pt \
  --draft-steps 3 --prompt '请介绍一下自己。' --device cuda:0
```

草稿必须匹配目标文件的精确 SHA256。当前入口限 batch 1、温度 1、top-p 1，支持原生视觉 prefill；浏览器尚未接入草稿。

目标验证候选后，拒绝分支按 `normalize(max(p-q,0))` 采样，回滚并重放已接受前缀及替代 token。全部接受时由目标补一个 token；EOS 后预测丢弃并回滚。缓存及 taps 只保留已接受 token，接受率单独统计草稿候选，不计必然接受的目标 anchor。

当前实现重算草稿前缀，并在拒绝时重放目标状态。尚未实现草稿缓存优化、QSA 索引跨步复用和 DSpark 自适应调度；逐位置接受率、confidence 校准及 1/3/5 步延迟仍待测量。FP32 全量/增量与回滚已有对照，BF16 仍需对具体目标核验原生 kernel 的舍入差异。
