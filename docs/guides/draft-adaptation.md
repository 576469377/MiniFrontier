# 草稿训练与投机推理

草稿入口已实现，但三个正式目标模型和 10M–30M 草稿训练尚未完成。
当前 demo 仍以通过能力验收的主模型为准；单元测试与随机小模型的双卡更新
不能证明草稿接受率、语言能力或端到端加速。

| 模型 | 目标冻结后的训练路径 | mini 结构与适配 |
| --- | --- | --- |
| MiniKimi-K3 | 温度 1 的七步自回馈 LK，无附加 CE | AttnRes 第 3/7/11 层输出；融合初始化 `[0,0,I]`；复制训练 MTP，转换为草稿局部残差 |
| MiniQwen4 | 目标回答上的未来 token CE，加三步自回馈 CE | 保留真实四流 hidden、共享 H→H 投影、GR/MoE/QSA 和独立 MTP head；无 PLE |
| MiniDeepSeek-V4 | 0.1 CE + 0.9 全词表 L1 + confidence BCE；gamma=4 | 三阶段、block5、末三层 mHC 均值 taps、rank64 Markov、noise id20；三阶段从训练后的文本 MTP 独立初始化 |

Kimi 的转换、Qwen 的训练课程和 DSpark 的 mini 初始化都是明确的本地配方。
DSpark confidence 的监督是分布 overlap 软标签。它的损失分母是有效位置衰减
权重之和，预算单独累计整数有效位置；跨 microbatch/DDP 分别汇总。
Kimi LK 用 log-space overlap，避免概率下溢后失去梯度。

`train-draft` 读取相同冻结词表的不可变 SFT 编码，只取首个回答之前的 prompt
及其完整原生图像/视频。回答由冻结目标重新采样，原语料答案不传给草稿目标。
Kimi/Qwen 的未来特征自回馈，DSpark 的噪声 backbone 不接触未来 token；
Markov head 仅接触前一个 token。EOS 终止监督，预算截断不会增加 EOS 标签。
实际部署建议使用经过来源、领域、模式和视觉覆盖审核的 prompt 池。

下面是未来选定最终目标后的小型适应试验命令，**不是目前后台训练的阶段**：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m minifrontier train-draft \
  --target /path/to/selected-final/model.pt \
  --data /path/to/audited-native-sft \
  --output outputs/draft-pilot \
  --draft-positions 10000 --sequence-length 1024 \
  --rollout-tokens 64 --grad-accum 4 --run-kind acceptance
```

省略 torchrun，改用 `.venv/bin/python -m minifrontier train-draft` 即为单卡入口。
默认本地 AdamW LR=1e-4，norm/bias 无 decay，按有效草稿位置 warmup/cosine；
需要按方案进行 LR、模态和长度试验。正式执行另需 `--run-kind strategy`、
`--strategy-plan`、`--strategy-phase`、`--strategy-evidence` 和 `--config`。
门禁核验前置阶段的精确目标权重、10M–30M 预算、冻结词表、数据审阅、实际
attention phase/卡数及对应 profile。现有数据与目标尚不能通过这些门禁。

输出仅保存草稿，不重复保存整份冻结目标：

- `checkpoint.pt`：草稿、优化器、各 rank RNG、数据游标、独立位置账本和未完成的 profile。
- `draft.pt`：最后草稿；`best-draft.pt`：同一留出集上草稿目标最低的版本。
- `run.json`、`metrics.jsonl`、`performance.json`：目标/词表/数据/源码 hash，实际
  有效位置和损失分母，以及 50 次预热后 200 次更新的时间、I/O、图像/帧、显存。

`--resume` 必须复用同一输出目录和配方；`--steps` 是安全上限，先到上限只标记
`incomplete`。目标权重不会因草稿训练改变；阶段完成时逐项检查 DDP 参数一致。
写入使用共享磁盘锁、原子替换及默认 50 GiB 保留线。

```bash
.venv/bin/python -m minifrontier generate \
  --checkpoint /path/to/selected-final/model.pt \
  --draft outputs/draft-pilot/best-draft.pt \
  --draft-steps 3 --prompt '请介绍一下自己。' --device cuda:0
```

加载时必须匹配精确目标文件 SHA256。当前投机入口限 batch1、温度 1、top-p 1，
支持完整原生视觉 prefill；浏览器默认尚未接入草稿。目标验证后，拒绝分支从
`normalize(max(p-q,0))` 采样，回滚并重放已接受前缀及替代 token。
全部接受时使用目标分布补一个 token；EOS 后的预测也会丢弃并回滚。
三个目标缓存与对应 taps 只保留已接受 token。统计区分目标 anchor 与草稿接受数，
不能用永远接受的 anchor 抬高草稿接受率。

当前实现重算草稿自身的前缀，并在拒绝时重放目标状态，是可核验的正确性路径。
尚未实现优化后的草稿缓存、QSA 索引跨步复用、DSpark 自适应成本调度；也没有
训练后逐位置接受率、confidence calibration、分布和 1/3/5 步延迟结论。
FP32 的全量/增量与回滚已对照；BF16 原生 kernel 的舍入差异需继续按实际目标
验证，不能把采样公式正确等同于逐位数值相同或已经加速。
