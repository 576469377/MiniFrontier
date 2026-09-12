# 单卡独占调度与异机复验（2026-09-10）

**08:34 UTC 启动快照。** 本轮在远端七张空闲 RTX 3090 上分别运行一个实验，另一张卡的既有任务保持原状。旧任务的停止与结果见[后续复核](2026-09-10-batch-frontier/README.md)，当前安排见[预训练计划](../pretraining-plan.md)。

| 远端 GPU | 任务 | 配方与预算 | 要回答的问题 |
| --- | --- | --- | --- |
| 0 | MiniQwen4 reference | Muon LR 0.01，seed 42，20M CE | 单卡参考配方的损失和吞吐表现 |
| 1 | MiniQwen4 lower-lr | Muon LR 0.003，seed 42，20M CE | 降低 Muon 学习率是否改善验证损失 |
| 2 | MiniKimi-K3 seed 43 | Muon LR 0.005，20M CE | 候选配方能否在另一随机种子下复现 |
| 3 | MiniKimi-K3 seed 44 | 同上，seed 44 | 候选配方的种子间波动 |
| 4 | MiniDeepSeek-V4 seed 43 | DeepSeek Muon / Adam 共用 LR 0.0003，20M CE | 候选配方能否在另一随机种子下复现 |
| 5 | MiniDeepSeek-V4 seed 44 | 同上，seed 44 | 候选配方的种子间波动 |
| 6 | MF1 steady-512 | 序列 512，microbatch 8，全局输入 4096；20 次预热和 100 次测量 | 单卡独占时优化后实现的持续吞吐 |

快照时七项均已启动：Kimi/DeepSeek 已写出更新，MF1 已写出预热记录。Qwen 原 Muon/AdamW 对照结束，其中 AdamW 为 **20,012,550 CE、1198 updates**；两项远端单卡 Qwen 对照随后启动，`world_size=1`。

**LR 更正：**DeepSeek 的 Muon/Adam 共用 `--lr`，旧命令中的 `--muon-lr 0.01` 未生效，表格按实际参数组记录。

三个来源模型沿用 64K tokenizer、长度 512、global input 16384、microbatch 2、400K CE warmup/WSD 与原 MTP 设置。部署核对 **201 个文件、约 2.34 GiB**；视觉路径重定位后重建字节索引，token、labels、样本、媒体和词表保持。Kimi/Qwen 训练及验证读图检查通过。

两机均为 PyTorch `2.13.0+cu130`，解释器分别为 Python **3.13.5 / 3.13.13**。硬件负载和运行环境差异记录在案，跨主机耗时只作观察。

当时本机 MF1 30K CE 控制实验继续运行，100K 指令扩展等待训练问答、EOS 和视觉检查结果。远端日志约每 30 秒同步到 TensorBoard；来源模型保留训练/验证曲线，MF1 合成测速仅显示性能。全部任务为 `acceptance` 或性能测量。

后续本机 batch16 复验见[测速档案](2026-09-10-batch-tuning/README.md)，调度工具见[操作说明](../operations/exclusive-gpu-queue.md)。

<details>
<summary>复现信息：本轮使用的代码版本</summary>

| 用途 | Git 提交 |
|---|---|
| 三个来源模型训练器 | `75a3364f586936d764c31376e185747973d2bed6` |
| MF1 训练器 | `d77c51c4ba074ba2d325f0b52e2ecceca392972e` |
| 调度器 | `66a4ef1` |

复现还需相应数据、tokenizer 和完整运行参数；原日志、传输清单及环境记录保存在本地。

</details>
