# MF1 语言诊断与性能实验调度

本页记录 2026-09-10 的语言诊断和性能测量。实验结果见[语言诊断档案](../experiments/mf1-language-performance-v1/README.md)，当前训练安排见[预训练主计划](../pretraining-plan.md)。

两组 SFT 各用 30K CE，主干 LR 分别为 2e-5 和 1e-4；前两次更新后退出、重载，再继续原预算。完成后检查 32 条训练问答及全部 138 条留出样本。扩展条件为训练问答精确匹配至少 29/32、EOS 至少 30/32、视觉留出正确至少 28/32。

两组后来通过控制条件并启动 100K CE 扩展，扩展运行因吞吐过低而停止，未完成预算；见[停止记录](../experiments/2026-09-10-batch-frontier/README.md#处理决定)。[07:51 更新快照](../experiments/2026-09-10-mf1-update/README.md)保留扩展启动前的采集状态。

<details>
<summary>历史设备、资源与输出目录</summary>

| GPU | 任务 | 显存控制 |
|---:|---|---|
| 0、1 | 已有 Qwen MTP 两组实验 | 保持原队列 |
| 2、3 | 已有 Qwen AdamW 双卡及后续 LR 队列 | 保持原队列 |
| 4 | MF1 228M 基础问答 SFT，LR 2e-5 | allocator 上限 6 GiB，物理余量至少 5 GiB |
| 5 | MF1 独占性能短筛与持续测量 | allocator 上限 21 GiB，每次更新检查物理余量至少 2 GiB |
| 6 | MF1 同初始化、同数据 SFT，LR 1e-4 | 与既有进程共卡；allocator 上限 5.5 GiB，另留约 1 GiB context 预算，物理余量至少 2 GiB |
| 7 | 既有其他任务 | 不加入本轮调度 |

GPU 6 的显存上限按启动时约 9 GiB 的空闲量设置。首次实际 SFT 更新后，该卡仍空闲约 4.4 GiB；既有占用保留。两组都是独立单卡，6 号卡上的速度不能标记为独占性能。

训练由 `scripts/run_mf1_trial.py` 启动，监督器只管理自己创建的子进程。显存或磁盘余量不足时停止自己的工作进程，保留最近滚动检查点。启动需磁盘至少 66 GiB，监督器在低于 60 GiB 时停止；原子写入统一保留 50 GiB。新建产物暂按不超过 40 GiB 控制，不保存逐步权重。

本地学习产物为 `outputs/mf1-language-sft-v1/`，性能为 `outputs/mf1-performance-v2/`，调度日志为 `outputs/services/mf1-next-v1/`。TensorBoard 使用既有 6007 服务，新增 `mf1-language/` 分组。两组使用相同版本的独立代码目录，版本信息见[实验运行快照](../experiments/mf1-language-performance-v1/sft-running-snapshot.json)。

</details>

## TensorBoard 分组

本轮首先将 MF1 日志整理为 `train / eval / perf`，训练 CE 与验证 CE 分母分开显示，保留原始 step 和 wall time。历史视图使用 6007 端口，原日志与检查点保持原样。

四模型现已统一采用相同分类，正式预训练使用独立视图。同步器的注册表、事件模式与命令见[实验管理](experiment-management.md#tensorboard)。
