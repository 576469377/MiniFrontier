# MF1 语言诊断与性能实验调度

> 历史执行记录：此页保留当时的设备安排和候选值。当前配方、暂停/停止决定与新任务顺序以[实验总计划](../experiments/current-plan.md)为准。

2026-09-10 的本机执行记录。通用配方和复现入口见[实验档案](../experiments/mf1-language-performance-v1/README.md)。编号只表示本次设备安排。

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

两组 SFT 各 30K CE，前两次更新后强制退出、重载，之后继续同一预算。训练完成后自动执行 32 条训练问答和全部 138 条留出样本的生成检查。通过训练题记忆并不等于泛化通过。已配置条件队列：训练问答精确匹配至少 29/32、EOS 至少 30/32、视觉留出正确至少 28/32 时，该组在原卡上继续 100K CE 的扩展生成指令诊断，否则停止扩展。快照采集时，两组均等待控制结果，未启动 100K；不会自动进入 2M 架构对照或正式训练。

本地学习产物为 `outputs/mf1-language-sft-v1/`，性能为 `outputs/mf1-performance-v2/`，调度日志为 `outputs/services/mf1-next-v1/`。TensorBoard 使用既有 6007 服务，新增 `mf1-language/` 分组。两组使用相同版本的独立代码目录，版本信息见[实验运行快照](../experiments/mf1-language-performance-v1/sft-running-snapshot.json)。

## TensorBoard 分组

本轮首先将 MF1 日志整理为 `train / eval / perf`，训练 CE 与验证 CE 分母分开显示，保留原始 step 和 wall time。历史视图使用 6007 端口，原日志与检查点保持原样。

后续四个模型统一采用相同分类，正式预训练使用独立的 6008 服务。同步器的注册表、事件模式、启动命令及日志保护统一维护在[实验管理](experiment-management.md#tensorboard)，本页不再保留另一套操作说明。
