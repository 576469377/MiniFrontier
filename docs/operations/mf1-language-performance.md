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

用户明确允许使用 6 号卡后，按其约 9 GiB 空闲量选择了上述上限。首次实际 SFT 更新后，该卡仍空闲约 4.4 GiB；既有占用保留。两组都是独立单卡，6 号卡上的速度不能标记为独占性能。

训练由 `scripts/run_mf1_trial.py` 启动，监督器只管理自己创建的子进程。显存或磁盘余量不足时停止自己的工作进程，保留最近滚动检查点。启动需磁盘至少 66 GiB，监督器在低于 60 GiB 时停止；原子写入统一保留 50 GiB。新建产物暂按不超过 40 GiB 控制，不保存逐步权重。

两组 SFT 各 30K CE，前两次更新后强制退出、重载，之后继续同一预算。训练完成后自动执行 32 条训练问答和全部 138 条留出样本的生成检查。通过训练题记忆并不等于泛化通过。已配置条件队列：训练问答精确匹配至少 29/32、EOS 至少 30/32、视觉留出正确至少 28/32 时，该组在原卡上继续 100K CE 的扩展生成指令诊断，否则停止扩展。两组目前均等待控制结果，未启动 100K；不会自动进入 2M 架构对照或正式训练。

本地学习产物为 `outputs/mf1-language-sft-v1/`，性能为 `outputs/mf1-performance-v2/`，调度日志为 `outputs/services/mf1-next-v1/`。TensorBoard 使用既有 6007 服务，新增 `mf1-language/` 分组。两组共享同一冻结源码 `d77c51c`；不修改运行中的冻结 checkout。

## TensorBoard 分组

MF1 的新日志统一使用三个一级分组：`train/` 放 LM loss、梯度和累计 token 计数，`eval/` 放验证损失、各领域损失、验证样本数及媒体置黑对照，`perf/` 放吞吐、更新时间和显存。`step` 用作横轴，不再另画一个 step 曲线。训练 CE 累计量与验证 CE 分母分别显示。

运行中的冻结训练器继续写自己的原始日志。`scripts/sync_mf1_tensorboard.py` 从这些 JSONL 记录生成分组视图，并从原 event 文件保留每个点的 step 和 wall time；它还补充 JSONL 中已有但旧 event 文件没有展示的领域验证损失。旧版写到训练曲线中的验证 CE 分母会归回 `eval/ce_tokens`。

视图单独存储，原始事件、JSONL、检查点和冻结源码都不改写。同步器每 5 秒检查新增记录，遇到尚未写完的 JSON 行或尚未刷新到 event 文件的时间点会等待。一次启动要求新的视图目录，避免重复导入；若源日志被截断或替换会停止并报告原因。

注册表是一组 `{name, source, publish}` 对象：`name` 为相对 run 名，`source` 为原训练输出目录，可选 `publish` 为展示入口的符号链接。不存在的后续实验会保持等待，出现日志后自动纳入。示例：

```json
[
  {
    "name": "mf1-language/adamw-lr2e-5",
    "source": "outputs/mf1-language-sft-v1/adamw-lr2e-5",
    "publish": "outputs/tensorboard-strategy-v2/mf1-language/adamw-lr2e-5"
  }
]
```

```bash
uv run python scripts/sync_mf1_tensorboard.py \
  --registry outputs/services/mf1-tensorboard/registry.json \
  --output outputs/tensorboard-mf1-grouped-v1 --watch
```

首次切换旧视图后，TensorBoard 服务需要重载，清除内存里已有的无前缀标签；训练进程不需要重启。后续刷新仍使用 6007 端口与原 run 名。
