# 单卡独占实验队列

本页说明单卡独占和历史实验计划格式。当前正式阶段及依赖见[预训练主计划](../pretraining-plan.md)，运行登记与暂停方法见[实验管理](experiment-management.md)。

队列采用一张物理 GPU 同时执行一个项目任务的方式。每台机器独立调度，跨主机转移数据与源码时校验输入身份。

设备准入同时检查 GPU 计算进程、空闲显存和设备锁，避免短暂的低利用率被误判为空闲设备。

## 运行入口

在 Git checkout 中准备并冻结队列 JSON，再运行：

```bash
python -m scripts.run_exclusive_gpu_queue --plan /path/to/queue-plan.json
```

历史实验计划记录 `workspace`、`allowed_gpu_ids`、`controller_sha256` 和 `main_budget_eligible: false`。每项任务记录唯一 `id`、`output`、`command`、`source_root`、`source`、`inputs` 的 SHA-256，以及 `memory_gib` 和 `reserve_gib`。命令参数 `{gpu}` 在启动时替换为物理编号，`CUDA_VISIBLE_DEVICES` 使用对应 GPU UUID。各主机可以使用不同数量的卡，编号从 0 开始。

该格式用于配方试验，不计入正式训练预算。正式阶段队列的计划类型、预算声明和前驱证据按[主计划](../pretraining-plan.md)及部署时固定的控制器版本核对，不能仅修改实验计划中的预算标志来启动正式阶段。

队列按顺序等待前置结果及可用 GPU。设备锁在创建 CUDA 上下文之前取得，并由启动的工作进程继承。调度器退出不会终止训练；以相同计划重启时，通过 PID 与启动时间识别原进程，避免重复启动。已有输出不会被覆盖，失败任务需检查原因后显式恢复。

正式训练的磁盘保留量为 80 GiB，完整规则见[主计划](../pretraining-plan.md#resources)。显存准入要求为任务预计用量加 1 GiB 上下文余量和声明的设备保留量。任务启动后立即登记设备占用，避免在下一次 `nvidia-smi` 更新前重复分配同一张卡。

遗留上下文例外只适用于已核实的 GPU UUID 与宿主 PID，仍须满足剩余显存要求。宿主 PID 与容器内 PID 的含义不同，发送信号前须在实际进程命名空间内确认身份。

## 迁移与记录

运行中的任务继续使用原源码、数据顺序和 token 账本。迁移停下来的任务时，保存文件传输校验值、原始和新数据清单的对应关系、运行环境版本与命令。训练清单及媒体记录里的绝对路径可以按主机重定位；使用 [`relocate_native_media.py`](../../scripts/relocate_native_media.py) 重建媒体行的字节索引，并记录清单、媒体文件和索引修改前后的 SHA-256。文本、样本身份、图像、token ID、标签和 tokenizer 保持一致。

主机地址、用户目录、PID 和设备 UUID 保存在本地运维产物中；公开实验档案使用主机标签及归一化路径。合成性能测试单独记账，吞吐不作为能力结论。

<details>
<summary>2026-09-10 的资源默认值与配方衔接</summary>

历史实验队列在磁盘至少有 66 GiB 空间时启动，其中 50 GiB 为保留量；`MINIFRONTIER_MIN_FREE_GIB` 可提高保留量。这些默认值早于当前正式训练的空间规则。

当时双卡对照完成后转入单卡试验，旧双任务共享队列保留供复盘。Qwen 的单卡学习率对照使用 20M CE、16384 输入 token 的全局批次和独立随机初始化。Kimi、DeepSeek 的候选学习率及后续种子复验安排已由[当前工作配方](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)取代。

MF1 的指令扩展先检查 32 条训练语言题和 32 条视觉保留题，门槛为语言正确至少 29 条、输出 EOS 至少 30 条、视觉正确至少 28 条。实际扩展与停止情况见[语言试验记录](mf1-language-performance.md)。

</details>
