# 实验登记、调度与归档

日常入口：[当前实验计划](../experiments/current-plan.md)、[公开实验索引](../experiments.md)。计划内容只在 `configs/experiments.json` 维护；每台机器的执行队列是冻结后的命令实例，不另写一份科学配方。

每次运行以主机标签和相对输出目录生成稳定的 `exp-*` 编号。父试验、实际训练阶段及合成测速分别登记；它们可以属于同一个比较组，台账条数不等于 GPU 任务数。台账自动读取原始运行记录，更新时不会启动训练或修改配方。

```bash
# 输出 outputs/experiment-registry/current.json 和 current.md
python -m scripts.experiment_registry
# 可选的传统轮询模式；本轮数据准备已使用下文的事件接续
python -m scripts.experiment_registry --watch 30
# 导出一次不可覆盖的公开台账快照（只含元数据）
python -m scripts.export_experiments --workspace "$PWD" \
  --output docs/experiments/YYYY-MM-DD-review --registry-only
```

重复刷新受文件锁保护。台账包含来源模型、MF1、历史诊断、本机队列及已经同步的远端记录；不会扫描冻结源码 checkout。迁移后遗留的空白等待记录合并到实际远端运行，已产生结果的独立运行保留各自编号。缺少完成证据或记录过期时显示 `unverified` / `unverified_stale`，不能根据目录存在推断正在训练。

## 启动前记录什么

正式预训练使用独立的 TensorBoard：本机端口 `6008`，日志视图为
`outputs/tensorboard-pretraining-v1`。`6007` 保留策略探测实验，`6006` 保留更早的训练记录。
服务状态和启动命令记录在 `outputs/services/tensorboard-pretraining-v1/service.json`。

```bash
# 仅在 6008 未运行时启动；远程访问可转发该端口
uv run tensorboard --logdir outputs/tensorboard-pretraining-v1 \
  --host 127.0.0.1 --port 6008 --reload_interval 5
```

正式运行写入 `outputs/strategy-base-pretraining-v1/<model>/<phase>/`。核对正式运行身份后，
在日志视图中按 `<model>/<phase>` 建立对应事件目录的链接，保留原始事件文件。
数据构造、测速和准入检查继续登记在实验台账；正式训练首次写入事件前，6008 的 run 列表为空。


| 信息 | 保存位置和用途 |
| --- | --- |
| 目的、比较组、改变的变量 | `experiment.json` 或队列计划；解释该次运行要回答的问题 |
| 源码、数据、tokenizer、配置及 runner 校验值 | 队列 `source` / `inputs`、训练 `run.json`；启动前校验冻结输入 |
| 命令、seed、batch、实际 token 预算 | 队列计划和 `run.json`；比较时核对实际 token 账本 |
| 前置结果 | `result_gates`；未完成则等待，失败或主动停止则阻断后续任务 |
| GPU 和存储上限 | 单卡独占队列；同一物理卡共用设备锁，检查实际显存和磁盘余量 |
| 产物保留规则 | `retention`；短测速的临时权重与长期配方权重分别声明 |
| 曲线入口 | 按 cohort / 主机 / 模型 / variant 命名的 TensorBoard 目录 |

新增实验放在一个 cohort 目录内，例如 `outputs/strategy-batch-frontier-gpu-v1/trials/<model>/<variant>/`。每次尝试使用独立输出，旧命令、旧曲线和停止原因保留在原位置。正在运行的源码与目录不搬迁。

## 状态与结论

| 状态 | 含义 |
| --- | --- |
| `waiting_results` / `waiting_gpu` / `waiting_disk` | 已登记，尚未启动对应工作 |
| `running` | 记录显示该次运行正在执行；同时关注更新时间 |
| `complete` | 该次声明的执行过程完成；单独检查 token 预算和能力状态 |
| `stopped_by_user` | 因调整计划主动停止，保留中断记录；不能算完成 |
| `stopped_for_review` / `paused_for_review` | 因实验复核停止工作或暂停后续派发；记录具体原因和恢复条件 |
| `memory_boundary` | 某档 batch 在声明的显存限制下不可行；属于测速结果 |
| `failed` / `blocked_by_result` | 执行异常或前置失败，需要先处理原因 |
| `budget_complete_unqualified` 等 MF1 状态 | 保留训练器原始语义，不改写成能力通过 |

一个 batch 扫描可以正常完成并包含 OOM 档位；这只表示找到了该配置的边界。长期配方运行只有实际 token 预算完成后才算完成，OOM 不能替代成功。台账会报告重复活跃输出或同卡多个活跃任务，设备独占仍由队列锁执行。

主动停止时，先停用相关自动派发，核对进程身份和现有恢复点，再终止指定任务。`interruption.json` 保存最新已记录指标、停止原因和恢复点情况。没有新检查点时，内存中的训练状态不会被描述为已保存；更换 batch 后的新初始化试验也不能把旧试验的部分 token 合并进来。

## batch 筛选与训练计划分开

三模型先在同一台机器上测固定合成输入的 16/32/64/128，再完成 1M CE 的真实数据对照。选择器要求来源、数据、tokenizer、seed、学习率和输入目标一致，16/32 对照均完成，并具有足够测量次数和有限的验证损失。它选择实际吞吐距最快档不超过 5% 的最小 batch。

这一选择仅用于最多 1M CE 的执行确认。**2026-09-10 复核已取消按吞吐自动派发 20M CE 的规则**：旧训练器的大 microbatch 会扩大实际全局 batch，有限的短测损失不足以选择长期配方。已有自动派发结果保留原始身份并标为探索证据；新长训必须来自独立登记的全局 batch/LR 计划。不同模型的 LR/batch schedule 依据见总计划。

在队列目录放置 `dispatch-pause.json` 可阻止新版控制器派发新任务，已有 worker 仍受观察。旧冻结控制器没有该功能，需核对身份后停止控制器并保留 worker；不得修改正在执行的冻结脚本。恢复时创建经过复核的新计划，不能删除记录后自动沿用已经废止的选择规则。

1M CE 测速预先声明临时权重：完成后先记录本次目录内权重的文件名、大小和 SHA-256，再清理这些权重。配置、验证结果、数值曲线及 TensorBoard 保留；既有检查点与 20M CE 配方产物按原规则保留。磁盘始终留出至少 50 GiB，显存留出至少 2 GiB。

本机和远端可用不同的调度器进程，但共用各自机器的物理卡锁。短任务结束后，后续任务按已登记的依赖接续；等待某个候选结果的设备会明确显示等待原因。当前安排和测量见[batch 边界与重排记录](../experiments/2026-09-10-batch-frontier/README.md)。

## 目录与记录边界

- 模型计算在 `minifrontier/models/<model>/`，训练算法在 `minifrontier/training/`；新 batch 控制复用现有训练器和游标，不新增另一套 trainer。
- 新工作站队列统一使用 `run_exclusive_gpu_queue.py`。旧 single/shared 调度器保留供历史复现，不再为新研究增加 `queue_vN.py`；设备分配变化写计划 JSON。
- `experiment_registry.py` 负责观测与用途核对；`export_experiments.py` 保留历史数值导出职责。台账不隐式修改运行配方。
- `docs/experiments/current-plan.md` 解释当前选择，日期目录保存不可覆盖的快照，`docs/operations/` 解释如何执行和管理。历史策略/复盘不混入入门指南。
- 台账中的 `audit_findings` 如实列出旧记录缺少来源/数据身份和实际 batch 越界等问题。缺少的历史 hash 不补造，未知记录不能作为正式准入证据。


## 数据准备的事件接续

`scripts/pretraining_data_events.py` 监听已声明任务的原始 `run.json` 与进程退出，复用现有实验台账。Linux 优先使用 inotify；同用户监听额度不足时使用 dnotify，不调整系统额度。pidfd 跟踪生产进程退出；每 60 秒的心跳仅确认事件连接存活，不定时扫描生产任务。生产任务只更新时间戳时，不派发重复完成回调。

```bash
# observer.json: tasks 列出 id、run，以及可选的小型 files；全部为本机绝对路径
python scripts/pretraining_data_events.py observe --plan /absolute/path/observer.json
# plan.json: 声明本机/SSH observer 命令、元数据路径映射和有界完成回调
python scripts/pretraining_data_events.py run --plan /absolute/path/plan.json
```

控制计划包含 `control`、`workspace`、`wall_seconds`、`observers` 和可选 `hooks` / `on_event`。每个 observer 指定 `id`、`command`、`tasks`；任务指定 `id`、`local_run` 和来源路径到本地路径的 `files` 映射。`mirror: true` 仅用于声明的远端小型元数据。`hooks` 按任务 id 索引，回调声明 `id`、`command`、`cwd`、`timeout_seconds` 和 `input_sha256`，并接收 `--event <path>`。回调返回 0 表示完成，2 表示需要处理，其他值为失败。`on_event` 用于刷新已有台账，不在心跳时执行。

服务状态记录在控制目录的 `run.json`；`actions/` 先保存执行意图再运行回调，同一个完成事件不会因重连或重启而重复启动任务。中断时留下的执行意图需要核对，不能直接删除后重跑。`completion-results/` 是当前部署回调的结果，`needs_attention` 汇总待处理项。连接中断或监听失败会停止控制服务并记录原因，生产任务继续保留自己的记录；脚本不重试生产任务、不分配 GPU、不自动授予数据准入。停止服务只终止它启动的监听器。

本轮本机和辅助主机已部署该入口，取代原每 30 秒刷新台账的常驻观察进程。已有文档编码与过滤任务继续运行；自然图片补充完成后自动校验导出哈希并接续身份/固定 benchmark 检查。新 OCR 的划分冲突和视觉候选进入待处理清单。部署计划、主机地址、原始样本和回调实例保存在被忽略的 `outputs/`；公开仓库保留通用入口与回归测试。
