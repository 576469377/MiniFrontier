# 实验登记、调度与归档

执行安排见[预训练主计划](../pretraining-plan.md)，公开结果见[实验索引](../experiments.md)。`configs/experiments.json` 登记比较组与运行目录；各机器的队列计划保存实际命令、输入身份和资源约束。

每次运行以主机标签和相对输出目录生成稳定的 `exp-*` 编号。父试验、实际训练阶段及合成测速分别登记；它们可以属于同一个比较组，台账条数不等于 GPU 任务数。台账自动读取原始运行记录，更新时不会启动训练或修改配方。

纯推理运行以 `inference_evaluation` 登记，台账分类为 `evaluation`，保留检查点、生成 token 数和审核记录的绑定。旧检查点的训练计数不会计入这次评估的 CE、优化器更新或正式训练预算。数据或评估进程的身份在本次读取中已核实存活时，不因其等待事件、长时间未改写记录而标成过期；缺失或不匹配的进程仍明确显示未核实。

```bash
# 四个模型的最新正式进度；历史诊断使用 --run strategy-v2
python -m scripts.training_status --run formal
# 输出 outputs/experiment-registry/current.json 和 current.md
python -m scripts.experiment_registry
# 持续刷新台账；数据准备的完成回调使用下文的事件接续
python -m scripts.experiment_registry --watch 30
# 导出一次不可覆盖的公开台账快照（只含元数据）
python -m scripts.export_experiments --workspace "$PWD" \
  --output docs/experiments/YYYY-MM-DD-review --registry-only
```

正式状态命令结合最新 JSONL 和本地训练进程显示进度，优先于保存检查点时的滞后状态。没有进程可见性时报告 `unknown`；ETA 仅涵盖当前阶段的更新时间，验证、存盘和后续阶段另计。

截至 2026-09-12，状态脚本尚未把 MF1 的 `budget_complete_unqualified` 映射为完成：进程退出后可能显示 `stopped`。遇到该状态，应结合 `recorded_state`、实际 token 账本和检查点确认预算完成情况；能力是否通过仍须单独判断。

重复刷新受文件锁保护。台账包含来源模型、MF1、历史诊断、本机队列及已经同步的远端记录；不会扫描实验代码目录。迁移后遗留的空白等待记录合并到实际远端运行，已产生结果的独立运行保留各自编号。缺少完成证据或记录过期时显示 `unverified` / `unverified_stale`，不能根据目录存在推断正在训练。

## TensorBoard

四个模型统一使用 `train / eval / perf`。训练计数与验证分母分开，常规和阶段末验证分别显示。启动、QK 裁剪及缓存回收等事件保留在 JSONL，不重复画成训练曲线；具体指标见[展示说明](../audits/training-infrastructure.md#tensorboard-展示)。

[`sync_mf1_tensorboard.py`](../../scripts/sync_mf1_tensorboard.py)兼容四个训练器，从原始 JSONL 和 event 文件生成独立视图，保留 step 与 wall time。脚本名称为兼容已有服务保留。注册表为以下对象的列表，`source` 指向训练目录，`publish` 可选，用于建立展示链接：

```json
[
  {
    "name": "MiniKimi-K3/K1",
    "source": "outputs/strategy-base-pretraining-v1/minikimik3/K1",
    "publish": "outputs/tensorboard-pretraining-v1/MiniKimi-K3/K1"
  }
]
```

```bash
# 在独立环境安装监控依赖，避免训练期间同步共用 .venv
UV_PROJECT_ENVIRONMENT=outputs/envs/monitoring uv sync --locked --extra monitoring
# 每次启动指定新的视图目录；源训练目录及其 tensorboard 子目录须已存在
UV_PROJECT_ENVIRONMENT=outputs/envs/monitoring uv run --no-sync python scripts/sync_mf1_tensorboard.py \
  --registry /path/to/registry.json --output outputs/tensorboard-grouped-new \
  --watch --event-driven --interval 5
# 仅在端口空闲时启动；已有服务沿用其记录的启动命令
UV_PROJECT_ENVIRONMENT=outputs/envs/monitoring uv run --no-sync tensorboard \
  --logdir outputs/tensorboard-pretraining-v1 \
  --host 127.0.0.1 --port 6008 --reload_interval 5
```

Linux 事件模式等待文件写入，`--interval` 合并短时间内的更新。普通 `--watch` 为兼容的轮询模式。半行 JSON 等待写完后读取；源日志被截断或替换时停止并报错，需新建视图。原始日志、事件和检查点均不改写。切换视图时重启 TensorBoard 可清除其内存中的旧标签，训练进程继续运行。

<details>
<summary>维护环境的展示目录与端口</summary>

2026-09-12 的正式训练视图为 `outputs/tensorboard-pretraining-v1`，端口 `6008`；`6007` 为策略探测，`6006` 为早期教学训练。服务命令保存在 `outputs/services/tensorboard-pretraining-v1/service.json`，同步器配置保存在 `outputs/services/pretraining-tensorboard-grouped-v1/`。复现时按自己的目录和空闲端口配置。

</details>

## 启动前记录什么

正式产物放在 `outputs/strategy-base-pretraining-v1/<model>/<phase>/`。数据构造、测速与运行检查另行登记；它们的计数不加入正式训练预算。

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

需要中断时，先暂停相关后续派发，再核对训练进程身份和恢复点。支持 `pause.request` 的单卡训练器可在输出目录收到该文件后完成当前更新、保存检查点并退出；恢复前检查 `status.json`、检查点和进程退出状态。固定源码目录可能使用较早版本，操作前核对该版本是否支持请求文件。

截至 2026-09-12，来源模型 DDP 仍由各 rank 独立检查 `pause.request`，暂停判定尚未同步，可能导致不同 rank 进入不同流程。该文件不能视为已验证的 DDP 安全暂停方式。直接终止进程也不保证新增检查点，只能恢复到最后成功保存的位置。具体续训参数见[来源模型指南](../guides/training.md#保存恢复与评估)和 [MF1 指南](../guides/minifrontier1.md)。

中断记录 `interruption.json` 保存最新已记录指标、停止原因和恢复点情况。更换 batch 后重新初始化的试验保留独立身份和 token 账本。

## 历史 batch 筛选与当前训练计划

当前工作参数见[执行配方](../experiments/2026-09-10-pretraining-cutover/working-recipes.md)。早期 batch 扫描和停止原因保存在[边界与重排记录](../experiments/2026-09-10-batch-frontier/README.md)。

在队列目录放置 `dispatch-pause.json` 可阻止支持该功能的控制器派发新任务，已有 worker 继续运行。旧冻结控制器需核对身份后停止控制器并保留 worker。恢复时使用符合当前安排的计划，保留此前的暂停与停止记录。

<details>
<summary>历史 batch 选择器与临时权重规则</summary>

此前三个模型先测固定合成输入的 16/32/64/128，再进行 1M CE 的真实数据对照。选择器要求来源、数据、tokenizer、seed、学习率和输入目标一致，16/32 对照均完成，并具有足够测量次数和有限的验证损失。它选择实际吞吐距最快档不超过 5% 的最小 batch。

这一选择仅用于最多 1M CE 的执行确认。**2026-09-10 复核已取消按吞吐自动派发 20M CE 的规则**：旧训练器的大 microbatch 会扩大实际全局 batch，有限的短测损失不足以选择长期配方。已有自动派发结果保留原始身份并标为探索证据；新长训必须来自独立登记的全局 batch/LR 计划。不同模型的 LR/batch schedule 依据见总计划。

历史 1M CE 测速预先声明临时权重：完成后先记录本次目录内权重的文件名、大小和 SHA-256，再清理这些权重。配置、验证结果、数值曲线及 TensorBoard 保留；既有检查点与 20M CE 配方产物按原规则保留。当时磁盘保留 50 GiB，显存保留 2 GiB；当前正式训练的空间规则以[主计划](../pretraining-plan.md#resources)及冻结运行配置为准。

</details>

## 目录与记录边界

- 新队列使用 `run_exclusive_gpu_queue.py`，设备分配和依赖写入计划 JSON；旧 single/shared 调度器用于历史复现。
- `experiment_registry.py` 观测运行并核对用途；`export_experiments.py` 导出数值快照。台账的 `audit_findings` 标注来源、数据身份缺失及实际 batch 越界等问题，缺失证据保留为未知。
- 当前安排维护在 `docs/pretraining-plan.md`，日期目录保存历史快照，`docs/operations/` 说明执行方法。运行中的源码和产物目录保持原位，新的代码整理在独立工作副本进行。

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

2026-09-11 的数据准备使用该入口替代定时刷新，部署与失败处置见[启动档案](../experiments/2026-09-10-pretraining-cutover/execution.md)。当前任务状态从相应控制目录读取；主机地址、原始样本和回调实例留在被忽略的 `outputs/`。
