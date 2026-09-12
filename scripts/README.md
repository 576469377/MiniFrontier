# 脚本导航

本目录提供 Git checkout 中的观测、调度、诊断和开发工具，命令在仓库根目录执行。日常训练与推理使用 CLI，见[MF1 指南](../docs/guides/minifrontier1.md)或[来源模型最小示例](../docs/guides/quickstart.md)；wheel 的支持范围见[版本说明](../docs/releases/v0.1.0.md#分发支持范围)。

[返回文档导航](../docs/README.md) · [实验管理](../docs/operations/experiment-management.md) · [当前训练计划](../docs/pretraining-plan.md)

## 当前运行与维护

| 用途 | 脚本 |
|---|---|
| 正式训练进度 | [`training_status.py`](training_status.py) 默认显示四模型正式训练；`--run strategy-v2` 查历史诊断 |
| 实验台账与用途核对 | [`experiment_registry.py`](experiment_registry.py) 读取 `configs/experiments.json` |
| 单卡独占队列 | [`run_exclusive_gpu_queue.py`](run_exclusive_gpu_queue.py) 按计划、前驱证据和设备状态派发；见[队列说明](../docs/operations/exclusive-gpu-queue.md) |
| 数据准备的依赖接续 | [`pretraining_data_events.py`](pretraining_data_events.py) 监听完成/退出事件，执行已登记回调 |
| 四模型 TensorBoard 视图 | [`sync_mf1_tensorboard.py`](sync_mf1_tensorboard.py) 从原日志生成独立视图，支持文件事件同步 |
| 等待可用设备 | [`wait_for_gpus.py`](wait_for_gpus.py) |
| 媒体迁移 | [`relocate_native_media.py`](relocate_native_media.py) 重建媒体路径与字节索引，记录迁移前后校验值 |
| 公开数值与曲线 | [`export_experiments.py`](export_experiments.py)、[`plot_experiments.py`](plot_experiments.py) |

状态解释、暂停与恢复、独立监控环境和归档命令见[实验管理](../docs/operations/experiment-management.md)。运行中的实验保持其固定源码目录、配置与产物原位。

正式数据的构造、合并和评测内容排除使用数据模块入口 `python -m minifrontier.data.pretraining`、`python -m minifrontier.data.evaluation`，完整命令见[数据指南](../docs/guides/data-sources.md#在本地构造数据)。

## 诊断与历史复现

按要检查的模型或机制选择工具。诊断和 GPU 短测使用独立预算，结果按各自配置与检查点解释。

| 用途 | 脚本与记录 |
|---|---|
| MF1 完整配置资源短测 | [`profile_mf1.py`](profile_mf1.py) |
| MF1 词表、机制数据与学习试验 | [`prepare_mf1_mechanism_data.py`](prepare_mf1_mechanism_data.py)、[`run_mf1_trial.py`](run_mf1_trial.py)；见[机制试验](../docs/experiments/mf1-gpu-mechanism-v1/README.md) |
| MF1 语言诊断与生成检查 | [`prepare_mf1_language_data.py`](prepare_mf1_language_data.py)、[`check_mf1_language.py`](check_mf1_language.py)；见[语言诊断](../docs/experiments/mf1-language-performance-v1/README.md) |
| MF1 microbatch、长度和模态性能 | [`benchmark_mf1.py`](benchmark_mf1.py) |
| 来源模型容量与机制 | [`accept_miniqwen4_capacity.py`](accept_miniqwen4_capacity.py)、[`launch_strategy_diagnostics.py`](launch_strategy_diagnostics.py) |
| 来源模型配方与历史调度 | [`run_recipe_pilot.py`](run_recipe_pilot.py)、[`run_single_gpu_queue.py`](run_single_gpu_queue.py)、[`run_shared_gpu_queue.py`](run_shared_gpu_queue.py)、[`run_shared_gpu_trial.py`](run_shared_gpu_trial.py) |
| 历史 batch 性能筛选 | [`benchmark_training_batch.py`](benchmark_training_batch.py)、[`run_batch_frontier.py`](run_batch_frontier.py)；选择规则与停止原因见[实验管理](../docs/operations/experiment-management.md#历史-batch-筛选与当前训练计划) |
| 检查点评估 | [`evaluate_checkpoints.py`](evaluate_checkpoints.py)、[`evaluate_arithmetic_diagnostics.py`](evaluate_arithmetic_diagnostics.py)、[`evaluate_visual_diagnostics.py`](evaluate_visual_diagnostics.py) |
| 专家执行性能 | [`benchmark_expert_execution.py`](benchmark_expert_execution.py) |
| educational-v1 启动器 | [`launch_training.py`](launch_training.py)，保留兼容与复盘 |

旧 single/shared 调度器用于历史复现及共用检查函数，新队列使用独占入口。历史设备与命令见[工作站记录](../docs/operations/local-training.md)。

## 开发与分发维护

| 任务 | 脚本与说明 |
|---|---|
| 验收已安装 wheel | [`check_installed_wheel.py`](check_installed_wheel.py)，按脚本说明在 checkout 外运行 |
| 提取固定上游源码 | [`extract_training_sources.py`](extract_training_sources.py)、[`extract_vision_sources.py`](extract_vision_sources.py)；保留来源与归属，见[第三方说明](../THIRD_PARTY_NOTICES.md) |
| 从原始方案生成来源模型计划 | [`build_strategy_plans.py`](build_strategy_plans.py)；原文绑定方式见[方案索引](../docs/training-strategies/README.md#原文与复现) |
