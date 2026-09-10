# 脚本导航

在仓库根目录执行脚本。日常 MF1 数据、训练、评估、导出和 Demo 使用 `minifrontier mf1`，入口见[操作指南](../docs/guides/minifrontier1.md)。下表区分融合模型与三个来源模型的实验工具。

| 用途 | 脚本 |
|---|---|
| 提取和核对固定上游来源 | `extract_training_sources.py`、`extract_vision_sources.py` |
| 从冻结方案生成原三模型机器计划 | `build_strategy_plans.py` |
| MF1 完整配置的短程 GPU 资源探测 | `profile_mf1.py`；不作为完整性能验收 |
| MF1 候选词表与机制数据、受资源保护的学习试验 | `prepare_mf1_mechanism_data.py`、`run_mf1_trial.py`；见[实验记录](../docs/experiments/mf1-gpu-mechanism-v1/README.md) |
| 来源模型容量与机制检查 | `accept_miniqwen4_capacity.py`、`launch_strategy_diagnostics.py` |
| 原三模型的配方、单卡与共卡试验 | `run_recipe_pilot.py`、`run_single_gpu_queue.py`、`run_shared_gpu_queue.py`、`run_shared_gpu_trial.py` |
| 读取训练进度、等待设备可用 | `training_status.py`、`wait_for_gpus.py` |
| 检查点、算术、视觉与专家执行评估 | `evaluate_checkpoints.py`、`evaluate_arithmetic_diagnostics.py`、`evaluate_visual_diagnostics.py`、`benchmark_expert_execution.py` |
| 整理实验数值和重画曲线 | `export_experiments.py`、`plot_experiments.py` |
| 全部实验台账与当前计划核对 | `experiment_registry.py`；读取 `configs/experiments.json`，见[实验管理](../docs/operations/experiment-management.md) |
| 新单卡队列与 batch 性能短筛 | `run_exclusive_gpu_queue.py`、`benchmark_training_batch.py`、`run_batch_frontier.py`；短筛不能自动晋级长期配方 |
| 安装后的 wheel 验收 | `check_installed_wheel.py`；按脚本说明在 checkout 外运行 |
| 历史 educational-v1 启动器 | `launch_training.py`；保留兼容与复盘，不执行新融合方案 |

调度记录和冻结执行约定见 [operations](../docs/operations/local-training.md)。正在运行的任务按自己的源码副本和输出路径继续；更新开发目录不会自动迁移它们。

新实验统一使用独占队列。`run_single_gpu_queue.py` 与 `run_shared_gpu_queue.py` 保留历史复现和共用的检查函数，不作为新的调度方式；新的设备安排写 JSON，不再复制一份版本化调度脚本到源码树。完整研究取舍见[当前计划](../docs/experiments/current-plan.md)。
