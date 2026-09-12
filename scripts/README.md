# 脚本导航

在仓库根目录执行脚本。日常 MF1 数据、训练、评估、导出和 Demo 使用 `minifrontier mf1`，入口见[操作指南](../docs/guides/minifrontier1.md)。下表区分融合模型与三个来源模型的实验工具。

| 用途 | 脚本 |
|---|---|
| 提取和核对固定上游来源 | `extract_training_sources.py`、`extract_vision_sources.py` |
| 从原始方案生成原三模型机器计划 | `build_strategy_plans.py` |
| MF1 完整配置的短程 GPU 资源探测 | `profile_mf1.py`；不作为完整性能验收 |
| MF1 候选词表与机制数据、受资源保护的学习试验 | `prepare_mf1_mechanism_data.py`、`run_mf1_trial.py`；见[实验记录](../docs/experiments/mf1-gpu-mechanism-v1/README.md) |
| MF1 语言诊断数据与生成检查 | `prepare_mf1_language_data.py`、`check_mf1_language.py`；见[语言诊断](../docs/experiments/mf1-language-performance-v1/README.md) |
| MF1 microbatch、长度和模态性能测量 | `benchmark_mf1.py`；短筛与持续测量分别记录 |
| 四模型 TensorBoard 统一分组 | `sync_mf1_tensorboard.py`；从原日志生成独立视图，支持文件事件同步 |
| 数据准备的依赖接续 | `pretraining_data_events.py`；监听完成/退出事件，执行已登记回调 |
| 来源模型容量与机制检查 | `accept_miniqwen4_capacity.py`、`launch_strategy_diagnostics.py` |
| 原三模型的配方、单卡与共卡试验 | `run_recipe_pilot.py`、`run_single_gpu_queue.py`、`run_shared_gpu_queue.py`、`run_shared_gpu_trial.py` |
| 跨主机迁移后的媒体路径和索引重建 | `relocate_native_media.py`；输入内容与迁移前后校验值单独记录 |
| 读取训练进度、等待设备可用 | `training_status.py` 默认显示四模型正式训练；`--run strategy-v2` 查历史诊断。`wait_for_gpus.py` 等待设备 |
| 检查点、算术、视觉与专家执行评估 | `evaluate_checkpoints.py`、`evaluate_arithmetic_diagnostics.py`、`evaluate_visual_diagnostics.py`、`benchmark_expert_execution.py` |
| 整理实验数值和重画曲线 | `export_experiments.py`、`plot_experiments.py` |
| 全部实验台账与当前计划核对 | `experiment_registry.py`；读取 `configs/experiments.json`，见[实验管理](../docs/operations/experiment-management.md) |
| 新单卡队列与 batch 性能短筛 | `run_exclusive_gpu_queue.py`、`benchmark_training_batch.py`、`run_batch_frontier.py`；短筛不能自动晋级长期配方 |
| 安装后的 wheel 验收 | `check_installed_wheel.py`；按脚本说明在 checkout 外运行 |
| 历史 educational-v1 启动器 | `launch_training.py`；保留兼容与复盘，不执行新融合方案 |

当前调度与归档操作见[实验管理](../docs/operations/experiment-management.md)，历史设备安排见[工作站记录](../docs/operations/local-training.md)。长时间运行的实验应使用独立的代码目录，并记录版本、配置和输出位置，便于恢复和核对结果。

新实验统一使用独占队列。`run_single_gpu_queue.py` 与 `run_shared_gpu_queue.py` 保留历史复现和共用的检查函数，不作为新的调度方式；新的设备安排写 JSON，不再复制一份版本化调度脚本到源码树。完整研究取舍见[当前计划](../docs/experiments/current-plan.md)。

正式预训练候选的构造、合并和评测内容排除使用 `python -m minifrontier.data.pretraining`、`python -m minifrontier.data.evaluation`；它们是数据模块入口，命令与限制见[数据指南](../docs/guides/data-sources.md#在本地构造数据)。
