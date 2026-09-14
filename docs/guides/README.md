# 操作指南

先选择模型入口，跑通离线示例，再按任务查找数据、训练和推理说明。

[返回文档导航](../README.md) · [安装与首页示例](../../README.md#快速开始) · [当前训练计划](../pretraining-plan.md)

## 选择模型入口

| 任务 | 指南 |
|---|---|
| 使用 MF1.0 / MF1.1：版本选择、数据、训练、评估、导出与多媒体 Demo | [MF1 全流程](minifrontier1.md) |
| 使用 Kimi、Qwen、DeepSeek-V4：离线 CPU / 单卡示例 | [来源模型最小示例](quickstart.md) |
| 使用 DeepSeek-V4.1：配置与文本训练入口 | [V4.1 模型页](../models/minideepseekv41.md) |

## 数据与后续流程

数据来源与实验 Demo 共用入口；各模型的媒体支持和训练覆盖在模型页中说明。来源模型训练指南区分 V4 与 V4.1，后训练和草稿流程按版本标明支持范围；MF1 的操作在其全流程指南中。

| 任务 | 指南 |
|---|---|
| 核对语料、处理方法、版本及许可 | [数据来源](data-sources.md) |
| 配置来源模型训练、阶段迁移和恢复 | [来源模型训练](training.md) |
| 使用对话模板、多模态 rollout 和工具任务 | [来源模型后训练](posttraining-adaptation.md) |
| 训练草稿并验证投机采样 | [来源模型草稿适应](draft-adaptation.md) |
| 在浏览器比较各版本的实验检查点 | [实验 Demo](demo-experiments.md) |

## 环境与运行约定

命令默认在 Linux 仓库根目录运行，需要 Python 3.11+ 和 uv。Windows 可使用 WSL2；文件锁与工具执行依赖 Linux/POSIX，原生 Windows 和 macOS 尚未完成验收。wheel 与源码安装的差异见[分发范围](../releases/v0.1.0.md#分发支持范围)。

| 功能依赖 | extra | 安装与操作说明 |
|---|---|---|
| CUDA KDA 内核 | `training` | [来源模型训练](training.md#数据与配置)、[MF1 指南](minifrontier1.md) |
| 公开语料准备 | `data` | [数据来源与处理](data-sources.md) |
| TensorBoard | `monitoring` | [独立监控环境](../operations/experiment-management.md#tensorboard) |

多个 extra 可以一起指定。训练主机使用独立开发和监控环境，避免 `uv sync` 调整活跃任务的依赖。调度、状态解释、暂停恢复与存储操作统一见[实验管理](../operations/experiment-management.md)。
