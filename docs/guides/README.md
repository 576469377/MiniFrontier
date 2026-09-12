# 操作指南

- [MiniFrontier1.0](minifrontier1.md)：融合主线的离线多模态示例、数据、训练、后训练、导出及 Demo。
- [来源模型最小示例](quickstart.md)：MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4 的 CPU / 3090 的数据、训练与生成流程。
- [数据来源](data-sources.md)：各实验实际使用的数据、处理方法、版本及许可。
- [来源模型训练](training.md)：三个来源模型的策略入口和阶段迁移。
- [来源模型后训练](posttraining-adaptation.md)、[草稿适应](draft-adaptation.md)：目标与执行边界。
- [实验 Demo](demo-experiments.md)：观察三个来源模型的独立试验检查点。

命令默认在 Linux 的仓库根目录运行，需要 Python 3.11+；Windows 请使用 Linux 环境（如 WSL2）。当前文件锁与工具执行依赖 Linux/POSIX 接口，未提供原生 Windows 或 macOS 的完整验收。安装和 CPU 示例见[项目首页](../../README.md#快速开始)。

GPU 路径还需 `uv sync --locked --extra training`；TensorBoard 需 `--extra monitoring`，公开语料准备需 `--extra data`。按所用功能增加 extra，锁文件不会因追加这些选择而改变。

需要实验调度、监控和存储安排时，阅读 [operations](../operations/local-training.md)；完整文档分类见[文档导航](../README.md)。
