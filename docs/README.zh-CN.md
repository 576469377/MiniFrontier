# MiniFrontier 中文入口

项目在固定官方源码上缩小容量，提供消费级 GPU 可实践的文本训练链路。

- [项目首页](../README.md)：模型命名、能力与快速开始。
- [训练指南](training.md)：数据、单卡/双卡、阶段切换、断点恢复、GRPO/MOPD、demo。
- [项目审查与整理](project-review.md)：原有基础、发现的问题、改动与尚未完成项。
- [架构与目录](architecture.md)：代码职责和固定源码边界。
- [模型清单](../configs/models.json)：唯一模型目录与能力状态。

模型显示名统一为 **MiniQwen4、MiniKimi-K3、MiniDeepSeek-V4**，Python 包名使用无连字符小写形式。

当前可执行的是文本教学训练。原生视觉、MTP、量化感知训练及完整旗舰效果复现仍待完成；不能将可启动训练与完整官方训练复现混为一谈。
