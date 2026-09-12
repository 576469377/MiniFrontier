# 离线示例测量（2026-09-09）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**六组微型示例均完成训练、暂停恢复与验证；固定加法问题没有得到正确答案。** 本页记录三个来源模型在 CPU 和一张共享 RTX 3090 上的安装与训练链路检查，输入为生成算术数据。

## 条件与完成范围

每组预训练和 SFT 各 8 次更新，预训练在第 4 步暂停后恢复；分别消费 224 和 36 个有效 CE token。CPU 使用两线程，GPU 与另一训练进程共用；下表耗时包含该次运行及内核初始化。如此短的任务主要反映启动与执行开销。

模型使用微型配置，按当时报告的 Parameter 元素计数为 Kimi 74,066、DeepSeek 79,255、Qwen 32,596；DeepSeek 的计数包含固定整数路由项。验证为 16 条样本、48 个监督 token；生成统一提问 `What is 10 + 2?`，正确答案应为 `12`。

## 实测结果

| 模型与原始报告 | 设备 | 耗时（秒） | 末次验证 LM NLL | 实际生成 |
|---|---|---:|---:|---|
| [Kimi](minikimik3-cpu.json) | CPU | 7.79 | 5.6107 | 空输出 |
| [Kimi](minikimik3-3090.json) | RTX 3090 | 108.17 | 5.6174 | 空输出 |
| [DeepSeek](minideepseekv4-cpu.json) | CPU | 5.68 | 5.4715 | 空输出 |
| [DeepSeek](minideepseekv4-3090.json) | RTX 3090 | 10.62 | 5.4718 | 空输出 |
| [Qwen](miniqwen4-cpu.json) | CPU | 3.72 | 5.7011 | `8` |
| [Qwen](miniqwen4-3090.json) | RTX 3090 | 9.29 | 5.7139 | 乱码，含重复 `11` |

显示值经四舍五入；报告保留完整精度、实际生成文本和每次验证记录。CPU/GPU 的一次性耗时包含不同的初始化与共用负载，不能用这张表推断稳定训练加速比。流程完成与生成正确分别记录：本次六组都完成了预定流程，但没有答对该问题。

## 如何查看证据

每个模型链接对应一份 JSON：`commands` 保存运行命令，`resume_executed` 和阶段状态记录暂停恢复，`evaluations` 保存验证轨迹，`generation` 保留空输出或错误答案。`data_sha256`、`config_sha256` 和 tokenizer 校验值用于核对输入。

首次运行按[入门指南](../../guides/quickstart.md)准备离线示例；复查旧结果时使用报告绑定的配置和输入。当前正式训练使用不同规模、数据与预算，安排见[预训练计划](../../pretraining-plan.md)。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [运行离线示例](../../guides/quickstart.md)
