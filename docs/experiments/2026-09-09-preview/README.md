# 配方实验早期快照（2026-09-09）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**Kimi 和 DeepSeek 的 Muon 对照已完成 20M CE；其余三项配方运行尚未完成。** 本页同时保留四项约 500K CE 的学习诊断，供检查早期曲线与后续配方选择过程。

采集时间：**2026-09-09 08:20:06 UTC**。本页状态和数值保持该时刻；完成后的比较见[9 月 10 日配方档案](../2026-09-10-recipe-snapshot/README.md)。

## 先看验证曲线

![三个来源模型的早期验证曲线](validation.svg)

横轴为优化器更新次数，纵轴为验证 LM NLL，越低越好。各面板对应一个模型及其验证集：Kimi、DeepSeek 展示 Muon/AdamW，Qwen 当时只有 Muon 曲线。未完成的轨迹只画到最后一个已记录点；跨模型的曲线高度不代表能力排名。

这一快照可观察训练进展，但 AdamW 与 Muon 的终点预算尚未全部对齐。约 500K 的诊断使用不同任务与训练条件，也应与 20M 配方实验分开阅读。

## 逐项数值与证据

点击实验名查看完整 ID、命令、配置和输入身份，点击 CSV 查看训练与验证曲线。累计 CE 与最近验证可能来自不同更新，比较时按报告中的验证步数对齐。

| 实验与报告 | 采集时状态 | 累计 CE | 最近验证 LM NLL | 测得 CE/s |
|---|---|---:|---:|---:|
| [DeepSeek · AdamW · 20M](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.csv)) | running | 13901894 | 5.855686939924099 | 526.3428746014318 |
| [DeepSeek · Muon · 20M](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.csv)) | complete | 20013084 | 5.334631086140065 | 516.8445481960501 |
| [Kimi · AdamW · 20M](strategy-recipe-pilots-v2--minikimik3--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--adamw-20m.csv)) | running | 17021901 | 5.632155408497192 | 570.1488719772458 |
| [Kimi · Muon · 20M](strategy-recipe-pilots-v2--minikimik3--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--muon-20m.csv)) | complete | 20010912 | 5.25667633755248 | 561.3834668387981 |
| [Qwen · Muon · 20M](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.csv)) | running | 17873675 | 5.230578570976059 | 287.222375773371 |
| [DeepSeek · 首轮诊断](strategy-diagnostics-v2--minideepseekv4.json) ([CSV](strategy-diagnostics-v2--minideepseekv4.csv)) | complete | 500735 | 1.452930539449056 | 163.5091131107459 |
| [Kimi · 首轮诊断](strategy-diagnostics-v2--minikimik3.json) ([CSV](strategy-diagnostics-v2--minikimik3.csv)) | complete | 500450 | 1.1787232716878255 | 135.16564801724988 |
| [Qwen · 首轮诊断](strategy-diagnostics-v2--miniqwen4.json) ([CSV](strategy-diagnostics-v2--miniqwen4.csv)) | complete | 500450 | 1.1579292233784992 | 55.71579888290027 |
| [Qwen · 续诊断](strategy-diagnostics-v2--miniqwen4-extension-1m.json) ([CSV](strategy-diagnostics-v2--miniqwen4-extension-1m.csv)) | complete | 500450 | 1.254478308359782 | 112.72600286013086 |

## 复现与公开范围

JSON 保存命令、seed、源码及数据/tokenizer 校验值，CSV 保存曲线。复现时将 `${WORKSPACE}` 绑定到工作目录，并按对应版本准备数据；本目录仅分发数值和元数据。重建方法见[实验说明](../../experiments.md)。

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [后续配方结果](../2026-09-10-recipe-snapshot/README.md)
