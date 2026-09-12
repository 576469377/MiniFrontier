# 三个来源模型的配方实验（2026-09-10）

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md)

**已完成的 Kimi、DeepSeek 对照支持保留 Muon 候选，学习率和 MTP 的收益仍需联合验证。** Qwen 的 AdamW 及部分 MTP 对照在采集时尚未结束。

采集时间：**2026-09-10 07:51:38 UTC**。本快照含 16 项已启动的 20M CE 配方实验和 4 项早期诊断：配方实验中 13 项完成、3 项运行，另有两项 Qwen LR 对照待启动。全部为 `acceptance`，不计正式主训练预算。

同一模型的验证 NLL 用于比较配方；训练 CE 与最后一次验证可能来自不同步数，应按 JSON 中的 validation step 对齐。后续选择见[工作参数](../2026-09-10-pretraining-cutover/working-recipes.md)，当前安排见[预训练计划](../../pretraining-plan.md)。

## 当时的选择依据

| 对照 | 已观察到的结果 | 对配方的含义 |
|---|---|---|
| Kimi 优化器 | 等 20M CE，Muon 验证 NLL 5.2567，AdamW 5.4632 | Muon 是较好的候选，仍需补种子验证 |
| DeepSeek 优化器 | 等 20M CE，Muon 5.3346，AdamW 5.4981 | 同样支持保留 Muon 候选 |
| Kimi Muon LR | 0.005 为 5.0802，0.01 为 5.2561 | 当前预算下更低的 LR 更好 |
| DeepSeek LR | 1e-4 为 6.0566，3e-4 为 5.3335 | 这次减小 LR 明显变差 |
| Kimi MTP | 系数 0 为 5.2893，0.2 为 5.2151 | 0.2 值得进入后续联合对照 |
| DeepSeek MTP | 系数 0 为 5.3342，0.1 为 5.3308 | 差异很小，尚不足以作稳定收益结论 |
| Qwen 优化器 | 同第 1000 步：Muon 5.2306，AdamW 5.8484；AdamW 尚未完成预算 | 有候选倾向，等待完整结果 |

这些结果支持保留 Muon 和相应 LR 候选。单独降低 LR、提高 MTP 的收益不能相加推断联合配置；未完成的联合对照与补种子是本组实验的局限。首版随后依据已有结果选择工作参数并开始预训练。

不同模型、tokenizer、单/双卡和共卡时段的 NLL、吞吐不作直接排名。每组 JSON/CSV 保留实际配置、命令、来源和计数；`${WORKSPACE}` 需绑定本地目录，数据按相应版本重建。

## 可重画曲线

![优化器对照的验证曲线](validation.svg)

图中为双卡 Muon/AdamW 对照，横轴是优化器更新次数，纵轴是验证 LM NLL。每个面板只比较同一模型；Qwen AdamW 的轨迹在采集时尚未完成。图中不包含下表全部 LR/MTP 变体，这些曲线可从对应 CSV 的累计 CE 或 input token 轴重画：

```bash
uv run --no-project --with matplotlib==3.10.7 python scripts/plot_experiments.py \
  docs/experiments/2026-09-10-recipe-snapshot
```

## 完整实验明细

展开查看原精度的数值。实验名链接到 JSON，CSV 保存逐次曲线；`running` 表示采集时仍在运行。前四项为独立学习诊断，其 NLL 不与后面的 20M 配方结果混比。

<details>
<summary>20 条实验记录：配方 16 条、早期诊断 4 条</summary>

| 实验与报告 | 采集时状态 | 累计 CE | 最近验证 LM NLL | 测得 CE/s |
|---|---|---:|---:|---:|
| [DeepSeek · 首轮诊断](strategy-diagnostics-v2--minideepseekv4.json) ([CSV](strategy-diagnostics-v2--minideepseekv4.csv)) | complete | 500735 | 1.452930539449056 | 163.5091131107459 |
| [Kimi · 首轮诊断](strategy-diagnostics-v2--minikimik3.json) ([CSV](strategy-diagnostics-v2--minikimik3.csv)) | complete | 500450 | 1.1787232716878255 | 135.16564801724988 |
| [Qwen · 首轮诊断](strategy-diagnostics-v2--miniqwen4.json) ([CSV](strategy-diagnostics-v2--miniqwen4.csv)) | complete | 500450 | 1.1579292233784992 | 55.71579888290027 |
| [Qwen · 续诊断](strategy-diagnostics-v2--miniqwen4-extension-1m.json) ([CSV](strategy-diagnostics-v2--miniqwen4-extension-1m.csv)) | complete | 500450 | 1.254478308359782 | 112.72600286013086 |
| [DeepSeek · AdamW · 20M](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.csv)) | complete | 20013084 | 5.498143784134394 | 526.3428746014318 |
| [DeepSeek · Muon · 20M](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.csv)) | complete | 20013084 | 5.334631086140065 | 516.8445481960501 |
| [Kimi · AdamW · 20M](strategy-recipe-pilots-v2--minikimik3--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--adamw-20m.csv)) | complete | 20010912 | 5.463168602161199 | 570.1488719772458 |
| [Kimi · Muon · 20M](strategy-recipe-pilots-v2--minikimik3--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--muon-20m.csv)) | complete | 20010912 | 5.25667633755248 | 561.3834668387981 |
| [Qwen · AdamW · 20M](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--adamw-20m.csv)) | running | 19212637 | 5.848418437576233 | 244.4018053020977 |
| [Qwen · Muon · 20M](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.csv)) | complete | 20012550 | 5.017851688746724 | 287.222375773371 |
| [DeepSeek · MTP 0](strategy-shared-gpu-v2--minideepseekv4--mtp-0.json) ([CSV](strategy-shared-gpu-v2--minideepseekv4--mtp-0.csv)) | complete | 20013084 | 5.334226843876047 | 503.4105322542753 |
| [DeepSeek · MTP 0.1](strategy-shared-gpu-v2--minideepseekv4--mtp-0.1.json) ([CSV](strategy-shared-gpu-v2--minideepseekv4--mtp-0.1.csv)) | complete | 20013084 | 5.330807047457807 | 466.06439897155093 |
| [Kimi · MTP 0](strategy-shared-gpu-v2--minikimik3--mtp-0.json) ([CSV](strategy-shared-gpu-v2--minikimik3--mtp-0.csv)) | complete | 20010912 | 5.289296183661418 | 548.8928905376886 |
| [Kimi · MTP 0.2](strategy-shared-gpu-v2--minikimik3--mtp-0.2.json) ([CSV](strategy-shared-gpu-v2--minikimik3--mtp-0.2.csv)) | complete | 20010912 | 5.215082979486988 | 508.55441988661386 |
| [Qwen · MTP 0](strategy-shared-gpu-v2--miniqwen4--mtp-0.json) ([CSV](strategy-shared-gpu-v2--miniqwen4--mtp-0.csv)) | running | 19546239 | 5.260053761240519 | 209.2897203502699 |
| [Qwen · MTP 0.2](strategy-shared-gpu-v2--miniqwen4--mtp-0.2.json) ([CSV](strategy-shared-gpu-v2--miniqwen4--mtp-0.2.csv)) | running | 18041550 | 5.235745998607435 | 196.8388169293865 |
| [DeepSeek · 低 LR](strategy-single-gpu-v2--minideepseekv4--lower-lr.json) ([CSV](strategy-single-gpu-v2--minideepseekv4--lower-lr.csv)) | complete | 20013084 | 6.0565902774399945 | 475.1350875442292 |
| [DeepSeek · 参考 LR](strategy-single-gpu-v2--minideepseekv4--reference.json) ([CSV](strategy-single-gpu-v2--minideepseekv4--reference.csv)) | complete | 20013084 | 5.333452095412703 | 474.58811979007544 |
| [Kimi · 低 LR](strategy-single-gpu-v2--minikimik3--lower-lr.json) ([CSV](strategy-single-gpu-v2--minikimik3--lower-lr.csv)) | complete | 20010912 | 5.080175087209892 | 528.1002065176235 |
| [Kimi · 参考 LR](strategy-single-gpu-v2--minikimik3--reference.json) ([CSV](strategy-single-gpu-v2--minikimik3--reference.csv)) | complete | 20010912 | 5.256071488071545 | 531.151021157216 |

</details>

---

[返回实验索引](../../experiments.md) · [当前训练安排](../../pretraining-plan.md) · [后续工作配方](../2026-09-10-pretraining-cutover/working-recipes.md)
