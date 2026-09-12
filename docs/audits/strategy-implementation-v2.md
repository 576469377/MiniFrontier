# 来源模型方案执行记录（2026-09-08—09）

本轮接通了三个来源模型的原生视觉、MTP、专用优化器、阶段迁移与草稿训练。短程诊断证明模型能够记忆训练题，并在简单色块任务中依赖图像；三个模型的留出算术均失败。`educational-v1` 权重继续作为失败对照。

设计原文见[训练方案](../training-strategies/README.md)，阶段预算在 `configs/strategies/*-plan.json`。下文为当时记录；三个来源模型后于 2026-09-11 开始正式预训练，当前安排见[主计划](../pretraining-plan.md)。

## 实现与数值检查

| 范围 | 本轮完成的实现与检查 |
|---|---|
| 训练计数 | 实际 CE/input/media 账本，整个累积窗口与 DDP 的有效 token 分母 |
| 优化器 | 各自 Muon、路由更新、Kimi QKClip，视觉/aligner 独立 LR |
| 模型迁移 | DeepSeek Text→Vision、V1 冻结文本、显式 QAT/MTP 系数切换 |
| 数据 | 不可变编码、分组去重、相同字节的 tokenizer 对比、磁盘写入保护 |
| 缓存 | Kimi/DeepSeek 增量与回滚；Qwen 原生视觉 prefill 位置和回滚 |
| 量化仿真 | Kimi expert MXFP4 / activation MXFP8；DeepSeek expert MXFP4、Hadamard indexer QK FP4 / BF16 scores；E2M1/E8M0、STE、MTP BF16 CUDA 反向 |
| 后训练 | SFT→MOPD/OPD 轨迹、全词表 KL、response 分母、零优势跳过与按 domain/effort 选教师 |
| 工具 | 隔离的 seccomp 代码 worker、本地检索、内存文件、状态修改与 Python 工具；工具观察不计动作损失 |
| 草稿 | 七步 Kimi LK、Qwen 四流 CE/三步自回馈、三阶段 DSpark；冻结目标、精确恢复、导出绑定目标 hash |

新配置采用方案默认的 64K 词表；32K 的等墙钟质量对照尚未完成。Kimi block scale 为本地量化仿真配方，当时没有 5M SFT 校准结果，也没有九/十二个合格领域教师。

缓存的严格 FP32 对照通过；BF16 逐元素绝对误差断言最初失败。路由改用 FP32 后，运算次序仍产生舍入差异。[三种子报告](cache-precision-v2.json)显示：缓存 RMSE 低于完整 BF16 相对 FP32 的 RMSE，贪心一致率均为 1。DeepSeek indexer 另采用并列时低索引优先的稳定选择规则。

GPU rollout 曾强制 BF16，与 FP32 调用者的概率分布不一致。修复后，微型原生视觉测试的 Qwen/DeepSeek FP32 ratio error 为零，Kimi CUDA KDA 约 0.000286；BF16 最大约 0.001468。更新前的本地误差上限分别为 CPU FP32 `2e-5`、CUDA FP32 `0.001`、低精度 `0.02`，详见[原始对照](native-rollout-precision-v2.json)。

实验性批量专家 GEMM 在完整 Kimi BF16 微基准中得到梯度 cosine **0.6575**、relative L2 **0.8345**，未通过数值检查。因此本轮训练保留专家循环；实验路径的 route padding 超过四倍时回退，保留全部 token。

## 数据与学习诊断

首批公开试验数据为 **44,695 条**，PT train 编码后 **34,576,919 CE**。第二批扩展到 **52,887 条、45,490,020 CE**，新增 8,192 个由整数/有理数不变量核验的合成数学文档；另有 96 张带原始字节与来源许可的 ALLaVA 图像。数学科学、真实视觉/视频、SFT 与独立验证规模仍不足完整方案。

独立诊断数据包括 256 张生成图像、2,048 条算术文本。相似图像归为同一训练组，视觉检查使用训练内样本；图片模板按有序 RGB 身份区分，纯文本首问按规范化问题聚合、最多三个答案。

| 诊断 | CE / updates | 训练内算术 | 留出算术 | 正确图 / 错图 / 遮图 |
|---|---:|---:|---:|---|
| Kimi K0 | 500,450 / 576 | 16/16 | 0/9 | 16/16、0/16、8/16 |
| DeepSeek D0 | 500,735 / 468 | 16/16 | 0/9 | 本表无分项记录 |
| Qwen 首轮 Q0 | 500,450 / 576 | 14/16 | 0/9 | 16/16、0/16、8/16 |
| Qwen 续诊断 | 新增 500,450 / 576；累计 1,000,900 CE | 16/16 | 0/9 | 16/16、0/16、8/16 |

上述运行均通过结束时的 DDP 参数逐项一致检查。Qwen 首轮验证 LM NLL 为 **1.157929**；训练内算术未达到 16/16，控制器停止。续诊断从该权重另建优化器，LR=1e-4、Muon=0.003、20K warmup/WSD；全局 batch 保持 32，每卡 microbatch/累积由 4/4 改为 16/1。其算术与视觉结果见[算术报告](qwen-q0-arithmetic-1m.json)和[视觉报告](qwen-q0-visual-1m.json)。达到本轮记忆条件后，Qwen 的 20M Muon 配方实验从随机初始化启动。

## 性能与后训练范围

三个长度 64 的诊断均完成 50 次预热＋200 次测量。Kimi/DeepSeek 的 长度 512 的配方测量同样采用 50＋200，分别约 **561.38 / 516.84 CE token/s**。主机开发负载会影响耗时；记录只支持相应配置的优化更新估计，验证和存盘另计。

草稿推理实现了正残差拒绝重采样、全部接受后的 bonus、EOS 和原生状态回滚。CPU 对照覆盖每个拒绝位置及三种原生图像 prefill；DSpark 权重分母与有效位置预算跨累积窗口/DDP 汇总。当时草稿前缀仍重算，没有正式接受率、校准、自适应调度或端到端加速结果，使用说明见[草稿训练](../draft-adaptation.md)。

`control-v1` 将 reasoning/final/tool/effort 编码用于训练、rollout 和推理；轨迹绑定实际策略 hash，保存工具输出、行为概率、终止和奖励分项。教师需匹配模板与 tokenizer，恢复后重新在线采样。相关实现见[后训练说明](../posttraining-adaptation.md)。

## 当时的运行与后续工作

三个模型分别使用 GPU 0–1、2–3、4–5；GPU 6、7 的既有任务保持原状，工作盘预留 50 GiB。训练源码为 `75a3364`，控制器另记内容 hash；run 保存数据、tokenizer、配置、账本、RNG、游标和优化器状态。配方曲线在本地 6007，旧实验在 6006。

本轮尚缺正式数据与完整配方对照、SFT/QAT 学习验证、领域教师、工具/多模态 RL、正式草稿训练及生成评价。完成执行路径和预算后，仍须分别检验这些阶段的训练效果。
