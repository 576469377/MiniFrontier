# 2026-09-10：单卡 microbatch 测速与配方复验

三条复现架构原来的单卡 microbatch 2 留有较大计算余量。固定每次更新的合成输入后，调到 16 明显提高吞吐；三组真实数据短训也完成了更新和评估。因此，后续新建 recipe pilot 默认使用 microbatch 16，并启动三组 20M CE tokens 的复验。当前结果支持这个执行配置进入进一步验证，尚不能证明完整训练质量。

测试设备为单张 RTX 3090 24 GiB，每张卡同时运行一个测试。PyTorch 为 `2.13.0+cu130`，CUDA 为 `13.0`。三模型固定训练源码 `75a3364f586936d764c31376e185747973d2bed6`；MF1 使用 `d77c51c4ba074ba2d325f0b52e2ecceca392972e`。配置、源码身份、runner 校验值及逐步数据保存在本目录 JSON 中。

## 固定输入的短测速

三模型均使用序列长度 512、seed 42、BF16 autocast 和原有梯度检查点。每次更新输入相同的 32 条随机序列，共 16,384 个输入位置、16,352 个 CE tokens；microbatch 从 2 增至 32 时，累积次数从 16 降至 1。每档重新初始化相同模型，包含前向、反向、梯度裁剪和对应的语义优化器更新。

| 模型 | batch 2 CE/s | batch 4 CE/s | batch 8 CE/s | batch 16 CE/s | batch 32 CE/s | 2 → 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniQwen4 | 359 | 646 | 1,099 | 1,624 | 2,096 | 4.53× |
| MiniKimi-K3 | 729 | 1,484 | 2,712 | 4,441 | 6,571 | 6.09× |
| MiniDeepSeek-V4 | 641 | 1,263 | 2,318 | 4,058 | 5,802 | 6.33× |

每档只有 **1 次预热、2 次测量**。这张表用于筛选 microbatch，未包含真实数据读取、图像、评估或训练器的路由偏置更新，也未达到正式稳态性能验收时长。它不代表端到端训练加速倍数。batch 32 的合成吞吐更高，但 DeepSeek 已使用约 19.23 GiB reserved 显存，真实数据和视觉路径尚未验证，因此本轮先采用 16。

原始记录：[Qwen](miniqwen4-synthetic.json)、[Kimi](minikimik3-synthetic.json)、[DeepSeek](minideepseekv4-synthetic.json)、[可重画 CSV](synthetic.csv)。CSV 中 MF1 的输入预算不同，不能与三模型直接排名。

## 真实数据短训

继续使用已有审核记录和校验值的 64K tokenizer 数据，Qwen/Kimi 包含真实图像；优化器、MTP、路由均衡及训练器的数据读取路径均实际执行。每组目标 80K CE，视觉配额按原 20M 配方比例缩至 4 张次。WSD 和原有 400K CE warmup 保留，因此整个短训都处于学习率预热期。

| 模型 | microbatch | 测得 CE/s | 平均秒/更新 | 峰值 reserved GiB | 实际累计 CE | 更新次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniQwen4 | 16 | 1,213 | 15.35 | 9.77 | 92,650 | 5 |
| MiniKimi-K3 | 16 | 3,473 | 5.30 | 5.41 | 92,360 | 5 |
| MiniDeepSeek-V4 | 16 | 3,271 | 5.72 | 11.54 | 92,567 | 5 |

性能统计为预热 1 次后的 3 次更新，包含数据、前后向、优化器和路由均衡，不包含独立验证及权重保存。CE 总量超过 80K 是因为训练器完成整个累积窗口后才停止。真实样本长度不同，`input_batch_tokens=16384` 是累积窗口的目标下限；加大 microbatch 会改变跨过该下限的位置，不能声称实际每次更新的 token 完全相同。表中统计窗口均为每次更新 3 个 microbatches，完整复验中也可能出现 4 个。

三组都完成了 5 次更新，梯度及验证损失为有限值；仅评估了 8 个样本。这验证了实际训练链路，未验证模型说话能力或长程收敛。Qwen 短训使用 Muon LR 0.003，Kimi 使用 0.005，DeepSeek 使用 Adam 分支 LR 0.0003 / Muon LR 0.01。

逐更新性能：[Qwen](miniqwen4-mb16-real.json)、[Kimi](minikimik3-mb16-real.json)、[DeepSeek](minideepseekv4-mb16-real.json)。对应 `*-mb16-run.json` 保存命令、数据/tokenizer 校验值和完成状态，`*-mb16-metrics.jsonl` 保存训练与验证曲线。[汇总数据](real-summary.json)还包括 IO 和优化器耗时。

平均读取耗时分别约 0.055、0.060、0.008 秒/更新，低于各自计算耗时；Qwen 优化器约占 4.01 秒/更新，仍有优化空间。当前观察支持“小 microbatch 下重复调度开销较大”的解释。异机 batch 2 运行的时间窗口与负载不同，不能用它们和本表相除来报告受控加速比。

## MF1 的不同瓶颈

[MF1 profiler 记录](minifrontier1-synthetic.json)包含一次预热、一次带 profiler 的更新以及一次不带 profiler 的测量。最后一次以 microbatch 1 处理 512 个输入位置，耗时 10.66 秒、47.94 CE/s；这里的更新预算不同于上表。

带 profiler 的算子表累计 Self CPU 为 12.927 秒、Self CUDA 为 736.210 毫秒，出现大量小型 `bmm`、索引、`nonzero` 和复制操作。这是算子累计时间，不能直接换算成墙钟 GPU 利用率。结合 `csa.py`、`qsa_mla.py` 中逐 query 的 Python 循环，下一步应检查批量注意力、索引与专家执行方式。当前尚未完成这部分模型优化，单纯加 batch 不能视为已解决 MF1 的性能问题。

## 已启动的 20M CE 复验

| 模型 | 候选学习率 | seed | microbatch | 预算 | 启动状态 |
| --- | --- | ---: | ---: | ---: | --- |
| MiniQwen4 | Muon 0.01 / Adam 分支 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |
| MiniKimi-K3 | Muon 0.005 / Adam 分支 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |
| MiniDeepSeek-V4 | Muon 0.01 / Adam 分支 0.0003 | 42 | 16 | 20M CE | 已开始权重更新 |

这三项与各自先前的候选配方比较 microbatch 变更。保留相同的源码、数据、tokenizer、学习率候选、seed、总 CE 预算、400K warmup、WSD 和 MTP；按实际 token 账本记录累积窗口差异。Qwen 的 20M 复验采用原 reference 学习率，区别于上面的低学习率短测速。性能档案恢复为 50 次预热、200 次测量，保留完整验证集评估。

三项仍是 `acceptance` 配方试验，不能计入正式主训练预算。[启动后快照](batch16-start.json)记录了实际更新和 token 账本。在运行的旧 batch 2 异机试验保留冻结配置，新建 pilot 默认值的修改不会改变它们。TensorBoard 新增 `batch16/` 和短测速分组 `performance-real-mb16/`。

## 复现与档案范围

通用测速入口为 [`scripts/benchmark_training_batch.py`](../../../scripts/benchmark_training_batch.py)。本轮执行时的原始脚本副本保存在 [runner 快照](benchmark_training_batch.py.txt)，其 SHA-256 与三模型合成报告的 `runner_sha256` 一致；通用入口后来仅补充了优化器类型注解。MF1 runner 可从上述 MF1 源码版本的 `scripts/benchmark_mf1.py` 获取。

队列命令与输入校验值分别保存在 [三模型/MF1 首批测速](strategy-performance-gpu-v1-plan.json)、[DeepSeek 测速](strategy-performance-gpu-v2-plan.json)、[Qwen/Kimi 真实数据短训](strategy-real-batch-gpu-v2-plan.json)、[DeepSeek 真实数据短训](strategy-real-batch-gpu-v3-plan.json)及 [20M 复验](strategy-batch16-gpu-v1-plan.json)。这些是本次执行记录，引用的前置实验和数据需自行准备，不能当作下载后即用的最小示例。公开副本将工作目录替换为 `${WORKSPACE}` 并去除设备唯一标识。

首次真实数据预检使用了冻结训练器不支持的 `--stop-after-updates`，在参数解析处退出，累计更新及 CE 都为 0。随后改用受支持的 CE 预算入口并建立新输出目录；[失败记录](preflight-failures.json)保留这个过程。

本目录只包含配置、数值指标与执行记录，不包含数据样本、tokenizer 文件或模型权重。20M 试验的最终曲线和结论待完成后追加。
