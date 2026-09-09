# 2026-09-08 训练方案执行记录

> 本页保留该方案的实现过程和当时的实验进度。当前概览见[模型说明](../README.md)，报告中的运行状态按记录时刻理解。

本记录区分实现、数值验证与能力验收；旧 educational-v1 权重保留作失败对照。
方案原文在 `docs/training-strategies/2026-09-08`，机器可读预算位于
`configs/strategies/*-plan.json`。正式主预算、SFT、RL、草稿和发布均未完成。

已实现的基础包括三个模型的原生视觉与 processor、训练 MTP、实际 token 账本、
全 accumulation/DDP 分母、各自 Muon 与路由更新、Kimi QKClip、DeepSeek Text→Vision
迁移、不可变数据编码、分组去重、同字节 tokenizer 对比及磁盘写入保护。
新配置使用 64K 冻结词表；32K 尚未完成等墙钟质量对比，64K 是方案默认选择。

Kimi/DeepSeek 增量缓存与回滚已实现，Qwen 已增加原生视觉 prefill 位置与回滚。
FP32 保持严格全前向对照。BF16 的逐元素绝对误差测试最初失败；固定路由为 FP32
后仍有不同计算顺序的舍入误差。`cache-precision-v2.json` 保留三个种子的
FP32/BF16 对照：缓存 RMSE 小于完整 BF16 相对 FP32 的 RMSE，贪心一致率均为 1。
这不是训练后模型的生成质量或草稿接受率验收。
DeepSeek indexer 并列分数采用稳定的低索引优先顺序，以消除前缀长度改变时的选择漂移；
这是显式的本地 tie policy，不能称作原始 topk 未定义并列顺序的逐位复现。

首批公开试验数据共 44,695 条，编码后 PT train 为 34,576,919 CE token。
数学科学、真实视觉/视频、SFT 及独立验证规模仍不满足正式配比；禁止据此启动正式主预算。
独立生成的诊断集包含 256 张图像、2,048 条可核验算术文本，仅供 K0/Q0/D0 学习诊断。
相似图像被分在同一训练组，当前视觉诊断没有独立视觉留出集；反事实/记忆检查
必须标为训练内诊断，不能作为视觉泛化成绩。

第二批实现补充了 Kimi expert MXFP4/activation MXFP8、DeepSeek expert MXFP4 与
Hadamard indexer QK FP4/BF16 scores；E2M1/E8M0 golden、STE、MTP BF16 CUDA
反向已检查。Kimi block scale 选择是显式本地仿真配方，尚未经过 5M SFT 校准。
SFT→MOPD/OPD 的教师轨迹、精确全词表 KL、response 分母与零优势跳过已接入；
教师资格要求互异权重及留出提升，当前没有九/十二位合格教师。
代码奖励使用独立无文件/网络/进程权限的 seccomp worker，代码不能接触标准答案；
现已增加本地结构化检索/内存文件与状态修改/Python 工具环境，及原生多模态
rollout：图像进入学生/reference/按域 effort 选取的教师，完整工具观察屏蔽动作损失。
这仍是本地同步短轨迹接口，未完成真实教师培养或正式 RL 训练。

训练入口新增显式 Text→Vision/QAT/MTP 权重系数切换，V1 冻结文本，视觉/aligner 独立 LR；
按实际非 padding 输入累积 batch、按 CE 的文本域与按样本的视觉域分别采样，图像和
视频按阶段暴露额度跟踪。PT/SFT 保存最低验证 LM NLL 权重，不能自动标为通过能力验收。
门禁绑定源码、配置、tokenizer、数据、阶段依赖及对应长度/模态/卡数的 profile；
人工来源/样本质量审阅仍需独立证据，尚不是完整主预算自动执行器。

当前实测：Kimi K0 为 500,450 CE token / 576 updates；DeepSeek D0 为 500,735 /
468 updates。两组结束均通过 DDP 参数逐项一致性检查。各自 16 个训练内算术问题
全部命中，9 个留出题全部失败，仅证明记忆能力。Kimi 最终颜色检查正确图像 16/16、
错误图像 0/16、遮图 8/16。Qwen 首轮 Q0 完成 500,450 CE / 576 updates，
DDP 参数一致，验证 LM NLL 1.157929；训练内算术 14/16、留出 0/9，未通过。
颜色正确图 16/16、错图 0/16、遮图 8/16；仍只是训练内视觉诊断。
三个 50 次预热 + 200 次更新的短诊断 profile 已生成；seq64 不能推算正式长文总时长。

试验数据扩展到 52,887 条，64K PT train 为 45,490,020 CE token；新增 8,192 个
由整数/有理数不变量核验的合成数学文档。它们不是自然科学数据，原代码许可和正式
验证规模仍未满足准入。96 张公开 ALLaVA 图像保留原始字节与来源许可，只供小型联合
配方试验。通用图像问句的去重身份包含按顺序排列的 RGB 哈希；不同图像的
“描述图片”不是同一个问题，同图像的模板和近重复仍绑定同组。纯文本首问仍按
规范化问句聚合，最多三个不同答案。

新增三类独立草稿模块、七步 Kimi LK、Qwen 四流 CE/三步自回馈、三阶段 DSpark，
以及目标冻结的 `train-draft`、精确恢复和目标 hash 绑定导出。主模型不纳入草稿
优化器或重复保存。DSpark 的衰减权重分母与有效位置预算分别跨累积窗口/DDP 汇总。
投机采样实现正残差拒绝重采样、全部接受后的 bonus、EOS 和原生状态回滚；
CPU 对照覆盖每个拒绝位置及三种原生图像 prefill。此路径尚重算草稿前缀，
不代表已完成正式草稿训练、接受率、confidence calibration、自适应调度或加速。
详细使用与限制见 `docs/draft-adaptation.md`。

Qwen 的配方 supervisor 在首轮算术门禁失败后停止，保留原始报告。
`miniqwen4-extension-1m` 从该权重另开 500K CE 诊断：明确重置优化器，
LR=1e-4/Muon=0.003、20K warmup/WSD，global32 序列保持不变，
microbatch 从每卡4/累积4改成每卡16/累积1；重新做性能 profile。
这是 Q0 允许的 0.5–2M 范围内的累计约 1M 诊断，不计正式 PT。
新的等待器 `miniqwen4-after-q0-extension` 仍要求 16/16 训练记忆和视觉依赖，
通过后才从零开始两组 20M 配方试验；不会降低原有门槛。
训练实现继续冻结在 `75a3364`，控制器单独记录内容 hash。

Qwen 续诊断已完成：新增 500,450 CE / 576 updates，累计 1,000,900 CE，
DDP 参数一致；训练内算术 16/16、留出 0/9。正确图 16/16、错图 0/16、遮图 8/16，
仍不具有泛化验收含义。报告分别保存为 `qwen-q0-arithmetic-1m.json` 和
`qwen-q0-visual-1m.json`。等待器已从零启动 Qwen 的 20M Muon 配方试验。

Kimi/DeepSeek 的 seq512 配方 profile 均完成 50 次预热 + 200 次实测更新：
分别约 561.38 / 516.84 CE token/s；只用于当前试验剩余优化更新估计，另加验证、
存盘和后续比较。共享机器的开发/测试负载会影响这些数值，不能推算正式长上下文总时长。
配方曲线在本地 6007 端口；6006 保留旧实验。实时进度用 `scripts/training_status.py`。

实验性批量专家 GEMM 在完整 Kimi BF16 微基准中出现梯度 cosine 0.6575、
relative L2 0.8345，尚未通过数值验收。单块对照误差较小，但完整网络差异
仍需定位，不能据微基准时间宣称训练加速。三个方案及在跑任务保持源码循环；
实验路径增加最多四倍 route-padding 限制，超过时回退源循环且不丢 token。

新增 control-v1 共享模板，训练/rollout/CLI 共用 reasoning/final/tool/effort 编码，
导出保留模板身份。正式 SFT/RL 要求该模板，RL 前驱及教师必须模板一致；不再
以普通提示文字冒充完整模式训练。完整轨迹按实际策略权重 hash 保存，包含工具输出、
真实行为概率、终止与奖励分项，恢复时不复用旧策略采样。RL 的输入/视觉账本和
验证分母已修正；quantity/box verifier 使用明确单位和容差。详见
`docs/posttraining-adaptation.md`。

GPU 测试发现并修正了 sampler 强制 BF16、调用者 FP32 的真实分布不一致。
修正后小型原生视觉测试 Qwen/DeepSeek FP32 ratio error 为零，Kimi CUDA KDA
不同累积路径约 0.000286；BF16 最大约 0.001468。实现分别采用 CPU FP32 2e-5、
CUDA FP32 0.001、低精度 0.02 的显式本地上限，超限立即在更新前失败。
这些上限与微型测试并非官方数值承诺或正式长序列 RL 的资格证明。
完整数据见 `native-rollout-precision-v2.json`，失败的原始日志保留。

待完成：正式数据规模/许可/配比/留出审计、长短上下文与批量课程、完整配方对照；
SFT/QAT 的学习验证、9/12 教师课程、思考模式与工具/多模态 RL 的实际学习验收；
三类正式草稿训练、优化与调度、端到端接受率/延迟、通过质量门槛的 demo。
单元测试、辅助损失实现或训练预算完成不代表这些阶段已经完成。

训练安排为 Kimi GPU 0–1、Qwen GPU 2–3、DeepSeek GPU 4–5。
GPU 6、7 的既有任务不改动。工作盘写入保留 50 GiB；根分区不放数据和大缓存。
每个 run 绑定源码 commit/内容 hash、数据 hash、冻结 tokenizer、配置、实际 token 和
RNG/cursor/优化器状态；诊断 run 不进入默认 demo。
