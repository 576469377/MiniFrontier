# MiniKimi-K3：全流程训练与结构改造方案

调研日期：2026-09-08。对象：175 的 `docker_jilin` 容器内 `/workspace/MiniFrontier`。本文是下一轮实施设计，未据此修改模型、重建数据或启动长训练。

## 1. 结论与复现边界

当前 MiniKimi-K3 已保留 KDA、周期性 Gated MLA、AttnRes、Stable LatentMoE 等主要文本骨架，但尚未形成完整 Kimi-K3 训练体系。下一轮应以“原生视觉与文本联合预训练 → 上下文扩展 → 多模态 SFT → 分领域、分推理预算 RL → MOPD → 草稿模型训练与部署”为主线，补齐 Quantile Balancing、Per-Head Muon、MTP、MoonViT-V2 和部署量化训练。

**建议保留现有 12 层、512 宽、32 专家文本容量，先修训练机制和数据，另建一个包含原生视觉结构的正式配置。** 随机初始化的小模型能学习结构与训练方法，但不会自动继承旗舰权重中的语言、知识或视觉能力。“单卡可训练”应指一个阶段的训练状态能在单张 24GB 卡上完成更新，而不是几小时能完成整个旗舰训练流程。

本文用以下证据等级：**官方报告**指已公开方法；**发布源码**指固定版本的实际代码；**当前实测**指本项目已检查的实现与产物；**本方案建议**指适配小模型的设计与数值；**待验证**指需要实验确认。文中全部 mini 数据量、学习率搜索区间、验收阈值均为本方案建议，不是官方未公开配方，也不是效果承诺。

### 1.1 已核对的基准

| 项目 | 基准与用途 |
|---|---|
| 技术报告 | [Kimi K3，v1，§2–4、附录 C/D][R1]：架构、QB、优化、联合预训练、MOPD、QAT、草稿训练 |
| 发布源码 | [Kimi-K3 固定版本][R2]，`c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`，与项目当前移植基准一致 |
| 更新情况 | 调研时模型仓库 main 已到 `f831ab66814297da540d832a5235f8e904f29d06`；本方案未把 main 自动替换为已验证基准 |
| 本地配置 | `/workspace/MiniFrontier/configs/minikimik3.json`；SHA256 `ba3333b8dc94453e82e7d2cd30e91fff0e8440b8c70ccc54b2e872d57998bf5d` |
| 当前证据 | `/workspace/MiniFrontier/docs/training-failure-v1.md`、`docs/audits/minikimik3-capability-v1.json`、`docs/audits/minikimik3-memorization-v1.json` |
| 版本记录限制 | 当前仓库源码仍显示为未跟踪文件，不能给出一个代表全部当前实现的有效提交号；正式训练前必须固化源码与配置哈希，并建立可追溯提交 |

官方源码并不等于公开了官方训练数据和全部训练程序。数据比例、mini 学习率、完整 RL 工程不能从推理代码中反推出“唯一正确答案”。本方案明确填补这些工程空缺。

## 2. 当前为什么不能正常说话

当前文本模型参数量为 **166,109,480**，不含待增加的视觉塔和 MTP。输入 embedding 与未绑定输出头共 **67,108,864** 参数，占约 **40.4%**。12 层中有 9 个 KDA、3 个 Gated MLA；专家为 32 选 2，另有 2 个共享专家；路由专家 latent 宽 256。

旧 `educational-v1` 从随机初始化开始，只执行了：

| 阶段 | 实际暴露量 | 含义 |
|---|---:|---|
| 基础预训练 | `1000 × 8 × 255 = 2,040,000` 个 next-token 监督位置 | 已准备训练集也仅约 763 万 token；实际只覆盖约 0.267 轮 |
| SFT | 4,000 条样本、532,176 个 assistant 监督 token | 不是 4,000 条全新独立问题，也未遍历完整 SFT 集 |
| DPO | 800 对偏好样本 | 在语言能力尚未形成时就进入偏好优化 |

完整 551 条 SFT 留出集上，SFT 检查点 token 加权 NLL 为 **6.868**，DPO 后为 **7.462**；生成仍重复、偏题。训练集 29,449 条记录仅有 11,865 种首个问题，单个“你今天过得怎么样？”出现 2,230 次，且 15.6% 对话被长度 256 截断。数据偏置、预算不足、过早 DPO 共同解释了现象；不能仅凭 loss 下降就排除实现问题。

最新 8 问答记忆诊断已达到 **8/8 精确复述、NLL 0.000448**。它说明当前链路能够学习并生成已记住的答案，不能证明泛化，也不能证明视觉、长序列和路由机制正确。

数据准备现在已改为全文件 reservoir 抽样，启动器取消隐式短预算，DPO 默认关闭，验证与 capability 标记也已修正；**旧数据与旧权重尚未因此重建**。下一轮应新建数据版本和 run，保留旧产物作为失败对照。

## 3. 模型结构：保留什么、必须改什么

### 3.1 文本主干

| 部分 | 当前状态 | 修改方案与验收 |
|---|---|---|
| KDA / Gated MLA | 3:1 混合已存在 | 保留；对 KDA 的逐 token、分块并行和增量状态做同权重输出及梯度对照 |
| MLA NoPE | 当前 MLA 使用 NoPE | 与发布结构一致；不能把没加 RoPE 当作 bug 随手补上 |
| Stable LatentMoE | latent 投影、RMSNorm、SiTU-GLU 等已移植 | 保留 routed/shared 的区别；核对 norm 位置、缩放、共享专家加和；记录各专家有效 token 和梯度 |
| AttnRes | block size 为 4 个 decoder 层 | 保留；checkpoint 重计算不得重复累积历史 block 或污染缓存 |
| 路由平衡 | 当前是按计数正负号调 bias | **新增 QB**，不是给现有策略换名字；详见 §4 |
| 优化器 | 当前主要 AdamW | **新增 Kimi 专属 Per-Head Muon 分组**，保留 AdamW 对照；不复用 Qwen 分组规则冒充等价 |
| MTP | 未实现 | 新增一个与主干结构对应的预测 block、融合投影和训练损失；报告描述与已发布文本配置 `num_nextn_predict_layers=0` 存在释放边界，不能声称现有源码已经提供完整 MTP 训练器 |
| 增量推理 | 尚缺完整高效状态链路 | 增加 KDA recurrent state、短卷积状态、MLA KV、AttnRes 单次前向状态的明确接口；训练与生成分别管理 |

此前检查中，小配置的 checkpointing 开关前后 logits 与参数梯度一致；这有助于排除已测路径错误，但不能覆盖完整 12 层 BF16、视觉、MTP、融合内核和长上下文组合。

### 3.2 词表与容量的决定

正式长训练前做一次固定文本字节预算的 32K/64K tokenizer 对照：中文、英文、代码、数学、OCR、特殊 token 分层统计压缩率、最长回复长度、低频 token 分布。32K 可以减少约 3,355 万 embedding/head 参数，但会改变序列长度、hash/特殊 token 映射和教师兼容性，不能只根据参数减少就判优。

本方案默认先保留 65,536 词表与 untied head，减少同时改变的变量。若 32K 在等字节数据和等墙钟预算的 pilot 中更好，则在正式预训练前一次性确定；之后 tokenizer 全流程冻结。教师、学生和草稿共享该 tokenizer。不得在旧检查点上无说明地截断词表，或把旗舰权重切片当作有效初始化。

### 3.3 原生视觉：应补 MoonViT-V2，而不是统一接 SigLIP

发布源码的视觉结构含 27 层、hidden 1024、attention projection width 1536、12 heads、patch 14，空间和时间处理及合并逻辑由 `modeling_kimi_k3.py` 与 processor 共同定义。它与 Qwen 的 Conv3D patch / mRoPE 路径不同。应直接移植这些模块，再通过配置缩小容量。[视觉源码][R3]

建议第一版 mini 视觉配置：

| 项目 | 本方案建议 |
|---|---|
| ViT 深度 / hidden | 12 / 384 |
| QKV 投影宽度 / heads | 576 / 6，head dim 96；需验证所选 GPU kernel 支持，不能悄悄改投影关系 |
| FFN 宽度 | 1536 |
| patch / 空间合并 / 时间合并 | 14 / 2×2 / 4，保持原生组织方式 |
| 投影到语言维度 | 原生 `PatchMergerMLPV2` 结构，输出 512 |
| 输入归一化 | RGB mean/std 都为 0.5，遵循 processor 的透明图处理和 resize 规则 |
| 图像分辨率课程 | 224 档启动，逐步加入 336/448 与动态宽高比，按合并后的视觉 token 分桶 |
| 视频课程 | 先 4 帧，再 8/16 帧；首轮 1–2 fps 是 mini 数据预算选择，区别于官方 processor 的更高采样密度 |

不要将上表当成已实现字段或已测显存。视觉塔每个 block 的注意力、FFN、位置编码、patch 排列、temporal merge 必须从原代码提取，不能只做一个名称相同的通用 ViT。原生 merger 将空间合并后的特征经过无 bias 的 MLP/GELU，再做输出 RMSNorm；不能换成 Qwen 的前置 LayerNorm merger。[视觉源码][R3]

图像先经过视觉塔双向编码，再作为语言上下文输入。必须由 processor 返回真实 feature 数、图像边界、grid、时间信息。图像占位符数量应严格等于合并后 feature 数；原始像素、placeholder 的语言 CE 标签均为 `-100`。相邻多图不得合成一个视觉注意力段。无图文本样本走真正的文本分支，不以黑图替代。

**正式路线：文本主干、mini 视觉塔和 projector 随机初始化后，从预训练早期联合学习。** 这保留了 Kimi 原生联合训练的目标；仅在开跑前做短暂模块调试并不构成一个长期“冻结随机视觉塔”的阶段。[报告 §3][R1] 如希望另做复用预训练视觉塔的快启动实验，应另命名为 transfer variant；换宽度后不能直接加载官方大视觉塔权重。

## 4. Quantile Balancing 与 Per-Head Muon

### 4.1 QB 的训练实现

对一次有效 optimizer update 中的全部非 padding token，设 token 数为 `m`、专家数 `n=32`、选中数 `k=2`。以旧 bias 做 `Top-(k+1)`；第 3 大的 biased score 是每个 token 的门槛 `alpha_i`。仅前 2 个专家参与前向。

按报告公式实现：

```text
margin[i,j] = raw_score[i,j] - alpha[i]
b_hat[j] = -quantile(margin[:,j], 1-k/n)
b_next = b_hat - mean(b_hat)
```

这是下一步 bias，当前 batch 不能用自身统计重新路由。最终推理冻结 bias。官方更新不是 `bias += lr * sign(target-count)`，也没有在该公式中加入一个任意 EMA 系数。[报告 §2.3.3、附录 C/D][R1]

实施细节：

1. 原始 score 与 biased cutoff 分开保存，不能把旧 bias 加两次；bias 只改变选择，不擅自改变专家混合权重公式。
2. 统计所有真实上下文 token，包括 prompt 和视觉位置；`labels=-100` 不等于 padding。padding、虚假补齐位置和 checkpoint 重算的重复 hook 不计入。
3. 32 专家的小模型先做精确 quantile oracle，规定 ties 与非整数 `mk/n` 的处理；验证报告中的小例子，再测随机同分、极小 batch、完全同质图像 batch。
4. 生产版本按专家累计 margin histogram，跨 microbatch 累积，DDP 时 all-reduce 计数，再求全局 quantile。范围溢出要计数、扩展并记录，不能把越界分布夹到边界而不知情。
5. 单次 optimizer update 后更新一次 bias；checkpoint 保存 histogram 状态、旧/新 bias、统计计数与更新时刻。评估、rollout、teacher forward 不更新它。
6. 记录每层 max/mean load、CV、空闲专家比例、bias 范围，并分别观察图像和文本路由。均衡不是要求每个短 microbatch 恰好平均，更不能通过丢弃 token 获得漂亮统计。

### 4.2 优化器与稳定性

Kimi 报告采用 Per-Head Muon，并结合 K2 的权重裁剪；学习率使用 1% warmup 后 cosine，weight decay 0.1。公开材料不足以支持直接给当前 mini 模型套一个“官方最佳 LR”。[报告 §2.5、§3][R1]

新增 `minikimik3_optim.py`：为每个矩阵登记其数学语义；注意力 Q/K/V 的 momentum 按 head 分块正交化，latent 下投影等非 head 矩阵按其独立矩阵处理；routed expert 分专家，不把专家轴混进一张矩阵。embedding、LM head、norm、标量和 bias 的 AdamW 分组要显式列出；官方未明确的特殊小矩阵分组标为本地决策，做等预算对照。不得仅用 `ndim==2` 决定所有参数归属。

建议 20M 有效预测 token 的 pilot 比较 AdamW 与 Per-Head Muon；固定数据顺序和总 token，Muon's LR/更新缩放成对记录。初始搜索为 AdamW 分组 LR `{1e-4, 3e-4}`，Muon 原始 LR `{0.005, 0.01}`，batch `{16K,32K}` 输入位置；这是搜索起点，不能与 DeepSeek 的 RMS 缩放 0.18 配方混用。先做小网格筛选，再在最佳两项上补种子。

主干/视觉/projector 的 LR multiplier 从 `1/0.5/1` 起测，视觉若明显欠拟合再比较 `1/1/1`；不得在视觉随机初始化时冻结其梯度。FP32 optimizer state，BF16 主计算；norm、路由 softmax、QB 统计、KDA 敏感累积用稳定精度。初始 global grad clip 1.0；记录裁剪触发率，长期高触发应诊断 LR、损失归一化或异常样本。

权重裁剪采用 K2 的 QK-Clip 作为明确来源：每个 head 统计该步有效 attention 的最大 pre-softmax logit，optimizer 更新后令 `gamma_h=min(1,tau/max_logit_h)`，对对应 Q/K 的非共享 head 分量各乘 `sqrt(gamma_h)`。K2 的参考阈值为 tau=100；本项目先以100作为对照，记录是否触发，而不是无依据强行裁剪全部层。[K2 §2.1][R8]

K3 的 mini MLA 使用 NoPE，因此不能照抄 K2 对共享 rotary key 的额外规则；应识别实际 Q/K 上投影中的 head 行，只缩放相应非共享分量，避免把共享 latent 下投影整体缩放影响所有 head。KDA 有自身归一化与状态更新，不能把普通 softmax attention 的 max-logit 裁剪机械套到其 recurrent state；公开 K3 材料未给出该映射的逐参数训练实现，需将 MLA 对应关系做 golden test，KDA 以源归一化与专门稳定性测试管理。记录裁剪发生在 optimizer 后、bias 更新前的时序，并验证 checkpoint resume。普通 global grad clipping 仍是另一项措施，不能代替 QK-Clip。

## 5. 数据设计：先重建分布，再扩大预算

### 5.1 数据源与准入

下表是候选池，不要求全量下载。先固定表末版本，按来源随机取样审阅，再构建本项目自己的不可变 shards。数据卡许可只是入口信息；混合集与原始图像/代码仍保留子来源说明。

| 数据源 | 使用位置 | 必须处理的问题 |
|---|---|---|
| [Fineweb-Edu-Chinese-V2.1][D1] | 中文自然文档 PT | 先取高质量目录，再核对该目录 score 的实际刻度；不对整个仓库盲写 `score>=4`；删除广告、导航、模板堆叠 |
| [FineWeb-Edu][D2] | 英文教育文本 PT | 固定子集与文档分组；避免与 SmolLM 的 fineweb 派生子集重复计算唯一数据 |
| [SmolLM-Corpus][D3] | cosmopedia-v2、python-edu | 合成文本与真实网页分开；代码保留原始许可、repo id，按仓库切分 |
| [Infinity-Instruct][D4] | 中文/双语指令、代码和数学补充 | 调研时为自动 gated，需已有授权才能取数据；按 task/首问模板去重，不能只按整段 JSON 去重 |
| [UltraChat 200K][D5] | 英文对话 SFT | 只从 `train_sft` 建训练池，保留官方测试 split；不把 generation split 混为同一 SFT 集 |
| [ALLaVA-4V][D6] | 双语 caption、视觉指令 | 图像级去重；NC 数据单独标识，caption 与 instruct 同图只进入同一 split |
| [FineVision][D7] | 文档、图表、科学图、定位指令 | 这是混合集，不能假设统一许可或全部样本适合 mini；按子集筛选，排除已选 ALLaVA/Docmatix 同源部分 |
| [Docmatix][D8] | 多页文档 QA | 按原 PDF 文档分组，核对回答页；先单页、后双页，防止只留下无答案的一页 |
| [LLaVA-Video-178K][D9] | 短视频时序与描述 | 原视频分组切分；保留时间戳和真实帧率，剔除下载失败及抽帧后丢失答案事件的样本 |
| [OpenThoughts-114k][D10] | 有限推理 SFT 候选 | 仅取能在 mini 长度内保留完整解答、且有校验的子集；长推理截断不作为正例 |
| [MiniMind dataset][D11] | 兼容性回归与少量对照 | 保留旧配方用于消融，不再让重复首问的前缀样本主导主训练 |

MiniMind-V 可参考数据加载和演示工程；其替代视觉塔方案不等同于本项目的 Kimi 原生结构复现。[MiniMind-V][R4]

### 5.2 建议的预训练混合

纯文本流按 **token** 配比：中文教育/自然文档 45%、英文教育/自然文档 25%、代码 15%、可校验数学与科学 10%、高质量对话式文本 5%。每桶内部按来源设置上限；“数学”不能仅按是否出现数字分类。

多模态流按 **样本数** 配比：caption 45%、OCR/文档 25%、图表/计数/空间关系 15%、图文交错 10%、短视频 5%。比例与文本 token 比例是不同分母，不能相加。早期视频比例可为 0，再在后续课程达到目标；记录每阶段真实比例。

默认主预算为 **2.0B 文本/描述 next-token 监督位置**，另跟踪 **2.0M 图像出现次数和 30K 短视频样本**。caption/OCR 答案的监督 token 包含在 2.0B 内；视觉 feature 位置不计为语言 CE token。图像出现次数不是唯一图像数；设唯一图像目标至少 0.8M，未达到时明确报告重采样倍数并扩大来源。若清洗后样本不够，降低阶段暴露目标并更新预算，不伪造唯一量。

### 5.3 清洗、切分、tokenizer、packing

统一存储 `sample_id/source/revision/source_item_id/license/split_group/lang/task/quality_flags/content_hash`；图像另含 decoded RGB hash、原图 id、宽高、pHash、所属文档/视频 id；推理数据保留 verifier、结果与生成模型来源。

按顺序做：解码与字段校验 → 语言/质量过滤 → 精确文本 hash → 近似段落去重 → 首问及回复模板聚类 → 跨源图像/文档/视频去重 → 按 group 切分 → tokenizer → packing。预训练验证至少保留各来源 0.5%，并固定约 5M token 的周期验证集；SFT 固定不少于 2K 独立首问组，多模态固定不少于 1K 独立图像/文档组，最终测试独立封存。

建议同一规范化首问在 SFT 中最多保留 3 个高质量且答案实质不同的版本；身份/存在意义类在整体 assistant token 中不超过 0.5%。短寒暄可以保留，但不得成为主模板。对话按轮构造完整回复，过长样本转入长序列桶或丢弃，不能裁掉答案尾部再补一个假的 EOS。

PT 文档之间加入 EOS，并在 packing 中隔离不同文档的 attention、KDA recurrent state 和卷积历史；单纯加 EOS 不会自动清空线性注意力状态。暂未实现 segment reset 时，一条 sequence 只装一个文档的连续块，接受利用率损失。多模态样本不能切断一张图或一组时间合并帧。

tokenizer 只在训练集上学习；预留角色、图像、视频、工具、effort、think 边界；模板转换集中于一个处理器，训练与 demo 调用同一实现。SFT 只监督 assistant 正文、有效输出结构与结束 token；user/system/tool observation 不参与 CE。采用逐 token provenance 检查 200 条样本，查看输入、标签、解码后完整答案及图像对齐。

## 6. 从随机初始化到基础模型：逐阶段执行

以下是**首轮完整预算**，每一段有进入条件。预算代表计划上限；如果前段已经显示数据/优化错误，不继续烧后面的 token。

| 阶段 | CE token 预算 | 图像 / 视频暴露 | 长度与任务 | 参数与退出要求 |
|---|---:|---:|---|---|
| K0 正确性与可学习性 | 0.5–2M，单列调试 | 64–512 张诊断图 | 64/256/512；小样本记忆、反事实图像 | 检查全部模块梯度、图像真的影响结果；不计主预算 |
| K1 联合短序列预训练 | 200M | 200K 图像；0 视频 | 512→1024，224 图为主 | 全部训练；QB/Muon/MTP 均进入正式路径，建立各域下降趋势 |
| K2 主联合预训练 | 1.20B | 1.20M 图像；20K 视频 | 1024→2048；加入 OCR、交错图文、4/8 帧 | 全部训练；维持文本回放，不能视觉 loss 降而文本崩溃 |
| K3 上下文和视觉复杂度扩展 | 400M | 500K 图像；10K 视频 | 2048→4096，448 图、双图/双页、8/16 帧 | 含跨段检索、长文、帧序辨别；实际训练到的长度才允许宣传 |
| K4 高质量冷却 | 200M | 100K 图像 | 1024/2048/4096 混合，增加高质量数据比例 | cosine 到峰值 LR 的 10%；产出 base 文本与视觉评估 |
| 合计 | **2.0B** | **2.0M 图像；30K 视频** | 不含 SFT/RL/draft | 各阶段按真实计数结算 |

2B 约为旧 Kimi PT 预算的 980 倍，这说明旧实验离此类训练规模很远；并不意味着达到 2B 就必然可用。若 100–200M token 后语言仍严重异常，先比较数据、路由、优化器和同容量 dense 对照；若 2B 时验证仍稳定改善且下游可见收益，可追加 1B，并保留原预算分界。

### 6.1 批大小、损失与课程

单卡从 text microbatch 1–2、图文 microbatch 1 开始。有效输入位置 batch 目标 16K，稳定后比较 32K；通过 gradient accumulation 实现。`B_input=microbatch×accum×sequence_length` 仅为无 padding 的上限；每次更新真正计数 `N_CE=sum(labels_next!=-100)`。SFT/多图的 `N_CE` 可能远小于输入位置。

训练器应对整个 accumulation window 的 CE 总和除以该窗口有效目标总数，而不是简单平均不同答案长度的 microbatch mean。DDP 时考虑框架的 gradient averaging，验证一个大 batch 与分片累积后的参数更新一致。零监督 batch 不更新 optimizer 或 QB。

```text
L = L_next_token + lambda_mtp * L_mtp
```

QB 是无额外辅助平衡损失的路由偏置更新，不为“看起来一致”再加入 DeepSeek 的 sequence balance loss。MTP 建议 `lambda_mtp=0.1` 起步，pilot 比较 0/0.1/0.2；每个预测深度独立归一化有效标签。MTP 的 shifted token embedding 只能看到该深度允许的 token，严禁答案泄漏；EOS、文档切分、图像位置和 padding 附近的未来标签必须掩码。它不增加“独立训练数据量”，不能把同一文本的附加预测重复算进 2B 主预算。

上下文课程基于实际文档长度，而不是给同一批 256-token 文本补 padding。KDA 的时间合并、MLA 的全局交互、视觉段及文档 reset 都做边界用例。官方长上下文路线是研究参照，mini 先验收 4K，8K 作为额外阶段单独测显存和真实检索，不能在配置中填 1M 后宣称支持。

## 7. 多模态 SFT：语言、视觉与可控推理

先由 K4 base 通过基础能力门槛，再开始 SFT。建议清洗后的独立池：文本 300K–600K 条、多模态 300K–600K 条，先 1 epoch、最多 2 epochs；用约 **80M–200M assistant token** 作为实际执行预算，不按固定 500 步终止。

按 assistant token 配比：日常解释/问答 25%、摘要改写翻译 15%、数学 10%、代码 10%、图像描述/VQA 15%、OCR/文档/图表 15%、工具调用与多轮纠错 5%、短视频 5%。同一图的多个问答按图像组限频。纯文本和视觉数据都要含“信息不足”的合理短回答，避免模型无论有无图都猜一个常见答案。

长度桶 512/1024/2048/4096，建议比例 15/35/35/15，按真实样本长度调整；保留完整 final answer。effort low/high/max 用不同预算的高质量解答，不把同一长答案换标签凑三份。首轮 max 输出上限建议 1024 token；困难样本需要更长时进入单独长推理阶段，不因旗舰有极长推理就给这个小模型无限生成。

优化器保留已验证的 Kimi 分组；主干峰值 LR 从 `{2e-5,5e-5}`、视觉塔倍率 `{0.2,0.5}`、projector 倍率 `{1,2}` 小范围选择；warmup 为该阶段 token 预算 1%–3%。SFT 的学习率不是把 PT 的 Muon 原始数值全部直接改成同一个 AdamW LR。

每 5M assistant token 评估完整周期集和固定生成；保存 val 最优与人工可用的多个候选，不依赖 latest。若文本和视觉发生取舍，记录 Pareto 候选，使用任务权重挑选；不得用总平均掩盖中文或 OCR 明显退化。

## 8. 领域 RL 与 MOPD：保留完整算法路线

### 8.1 RL 前置门槛与九位教师

Kimi 官方路线按领域和推理预算训练专门策略，再蒸馏进同一学生；本项目保留 **3 类领域 × 3 个 effort = 9 个教师槽位**。领域可对应通用任务、通用工具任务、代码工具任务，但训练环境需适配当前小模型能力。九个教师是九个明确 checkpoint，不要求九张卡同时训练，也不能把同一个 checkpoint 重复登记为九位已训练教师。[报告 §4.1][R1]

按顺序培养并保存教师，一次只驻留一个训练策略及必要 reference。以已经达到 SFT 门槛的模型为共同起点，先每域每 effort 2K 验证任务筛掉无效设置，再规划 **每位教师 10K–30K 独立训练 prompt、每 prompt 4–8 次采样**，最多 2 轮；保留任务 id 与 rollout 总 token，避免把 9×30K 当作 270K 唯一问题。

通用任务用可验证算术、结构化约束、封闭上下文问答；工具任务使用本地结构化数据库/文件检索、明确状态变更的模拟环境；代码任务使用小函数与单元测试，再进入小项目修复。视觉任务增加图片中的数值提取、图表计算和截图元素定位。只有任务通过率既非全 0 也非全 1 时才适合提供组内相对奖励；把稀疏失败变成难度课程，不能用输出长度奖励弥补。

### 8.2 RL 训练器必须先修的逻辑

当前 `/workspace/MiniFrontier/minifrontier/training/rollouts.py` 已有分组优势与 MOPD token reward，但生成器默认温度/top-p/特殊 token 处理与重算 `old_logp` 的原始 softmax 分布不一致。**先修行为策略概率定义，再进行任何有效性 RL 实验。**

首轮建议 rollout 使用 `temperature=1`、不做 top-p 截断，并统一合法 token mask；在采样时保存行为分布的 log probability。训练时重算同一分布规则，冻结 old/reference 参数。若恢复 top-p，必须保存当时支持集和归一化方式，并对 importance ratio 定义做专门验证，不能仅记录一个 top_p 字段。

GRPO 的基本组内标准化、clipped policy objective 可参考原始方法，但 Kimi 的完整 RL 系统还有长任务/部分 rollout 工程，当前模块不等于完整复现。[GRPO 原论文][R5] mini 同步实现先规定：group size 4，clip 0.2，reference KL 系数从 0.01 起调，LR 1e-6–5e-6，单批 rollout 最多训练 1–2 次；同组同任务同 effort，零方差组跳过并记比例。

工具 observation 不参与 policy action loss；每步保存 prompt、tool call、observation、终止原因、策略版本、奖励各项。工具执行使用隔离环境、固定超时与可重复初始状态。部分 rollout 必须保存 prefix 与生成版本，恢复时不能把旧采样伪装为当前 on-policy；第一版采用同步完整短轨迹，通过后再加入过期策略修正。

奖励优先 exact answer、解析正确、单测通过、定位误差等。无法程序化判断的任务先建立固定 rubric 与双评审抽检；若引入外部模型裁判，保存裁判版本/提示/分数并验证对长度、语言和模板的偏置。未经校准的裁判不得成为自动提升“质量”的唯一依据。

### 8.3 MOPD 的正确目标

学生在当前任务和 effort 下自己生成轨迹；由对应 `(domain, effort)` 教师对同一 token prefix 打分。Kimi 的目标使用采样 token 上的 log probability 差，经 stop-gradient 和 clipping 形成稠密奖励：

```text
r_t = clip(stopgrad(log p_teacher(y_t | x,y_<t)
                    - log p_student(y_t | effort,x,y_<t)), -Rmax, Rmax)
```

它经 policy loss 更新学生；不是在教师生成的固定答案上做 CE，也不是本方案 DeepSeek 所用的 full-vocabulary reverse KL。当前 `mopd_advantages` 的差值方向与此一致，但还需补足教师注册、采样分布、effort 提示和完整验收。[报告 §4.1.3][R1]

建议学生 MOPD 预算 **20M–60M 自生成 response token**，Rmax 从 5 起测并记录 clipped 比例。按域/effort 均衡抽样，再依据留出能力设置采样权重。教师与学生必须词表一致；外部旗舰 API 如果没有同词表全轨迹 logprob，只能提供离线指令/推理示范，不能直接称为精确 MOPD 教师。

单卡顺序执行：学生生成并存 token → 卸载/暂停学生训练状态 → 加载当前教师打分并缓存所需 logprob → 回到学生更新。缓存绑定教师 hash、学生 rollout 版本、processor/tokenizer、task id。9 个教师不同时驻留显存。MOPD 之后必须分别测九个区域，拒绝用一个平均分掩盖遗忘。

### 8.4 DPO 的位置

DPO 是可选偏好对照，不是 Kimi 官方主线的替代。只有 SFT/MOPD 已能稳定回答，且存在合格同 prompt 偏好对时才试 5K–20K 对、beta `{0.03,0.1}`、LR `{5e-7,2e-6}` 的小预算实验；reference 固定为进入 DPO 的 checkpoint。DPO 目标和长度归一化不能随手改变。[DPO 原论文][R6]

使用相同对话集 NLL、paired 生成与长度分布共同验收。本次旧 DPO 的退化说明必须允许该分支被拒收；它不能自动成为 demo 默认权重。

## 9. QAT、MTP 转草稿模型与部署

### 9.1 QAT 保留官方目标，但说明 3090 执行方式

Kimi 部署训练对 routed MoE expert 权重做 MXFP4、相应激活做 MXFP8，attention、latent 投影、shared experts、router 等保留更高精度；从 SFT 到 RL 保持同一量化方案。[报告 §4.1.4][R1]

3090 的实施路线是 FP32 master + BF16 计算 + 匹配 MX 格式的 quantize/dequantize 仿真和 STE。**仿真 QAT 能研究数值适应，不代表在 3090 获得原生 FP4/FP8 运算加速。** 量化格式、block scale、舍入、饱和和 transpose 的行为应按原生实现写 golden cases，不能用普通 INT4 近似后还标 MXFP4。

先以 5M SFT token 比较 BF16 和 QAT，记录专家输出误差、token NLL、各域生成、吞吐、峰值显存；校准通过后正式 SFT/RL/MOPD 使用一致 QAT。若格式仿真尚未验收，则产物明确标 BF16 分支，待补 QAT 从 SFT 起点重跑；不能在最终导出时临时加一个量化函数冒充训练过。

### 9.2 草稿训练有独立结构问题

Kimi 的 MTP 在后训练后转换为 EAGLE-3 风格单层草稿，只训练草稿与特征融合层，目标模型冻结；报告进一步采用 7 步 unroll 和以 acceptance overlap 为目标的 LK loss。通用 EAGLE-3 实现只能辅助实现，不能覆盖 Kimi 特有损失。[Kimi 报告 §4.1.4][R1] [EAGLE 官方训练代码][R7]

当前 12 层、每 4 层一个 AttnRes block，只有 3 个 block，不能直接引用旗舰的第 1、第 4、最后 block。建议保留主干，把 mini 融合 tap 明确定义为 **第 1、第 2、第 3 block 的输出**，记录为容量缩小后的适配；不要为草稿偷偷改变主干 block size。融合矩阵初始化为 `[0,0,I]`。

```text
L_LK = -log(sum_v min(p_target(v), p_draft(v)))
```

在 temperature 1 下计算，不附加未经依据的 ground-truth CE。unroll 后续步使用草稿自身输出，不能偷读该步不可用的 target feature；按报告保留 7 步训练结构，先 2 步测试再扩完整 7 步。建议 **10M–30M 有效 draft 训练位置**，纯文本和视觉 prompt 均覆盖；最后阶段使用与主模型一致 QAT。

先完成普通增量生成，再实现无损 speculative accept/reject 和缓存 rollback；测拒绝发生在任意草稿步时 KDA/卷积/MLA 状态是否回滚正确。验收目标是分布正确与实测延迟改善，草稿 acceptance 高不保证这个 mini 模型端到端更快。

### 9.3 Demo 与导出

导出包含 config、tokenizer、processor、主权重、QB bias、草稿权重（如有）、量化格式、训练数据 manifest hash、代码 hash、能力报告。界面支持无图/单图/多图/短视频和 effort 选择，但只展示实际验收过的范围。坏图、超限帧、图像 token 不匹配应报错；不得吞掉图像后继续给用户一个看似看图的回答。

## 10. 单张 3090 的执行预算

已有短文本测试约 **3.30GiB** 峰值只对应当前文本模型、batch 2、长度 256 的一次完整更新；不能外推到新视觉塔、4K、MTP、QAT 或九教师同时驻留。

FP32 参数+梯度+Adam 两个状态的粗略下界为 `16P bytes`，当前文本约 **2.48GiB**；Muon 分组状态大小不同，还会有 BF16 副本、激活、logits、临时矩阵、allocator 余量。`[B,L,65536]` FP32 logits 在 `B=1,L=4096` 时就约 1GiB，一次完整 materialize 多份会迅速放大。优先实现 chunked/fused LM loss、activation checkpointing、KDA fused kernel 与长度分桶。

每个阶段先 50–100 次 warmup，再测 200 次真实 optimizer update，记录训练用有效 CE token/s、图像/s、视频帧/s、峰值 allocated/reserved、I/O 时间；测量必须包含 backward、optimizer、QB、MTP。预留至少约 2GiB 运行余量，OOM 首先减 microbatch/视觉 token/并发教师，不能静默跳过样本。

时间估算用 `天数 = 有效 CE token / 实测 CE token每秒 / 86400`。假设速度为 500/1000/2000 token/s，**每 1B token** 分别约 23.15/11.57/5.79 天；这些是假设情景，不是当前单卡实测速率。2B 主训练另加视觉解码、SFT、九教师 RL、MOPD 和草稿的成本，完整路线需要按周甚至更久规划。视频/视觉比例变化后必须重新测，不可只用纯文本速度乘总量。

每阶段原子保存完整训练状态；同阶段 resume 恢复 RNG、数据游标、累积计数、QB、optimizer、scheduler。跨阶段显式指定继承权重与重置哪些状态，不把最后一个冷却 LR 直接当下一阶段峰值。

## 11. 验收：什么时候可以继续、什么时候应回退

| 层级 | 检查与建议门槛 | 失败处理 |
|---|---|---|
| 数学/结构正确性 | tiny FP32 同权重参考、因果扰动、分段重置、增量与全量、checkpoint 梯度对照；BF16 单独设相对误差容限 | 修实现后重新跑，不以降低 LR 掩盖 |
| 可学习性 | 32–128 个短文本/图文样本可过拟合；换图能改变对应答案，遮图明显变差 | 查 labels、placeholder、梯度与数据 |
| base 学习 | 各域 val NLL 随真实 token 改善，固定 200 个续写中逐步形成连贯短文；专家无持续饥饿 | 比较优化器、清洗和容量，暂停复杂后训练 |
| SFT 进入 RL | 固定 500 个未见问题人工/规则评估：基础指令完成率建议 ≥80%，明显乱码/无穷重复 ≤5%；至少 100 个独立视觉任务优于无图对照 | 补 SFT/数据，不使用 RL 修语言缺失 |
| 视觉能力 | caption/VQA、OCR CER、文档 ANLS、图表数值、定位误差分开记录；图像打乱/空白/冲突测试必须有区分 | 若无图区别很小，诊断是否忽略视觉 |
| RL/MOPD | paired holdout 改善，低/高/max 各域无重大回退；无额外模板/长度投机 | 回退对应教师或 MOPD 权重，重新选数据 |
| 发布 | 20 个固定 demo 场景完整复现；新会话、缓存清理、图像不匹配、长输入、断点恢复均通过 | 只发布通过的能力范围 |

上述数值是项目第一版 release gate，需冻结评测集后执行；它们不能与旗舰榜单分数混为一谈。NLL 只能在同 tokenizer、同数据、同标签和同归一化方式下比较。使用 bootstrap 或多种子比较关键方案，避免把几个样本的波动当改进。

## 12. 实施清单与依赖顺序

所有新路径以下均是**建议新增**，并非当前已经存在的 API。

| 优先级 | 文件或模块 | 工作与完成证据 |
|---|---|---|
| P0 | `/workspace/MiniFrontier/minifrontier/data.py` | 新数据 manifest、首问/图像 group 去重、完整回复、分桶；输出抽样审阅与覆盖报告 |
| P0 | `/workspace/MiniFrontier/minifrontier/training/runtime.py` | 将通用 RouterBalance 与 Kimi QB 分离，正确累计有效 token，防重算重复计数 |
| P0 | 新增 `minifrontier/training/kimi_quantile_balance.py` | 精确 oracle→histogram；单卡/累积/DDP 更新等价测试 |
| P0 | 新增 `minifrontier/training/minikimik3_optim.py` | 显式语义分组、per-head 正交化、更新参考；权重裁剪另验收 |
| P0 | `/workspace/MiniFrontier/minifrontier/models/minikimik3/modeling.py` | 暴露 inputs_embeds、segment、loss mask、MTP 及视觉 feature 插入接口 |
| P1 | 新增 `minifrontier/models/minikimik3/vision.py` 与 `processing.py` | 原生 MoonViT/merger/processor；单图、多图、视频 grid 和源代码对照 |
| P1 | 新增 `minifrontier/models/minikimik3/mtp.py` | MTP shifted mask、独立 loss、特征接口；主干 λ=0 等价 |
| P1 | `/workspace/MiniFrontier/minifrontier/training/train.py` | 按 token 预算、跨累积窗口归一化、阶段进入门槛、完整可恢复状态 |
| P1 | `/workspace/MiniFrontier/minifrontier/training/rollouts.py` | 行为概率一致、effort/teacher 路由、工具 observation mask、同步短轨迹 |
| P1 | 新增 `minifrontier/training/quantization.py` | MX 格式仿真、STE、训练/rollout 一致性与 BF16 对照 |
| P2 | 新增 `minifrontier/training/kimi_draft.py` | 三个 mini block tap、7 步 unroll、LK loss |
| P2 | `/workspace/MiniFrontier/minifrontier/inference.py` | KDA/MLA cache、视觉输入、speculative 回滚；只选择 capability 合格权重 |

推荐依赖顺序：固定源码与词表 → P0 数据/数值/QB/Muon → 视觉与 MTP 正确性 → K0–K4 → QAT 验收和 SFT → 九教师课程 → MOPD → draft → demo。每步产物包含权重与验证报告；没有可用 base/SFT 之前，不把后面阶段全跑一遍当作完成。

## 13. 固定版本与资料索引

| 数据集 | 调研固定 revision | 数据卡许可入口 |
|---|---|---|
| Fineweb-Edu-Chinese-V2.1 | `a5b574efa48beb3a8f6887ef0b093becf004328b` | Apache-2.0 |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-BY |
| SmolLM-Corpus | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-BY；python-edu 追溯原代码许可 |
| Infinity-Instruct | `bddc39a8feadbd679c30623197f4e736b7e75b48` | CC-BY-SA-4.0；自动 gated |
| UltraChat 200K | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT |
| ALLaVA-4V | `0fd42fce5c047d387a4bb5318d588eae9a9797f0` | CC-BY-NC-4.0 |
| FineVision | `3c380a731a3429c1d04693d6ec16d7e683def84c` | 顶层无统一标识；逐子源记录 |
| Docmatix | `0725b65616e0e5f6024be10e38ddf8d8c48664fd` | MIT 数据卡；文档来源仍单列 |
| LLaVA-Video-178K | `6d8c562dc26d70042a0d9704d1cae58c94b89098` | 顶层未统一标识；逐视频来源记录 |
| OpenThoughts-114k | `bd093c3994fd54d2390985b66988ddf282a55eb6` | Apache-2.0 |
| MiniMind dataset | `312afb4f76391145c6902f765bb51691c09a12f5` | 卡片列 Apache-2.0 与 CC-BY-NC-2.0，按文件追溯 |

[R1]: https://arxiv.org/html/2607.24653v1
[R2]: https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721
[R3]: https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_k3.py
[R4]: https://github.com/jingyaogong/minimind-v
[R5]: https://arxiv.org/abs/2402.03300
[R6]: https://arxiv.org/abs/2305.18290
[R7]: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/main.py
[R8]: https://arxiv.org/html/2507.20534v1
[D1]: https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/tree/a5b574efa48beb3a8f6887ef0b093becf004328b
[D2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
[D3]: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f
[D4]: https://huggingface.co/datasets/BAAI/Infinity-Instruct/tree/bddc39a8feadbd679c30623197f4e736b7e75b48
[D5]: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a
[D6]: https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/tree/0fd42fce5c047d387a4bb5318d588eae9a9797f0
[D7]: https://huggingface.co/datasets/HuggingFaceM4/FineVision/tree/3c380a731a3429c1d04693d6ec16d7e683def84c
[D8]: https://huggingface.co/datasets/HuggingFaceM4/Docmatix/tree/0725b65616e0e5f6024be10e38ddf8d8c48664fd
[D9]: https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/6d8c562dc26d70042a0d9704d1cae58c94b89098
[D10]: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/tree/bd093c3994fd54d2390985b66988ddf282a55eb6
[D11]: https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5
