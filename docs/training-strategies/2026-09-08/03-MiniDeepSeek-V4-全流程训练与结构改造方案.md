# MiniDeepSeek-V4：文本、Vision-Exp 与全流程训练改造方案

调研日期：2026-09-08。当前实现位于 175 的 `docker_jilin` 容器 `/workspace/MiniFrontier`。本文是研究与实施设计；未据此改模型或启动下一轮训练。

## 1. 结论：应区分两个官方版本，并保留完整训练主线

当前 MiniDeepSeek-V4 已保留 SWA、CSA/HCA、mHC、Hash MoE 等文本结构，但有一个需要优先修正的缩配问题：**SWA window 64 小于 HCA 的 128 压缩块，使尚未完成压缩的较早 token 在该层失去直接可见路径。** 应先恢复 window 128，再补 DeepSeek 专属 Muon、sequence-wise balance loss、MTP、全词表 OPD 和量化训练。

调研时官方已发布 **DeepSeek-V4-Flash-Vision-Exp**，包含原生视觉塔、图像 token 排列、视觉可见性、独立视觉路由 bias 及 DSpark 草稿路径。[官方模型说明][R4] 因此不能再简单说“DeepSeek-V4 不支持视觉”；准确描述应是：**当前项目固定的早期文本基准尚无视觉，最新官方视觉分支需要另行迁移。**

建议用一个文档管理两条连续、可追溯的产物线：

1. **MiniDeepSeek-V4-Text-v2**：先修正文本结构与训练，完整经过 dense 起步、indexer warmup、sparse PT、SFT、领域 GRPO、full-vocabulary OPD。
2. **MiniDeepSeek-V4-Vision-v1**：从已学成的 text base 接入官方 Vision-Exp 视觉结构，完成视觉继续预训练、多模态 SFT、视觉/文本领域 RL、OPD、QAT 与 DSpark。

视觉版官方说明明确属于加入视觉模块后的 continued training；具体语料、塔初始化和每阶段 token 预算没有全部公开。下文的视觉 warmup/继续训练数值是针对本项目的实现建议，不是对官方内部配方的猜测。[模型说明][R4]

## 2. 已检查的源码、配置与训练证据

本文区分 **官方报告、发布源码、当前实测、本方案建议、待验证**。所有 mini 预算、阈值和 LR 搜索默认属于本方案建议。

| 证据 | 固定版本 |
|---|---|
| [DeepSeek-V4 技术报告][R1] | `2606.19348v1`，架构、Muon、预训练、GRPO/OPD、QAT |
| [早期 Flash 模型与推理代码][R2] | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| [早期 Flash 实际 inference/config.json][R3] | 以发布配置区别于 ModelArgs 示例默认值 |
| [Flash-Vision-Exp][R4] | `6821d6ad3681a4b137b066b76094fa82ebd0a380` |
| [Vision-Exp inference/model.py][R5] | 图像可见性、bias_vl、视觉注入、DSpark |
| [Vision-Exp vision.py][R6] | ViT、2D RoPE、RMSNorm、SwiGLU、3×3 aligner |
| [Vision-Exp image_processor.py][R7] | resize、图像 sentinel、N-layout、padding 与图像排列 |
| [官方 DeepSpec][R8] | `005e03b81cec38b7da6399833d609ee89a2587f2`，已读 trainer、DSpark loss、Markov head、配置 |
| [DSpark 技术报告][R9] | `2607.05147v1`，半自回归与置信度调度 |
| 当前配置 | `/workspace/MiniFrontier/configs/minideepseekv4.json`，SHA256 `661d01ad19ccc6510a86ab48a89b5b159e7c19fd2ffcdb7401d385c06fc8932e` |

当前 Git 源码仍未跟踪，正式训练前必须固化 commit/文件 hash 与数据版本；本文不能提供一个代表全部当前实现的已提交修订号。

### 2.1 当前失败模型的真实规模

当前文本参数 **229,996,877**，另外有 262,144 个整型 hash routing 表项。12 层 hidden 512、8 heads×64、32 routed experts top-2、1 shared expert、前两层 hash routing；mHC mult 4、Sinkhorn 20 次。embedding 与 untied head 合计 67,108,864 参数，占约 **29.18%**。

旧训练从随机权重出发：base 2,040,000 个 CE 目标；dense indexer 阶段冻结主干，不计 LM 数据；sparse CPT 1,636,800 个 CE 目标。主干总计 **3,676,800**。SFT 4,000 样本、532,176 assistant token；DPO 800 对。

完整 551 条 SFT 留出集 token 加权 NLL：SFT **6.625**，DPO **7.392**，实际回答仍不连贯。旧 SFT 数据 29,449 条只有 11,865 种首问，少数模板占比极大，长度 256 截断 15.6% 对话。8 问答记忆实验 **8/8 精确复述、NLL 0.000222**，只证明短样本可记忆，不说明基础能力或架构全部正确。

当前数据准备已改全文件抽样，预算显式化、DPO 默认关闭、均匀验证与 failed capability 标识已完成；**旧数据与权重尚未按新策略重训**。下一轮不要在旧失败 DPO 上继续堆复杂 RL。

## 3. 文本模型结构修改：从配置契约到数学检查

### 3.1 必须先修：window 与压缩覆盖

当前 HCA ratio 128、window 64。以零起始 query 96 为例，SWA 能看 33…96，第一个 128-token HCA block 尚未完成，0…32 不在该层的直接可见集合。此前定向实验改变 token 0：window64 时该层最后位置输出变化为 0；同权重改 window128 后出现非零差异。这是具体的缩配失配，不是泛泛怀疑注意力结构。

建议将 `/workspace/MiniFrontier/configs/minideepseekv4.json` 的 `window_size` 改为 **128**，保留 HCA 128，先测试所有 phase `position % 128` 的覆盖。该结论只描述单层直接路径；其他层仍可能间接传信息，不能说整个模型完全看不到早文。[官方发布配置][R3]

若将来想缩 HCA 至 64，必须连同 pooling、RoPE、cache、训练长度与目标配置一起作为新变体实验，不能只改一个 ratio 来掩盖问题。

### 3.2 报告、发布配置与 mini 的差异

| 项目 | 当前 mini | 发布基准/建议 |
|---|---|---|
| 层压缩序列 | `0,0,4,128,...` 共 12 层 | 发布 Flash 也从两个 0 开始；报告文字与发布配置存在差别，以固定发布代码+配置作为本项目契约，不盲改前两层为 HCA |
| SWA window | 64 | 改 128，恢复与最大未完成压缩块的覆盖关系 |
| route scale | 使用示例默认 1.0 | Flash 实际配置 1.5；新初始化 v2 建议按 1.5，做等权重短对照，不在旧权重上无说明切换 |
| compressed RoPE theta | 40,000 | 发布配置 160,000；新 v2 优先以 160,000 为对照基准，保留 40,000 的短 pilot，以 mini 检索与长文曲线判断 |
| 普通 RoPE / YaRN | theta 10,000，max seq 4096 | 不复制旗舰 64K→1M 的 factor 16 到从未训练到该长度的 mini；本地先原生 4K，再做有预算的 8K 扩展 |
| hash layers | 2 | 旗舰 3；这是深度缩小的选择，不是已经证实的 bug；保留 2 并登记差异 |
| SwiGLU clamp | 10 | 已有，保留并核对 gate 仅上界、linear 两侧截断，避免写成所有张量统一 clamp |
| mHC | 4 streams / 20 Sinkhorn | 保留；映射矩阵的归一化和稳定精度不能用普通 residual 代替 |
| norm eps | 1e-6 | Vision-Exp 配置为 1e-20；这是版本迁移议题，不是全局搜索替换，特别区分 RMSNorm、Sinkhorn epsilon 与 Adam epsilon |
| MTP | 未实现 | 文本线补原 V4/V3 式预测结构，Vision/DSpark 另见 §11 |

`route_scale=1.5` 与 compressed theta 的差异来自实际发布配置，而非参数量缩小所必然需要的变化。建议新版本显式写入所有有效默认值，使配置能够独立复原模型。变更前后保存 logits、各层方差和路由分布；历史权重继续使用历史配置。

### 3.3 已测正确性与尚未覆盖的部分

此前使用从官方抽取的 Transformer/Block/Attention/Compressor/Indexer/Gate/MoE/Head，在 tiny 3 层、FP32、无量化路径做同权重比较；长度跨过 4/64/128/255 等边界，最大 logits 差约 **8.2e-8**。稀疏 attention 与 Sinkhorn 使用官方数学的 PyTorch 参考替代硬件 kernel；该实验不是完整旗舰量化推理对照。

下一轮补：12 层 BF16 forward/backward、HCA128 完整边界、document reset、mask 全空、长序列位置、dense/sparse teacher 对齐、mHC double-stochastic 行列和、cache ring wrap、视觉分支。当前短测试通过不覆盖这些新增路径。

### 3.4 词表与参数配置

默认保留 64K tokenizer、untied head 和现有层宽。32K 可减少约 3,355 万 embedding/head 参数，但会改变 hash token-to-expert 表、特殊 token、教师与数据编码。正式 PT 前做等字节/等墙钟 pilot，选择后一次冻结。hash table 需根据最终词表重建，并保存构造 seed 与 SHA；不能对新 tokenizer 复用旧 tid2eid 表。

## 4. DeepSeek 专属优化与路由训练

### 4.1 Muon 不能照搬 Qwen 的 8 步 Polar Express

DeepSeek-V4 的 Muon 使用自身 Hybrid Newton–Schulz：前 8 次系数 `(3.4445,-4.7750,2.0315)`，后 2 次 `(2,-1.5,0.5)`，momentum 0.95，Nesterov，update shape scaling 系数 0.18；embedding、prediction head、RMSNorm 用 AdamW。官方 AdamW 为 beta `(0.9,0.95)`、eps `1e-20`、WD `0.1`。[报告 §2.4、算法1、§4.2.2][R1]

新增 `/workspace/MiniFrontier/minifrontier/training/minideepseekv4_optim.py`：以数学上独立矩阵定义分组，专家轴逐专家，必要的 fused projections 拆语义子块；对 norm/head/embedding 与非矩阵参数显式处理。其“与 AdamW LR 相容”的缩放规则不能和 Qwen 的原始 Muon LR 0.01 混用。

mini pilot 搜索峰值 LR `{1e-4,2e-4,3e-4}`，batch `{16K,32K}` 输入位置；固定 20M–50M CE token，比较当前 AdamW 与此 Muon。Muon's WD 0.1、momentum 0.95 作为起点；Adam eps 同时比较 1e-8 与 1e-20 的数值稳定性，FP32 状态并测试零梯度/极小梯度。选择不同 eps 时标为 mini 稳定性适配，不声称逐项等同官方。

官方有 warmup→较长 plateau→cosine cooldown 的学习率安排，并有 batch 增长。本方案用 token 而非照抄官方 2,000 updates：1%–2% token warmup，主要预算 plateau，最后 10%–20% cosine 到 0.1 peak；输入 batch 8K→16K→32K 的爬升先在 pilot 比较，正式记录每段 CE 分母。不同于 Qwen 默认恒定 batch 的建议。

### 4.2 Auxiliary-loss-free bias 与 sequence balance

当前通用 RouterBalance 只有 count-based sign bias update，缺少报告中的轻量 sequence-wise balance。建议保留 DeepSeek 的 bias speed 0.001 为起点，统计一次 optimizer update 的全局有效 token，排除 padding 与重复重算；在下一步生效，推理冻结 bias。[报告 §4.2.2][R1]

新增 sequence loss，以一个真实样本内的 routed 专家分配统计为单位。常见表达为：

```text
f_i = n_experts/(top_k*T) * sum_t 1[i in TopK(unbiased_score[t,:])]
P_i = mean_t normalized_unbiased_router_score[t,i]
L_seq = alpha * sum_i f_i * P_i, alpha 起始 1e-4
```

此处按V3公开sequence loss公式使用未加bias的TopK统计，区别于sign-bias控制器统计的实际执行路由；二者必须分别命名。[V3公式17–20][R10] V4 score_func需按发布的sqrtsoftplus取值再对全部专家归一化，不能把raw score直接当概率；selection bias不混入混合权重。`T`是该样本真实有效token，prompt与图像位置不能因为无CE被当padding。packed数据按原sample而非整张packed sequence算序列平衡。hash路由层没有同样可优化的内容路由选择，不强行施加不相容的梯度项；分别记录hash与learned-router负载。[V4路由描述][R1]

验收包括单样本/多样本、完全失衡、均衡、全 padding、DDP/累积、checkpoint 重算。记录 token load CV、max/mean、router entropy、死专家与输出幅度；不能通过丢弃困难 token 获得平衡指标。

### 4.3 稳定性与 Anticipatory Routing

保留 SwiGLU clamp 10 的源行为，监控 gate/up outlier、mHC mixing matrix 与 attention logits。报告中的 Anticipatory Routing 用历史模型在当前批数据上预计算的路由 indices，必要时触发 rollback 后启用；不是把上一批 token 的路由拿来给下一批用。[报告 §4.2.3][R1]

本项目应完整设计该可选故障恢复机制：提前固定未来 batch id，缓存该 batch 在 `theta_(t-delay)` 下的 per-layer routing indices；当前 `theta_t` 计算特征但使用对应缓存 indices；cache 包含 token/图像内容 hash、router 版本、层号和 padding。先用 delay=1 的 tiny oracle，再试 2/4；多模态 processor 增强必须也固定，否则缓存与实际输入不同。

它只在确认 routing 相关 spike 且常规数据/数值诊断后启用，第一轮稳定时保持关闭；这保留完整方法且遵循触发式使用，不为凑齐名词平白增加每步前向。触发、回退、恢复普通路由的条件和额外成本必须记录。

## 5. CSA/HCA 训练课程与 MTP

### 5.1 dense 起步的真实含义

V4 的 dense warmup 不能直接替换成一个普通全 token Transformer。SWA、compressed KV、mHC 等依旧按结构存在；CSA 的 dense 模式先允许所有符合因果性的压缩候选，再学习 indexer，最后 top-k 稀疏选择。HCA 保持高压缩、全可见已完成块的路径。[报告注意力与训练段][R1]

mini index_topk=8 与压缩率4意味着到一定长度才有真正检索竞争；长度256对HCA128只有很少完成块，无法检验有意义的长程选择。建议主课程512→1K→2K→4K；每层记录完整压缩块数与选中数，不能把 max_seq_len=4096 配置值当实际训练覆盖。

### 5.2 Indexer warmup 与联合稀疏训练

在有可用主干表示后冻结主干，仅训练 CSA indexer 使其拟合 main attention 在压缩候选上的分布；teacher stop-gradient，indexer 的输入主干 hidden detach。遵循 DeepSeek 的 candidate/score/pooling 定义，**不复制 Qwen 的 block-max teacher 构造**。

实施时导出每个 query 的合法候选和 teacher probability，剔除 SWA 与压缩槽位之间的错误重复计数；teacher 的分母与 indexer 候选集合一致。当前已有 dense_distill/sparse_cpt 路径，应补参考对照、有效 token 归一化和更长数据，不是重命名已有命令就完成。

mini warmup 建议 10M–30M 有效输入位置，indexer LR `{3e-4,1e-3}`，warmup 2%，KL 按真实 query 归一化。稀疏 PT 联合 LM/MTP/平衡损失及 indexer KL；保留候选 mask、梯度分离，单独记录 KL/LM 和 top-k 命中质量。

### 5.3 原文本 MTP

V4 报告沿用 V3 的 MTP 机制：用当前隐藏表示和下一步 token embedding 构造更远位置预测，共享词表并保持因果性；报告给出主期系数 0.3、冷却时 0.1。[V4 报告][R1] [V3 技术报告][R10]

建议mini文本线先实现一个额外预测block，保留HC/attention/FFN结构。按V3明确共享主干token embedding与LM head，独立建立两个输入RMSNorm和融合矩阵`M:1024→512`；`h_i`与`Emb(t_(i+1))`融合后预测`t_(i+2)`。HC流的读出/扩展按V4 block接口核对，不把4个流直接拼进共享LM head。[V3 §2.2][R10] 主预算默认lambda0.3，pilot必须包含0/0.1/0.3检查是否压过主任务；若缩配后选择0.1则标本地修改。mask不能跨EOS/文档/图像占位位置；每深度单独按有效监督归一化，不能把MTP重复监督加进唯一CE数据预算。

当前 Vision-Exp 中 `mtp.*` 存的是 DSpark 三阶段，不能拿这些权重命名反推它等于一个原版 MTP block。文本 MTP 与新 DSpark 要有不同配置字段和导出键映射，详见 §11。

## 6. 数据设计：文字、视觉、推理与验证闭环

### 6.1 候选源及使用方式

| 数据 | 用途 | 具体处理 |
|---|---|---|
| [Fineweb-Edu-Chinese-V2.1][D1] | 中文 PT | 按高质量目录和实际 score 刻度选择，随机审阅，过滤模板、广告、导航 |
| [FineWeb-Edu][D2] | 英文 PT | 按 dump/文档 group，分层保留自然长文 |
| [SmolLM-Corpus][D3] | 代码和合成教育材料 | python-edu 按 repo split，许可追溯；fineweb 派生数据不重复算唯一量 |
| [Infinity-Instruct][D4] | 中文/数学/代码 SFT | 自动 gated；按任务、首问和原始来源清洗，不取文件前缀 |
| [UltraChat 200K][D5] | 英文短对话和解释 | train_sft 与官方测试集分离，消除反复身份类模板 |
| [OpenThoughts-114k][D6] | 推理与代码课程 | 有 verifier 的完整短/中解答先入；长解答不能截断后当正例 |
| [ALLaVA-4V][D7] | 双语图像 caption/VQA | NC 来源隔离；图像级去重与跨 caption/instruct split 绑定 |
| [FineVision][D8] | OCR、图表、文档、截图/科学图 | 子集 allowlist、原来源许可、可验证字段、去除已有同图/同题 |
| [Docmatix][D9] | 文档 QA | 按原 PDF 分组；单页问答先核对答案页，多页保留全部必要证据 |
| [UltraFeedback binarized][D10] | 可选偏好实验 | 同 prompt、同视觉上下文的 chosen/rejected，严格处理模板/长度偏差 |

初版 Vision-Exp 方案以图片和多图为正式能力，不仅凭其视觉塔可以处理帧就声称有原生视频时间编码。若扩展视频，另外研究 processor 的时间戳、帧组与训练数据，再建明确命名的 video variant；当前官方这一分支所读接口主要是图像。

### 6.2 文本与视觉配比

文本 PT 按 CE token：中文自然/教育 40%、英文自然/教育 25%、代码 20%、数学/科学 10%、高质量对话式材料 5%。对代码保留可运行文件与依赖信息，数学保存题干、最终答案和 verifier；不把无法检验的长推理当高质量的同义词。

视觉 CPT 按图片样本：caption 35%、OCR/文档 30%、图表/计数/空间 20%、截图/工具操作理解 10%、多图/多页 5%。另有 text replay，按 CE token 目标占 30%–50%，不是把 replay 比例与样本比例相加。

文本主 PT 预算建议 **2.5B CE token**。视觉延伸预算另列 **300M CE token、1.5M 图像出现次数**；caption/文档答案的 CE 计在300M内，image feature 不计入。视觉 warmup 的监督和图片重放单列。唯一图像建议至少600K，否则记录实际重采样轮次并扩大来源；不能以同一图片多问答虚增唯一视觉数据。

### 6.3 数据工程与可复现切分

统一 schema：`sample_id/source/revision/item_id/group_id/license/lang/task/content_hash/quality/turns/media/verifier`。image 存 RGB hash、pHash、原始 id、尺寸、文档 id、processor 参数；图表/OCR 带可对齐的答案字段、单位/数值范围。先清洗再 group split，跨语言同图、同 PDF 的所有页、同 repo 的代码落同一 split。

旧 SFT 的首问偏置必须单独修：规范化首问聚类，每组最多3条真正不同的优质答案；身份/意识类助手 token ≤0.5% 起测；短寒暄保留但限频。超过长度的样本转长桶或拒收，完整答案和EOS不得伪造。新 reservoir 默认只是减少源文件排序偏置，不能替代领域配比和模板去重。

预训练周期验证保留约5M CE token，覆盖中英/代码/数学/长文；SFT固定2K独立问题组；视觉固定1K图像/文档组，最终测试另封存。代码单测、图表答案和公开基准题必须与训练数据做题干/图像/源仓库去污染。

packing 必须同时隔离SWA/CSA/HCA的跨token状态和压缩组，mHC只在各自token内完成层间混合；EOS不是自动清空cache的指令。暂未实现segment-aware压缩/attention时，单条sequence只装一个连续文档。视觉整个N-layout图像块不能跨sequence切开，补齐产生的图像pad也不能当有效语言CE。

tokenizer选定后全流程冻结，文本teacher、视觉student和DSpark用同一词表。图像sentinel与词表内特殊token是两套概念，按§8实现，不能从65536任意扩成65541就算完整支持视觉。

## 7. Text-v2 的完整预训练预算

| 阶段 | CE预算 | 长度/attention | 参数与验收 |
|---|---:|---|---|
| D0 正确性与pilot | 0.5–2M调试，另20M–50M配方pilot | 边界长度、512/1K | window128、Muon、MTP、seq loss、mask逐项验证；不计正式主预算 |
| D1 dense起步 | 250M | 512→1024，全部合法compressed候选 | 训练主干/MTP/路由；建立各领域语言建模能力 |
| D2 dense主预训练 | 500M | 1024→2048 | 长文、代码、数学课程，检验HCA实际使用 |
| D3 indexer warmup | 10M–30M输入位置，另计 | 2048，部分4096 | 冻结主干，仅indexer；候选/teacher KL与梯度对齐 |
| D4 sparse主预训练 | 1.50B | 2K→4K，CSA top8 | LM+MTP+seq balance，独立indexer KL；按需比较top8/16 |
| D5高质量冷却 | 250M | 1K/2K/4K混合 | LR cosine至0.1 peak，MTP系数从0.3到0.1 |
| 合计 | **2.5B CE** | D3不计主干LM预算 | 产出可供SFT/视觉扩展的text base |

2.5B约为旧3.6768M主干预算的680倍，属于首轮可调整计划，不是按参数量推算出的“学会说话最低定理”。MoE总参数、每token激活参数、巨大词表与共享参数的样本效率不同；不能直接把dense模型的某个tokens/parameter常数作为精确门槛。

如果100–200M后文本仍严重重复，暂停后续：查数据标签、LR、route_scale、专家负载、输出头、mask，并与同数据同预算的dense小对照比较。若2.5B时验证与任务持续改善，可追加1B并保留阶段分界；不得用追加数据回避确定性实现错误。

损失组合：

```text
L_total = L_CE + lambda_MTP * L_MTP
          + L_sequence_balance + lambda_index * L_indexer
```

indexer warmup只保留最后一项。bias更新不混入梯度损失；teacher与主干输入在indexer KL路径detach。各项独立归一化有效位置，累计整个optimizer窗口再缩放。图像延伸后同样区分input位置/CE位置/有效路由位置。

## 8. Vision-Exp 迁移：不是加一个 projector 就完成

### 8.1 视觉塔与aligner

官方Vision-Exp配置：vision 32层、dim1024、16heads、inter2816、patch14、downsample3、feature预算384，RGB mean/std0.5。视觉块是RMSNorm、2D RoPE和SwiGLU；aligner将3×3邻域拼为9倍视觉维度，再经到语言维度的两层投影/GELU。[视觉源码][R6] [图像处理][R7]

mini初版建议：12层、vision dim384、6heads（head64）、inter1056，保留patch14、downsample3，aligner输出512。保留原生ViT、padding和N-layout，不能换成MiniMind-V的SigLIP或Kimi的2×2merger还叫Vision-Exp结构复现。

resize按feature预算与宽高比控制，不固定把所有图拉成正方形。先比较96/192/384个feature的课程，遵循原processor的min pixels约束；若小预算与原min pixels冲突，明确调低mini的min_pixels并记录，不让两个约束互相覆盖。最终384feature是视觉内容预算，**整段语言侧图像span还含start/end/newline/pad，可能超过384**，总长度必须从processor实际输出计算。

### 8.2 超词表sentinel与embedding注入

源码将图像token types组织成 `vocab_size + type` 的逻辑id，并用vision features/learned image embeddings替换；这些不是普通词表softmax目标。当前batch validator只允许`input_ids<vocab_size`，因此需要设计独立的多模态输入验证和安全embedding lookup，而不是把这些id直接送给原词嵌入。[processor][R7] [model forward][R5]

保留逻辑token types用于路由和可见性；普通word embedding仅对词表内位置查表；图像features和边界embeddings按N-layout插入，之后再按原路径进入mHC。图像pad的语义不同于batch padding，两者mask不能复用一个`input_ids!=0`。

图像内容feature的行列排列必须与processor给出的aligner-row order匹配；使用不同颜色格子/印有坐标的合成图做逐位置对照，比仅检查feature数量更有效。多图之间保留独立start/end，bad image或缺feature直接报错。

### 8.3 图像内可见性与路由

官方 `get_image_visible` 与 `get_window_topk_idxs_visible` 为同一image span提供额外左右可见范围。不能让旧纯因果SWA mask直接覆盖它，也不能为了图像双向可见而让文本query读到未来答案。图像完整块需同一prefill段处理；chunk边界不能切断未完成image span。[源码][R5]

router新增`bias_vl`：图像位置走视觉偏置，前面的hash层对图像位置改用内容top-k，而文本仍用tid2eid。把超词表imageid送入hash表会越界；简单取模虽不报错，也改变了官方语义。[Gate源码][R5]

训练端必须新增文本与视觉两套bias统计、有效token掩码以及恢复状态。报告未公开Vision-Exp的完整视觉平衡训练细节，建议初版分别按有效文本/图像计数做下一步bias更新，步幅从0.001起、视觉比较0.0003；sequence loss是否对视觉使用相同系数用短消融确定，明确标为本地补充。

### 8.4 迁移兼容性

Vision-Exp还包含norm_eps、DSpark等版本变化，不能只复制vision.py而保持其余默认。建立迁移表：每个旧权重键→新权重键、shape、dtype、是否新初始化、是否保持等价。`norm_eps=1e-20`只在已核对RMSNorm路径且零向量/低精度测试通过后使用；不要改Sinkhorn epsilon。

只加载结构兼容的text base，初始化新vision/aligner/image embeddings/bias_vl/DSpark。该迁移不能通过`strict=False`吞掉所有不匹配：只允许明确列出的新增键，其他missing/unexpected key失败退出。新版本无图forward与迁移前文本模型做逐层差异归因；若修改theta/scale/eps，必须在报告中分开说明，不把差异归为“视觉自然影响”。

## 9. 视觉继续预训练与多模态SFT

### 9.1 三个视觉阶段

| 阶段 | 建议预算 | 训练对象 | 数据与进入条件 |
|---|---|---|---|
| V0 接口与可学习性 | 64–512图诊断 | vision/aligner/image embeddings，按测试需要解冻文本 | N-layout、sentinel、可见性、bias_vl、换图反事实；无图回归 |
| V1视觉接入warmup | 100K–200K caption/OCR样本，1轮；约10M–30M CE另列 | 随机vision与aligner训练，文本冻结 | 不能冻结随机视觉塔；检查真实图像特征梯度和文本条件化 |
| V2视觉CPT | 240M CE、1.2M图像暴露 | 全部解冻，含bias_vl | text replay 30%–50% CE，caption→OCR/文档/图表/截图 |
| V3高质量视觉冷却 | 60M CE、300K图像暴露 | 全部，降低text/vision LR | 增加多图/难OCR，保留text replay；形成视觉base |
| 主视觉延伸合计 | **300M CE、1.5M图像** | V1重放单列 | 继承2.5B text base，不能把所有阶段称从零视觉联合PT |

V1是本方案为随机mini视觉塔设计的接入阶段。若选择预训练视觉初始化，必须提供相同结构/尺寸的checkpoint与兼容证明；不能裁切官方1024维视觉塔成384维。可另做大视觉塔教师的特征蒸馏，但需要新增目标和预算，不能无说明代替原生视觉训练。

V1视觉/aligner AdamW LR从1e-4/3e-4比较；V2文本继承已验证Muon分组，峰值LR为text PT的0.1–0.3倍，视觉LR 3e-5–1e-4，aligner倍率1–2；warmup2%，V3冷却到0.1 peak。这些是局部搜索起点，依据text/vision双验证选择。

### 9.2 SFT预算与推理模式

文本线可从D5开始做文本SFT，视觉线从V3做联合SFT，二者保持独立checkpoint。联合池建议300K–600K文本指令、300K–600K视觉指令，先1epoch、最多2，实际 **100M–220M assistant token**。

assistant token配比：通用解释/对话20%、摘要改写翻译10%、代码15%、数学15%、OCR/文档15%、图表/VQA/空间15%、工具与截图操作10%。按图像/首问组限频，包含无图、图像不足、数值单位与纠错样本。长度512/1K/2K/4K，完整答案优先。

保留non-thinking、thinking-high、thinking-max三类可控模式的训练与评估接口；本地max初版response cap1024，复杂任务另开2K输出课程。训练模板、mode选择与demo共用encoding，工具消息与reasoning/final边界明确。不能通过无限延长输出补偿基础推理不足。

主干SFT峰值LR取PT最优的0.1–0.25倍率，AdamW组可从2e-5/5e-5比较，视觉塔倍率0.2–0.5、aligner1–2。每5M assistant token评估；若文本显著遗忘，调整replay与LR，不优先冻结所有视觉层或接受整体退化。

## 10. 领域GRPO与全词表OPD：不能复用Kimi式MOPD代替

### 10.1 专家策略的培养

V4报告是领域SFT/RL专家后做OPD合并，mixed RL合并阶段由OPD替代；不应写成“所有模型统一SFT→DPO”。官方使用超过十位教师，不代表本项目必须同时驻留这些旗舰模型。[报告 §5.1][R1]

本方案保留可扩展teacher registry，首先建立4类领域：数学/结构化推理、代码/小项目、通用/工具、视觉/OCR/图表；每域支持三种mode。文本线先前三域，视觉线加入第四域；每个进入registry的教师必须在自己holdout上优于共同SFT起点，否则不纳入蒸馏。领域专家是独立训练权重，不是MoE内部的32个FFN专家。

建议每域每mode 10K–30K独立prompt，group4起、最多8，1–2轮；训练预算同时记录unique prompts、rollout数和response token。单卡逐教师训练保存，避免把12位教师同时加载。任务成功率接近0或1的组没有有效相对优势，须做难度课程并报告筛除比例。

GRPO参考原方法：[DeepSeekMath][R11]。初始LR1e-6–5e-6、clip0.2、reference KL0.01，single-rollout batch最多1–2次更新。数学用答案/数值verifier，代码用独立测试，工具用终态和约束，视觉用答案字段、单位和位置误差；开放问题的裁判须有固定rubric与人工校验，不以更长推理直接加分。

### 10.2 现有rollout要先修

当前 `minifrontier/training/rollouts.py` 使用默认生成分布，但old_logp由原始logsoftmax重算，temperature/top-p/特殊token mask未保证一致。首轮改为temperature1、无top-p截断、统一合法token mask，采样时保存行为logprob；training/old/reference用同一概率定义。

工具observation、prompt、image占位不计policy action loss；EOS计入真实动作，超长截断与正常结束分开。rollout时不更新balance bias、norm统计或dropout。轨迹存policy hash、mode、task/media id、tool输出、奖励拆解和终止状态。相同输入及token支持集下，冻结policy的ratio应接近1；这是进入RL的硬测试。

### 10.3 full-vocabulary reverse KL

DeepSeek-V4明确使用学生自己的轨迹，并在各token状态上计算 **全词表** reverse KL；报告指出采样token概率差的估计方差较大，因此采用全分布蒸馏。[报告 §5.1.2][R1]

```text
对于学生生成的状态 s=(prompt, generated_prefix)：
L_OPD(s) = sum_v p_student(v|s)
                 * [log p_student(v|s) - log p_teacher(v|s)]
L = 按domain/mode权重，对有效response位置的 L_OPD 求平均
```

teacher分布stop-gradient，student概率与log概率参与梯度；token序列本身作为本轮已采样轨迹固定。当前 `mopd_advantages` 只取采样token的logp差，不能代表这个full-vocab目标。新增 `/workspace/MiniFrontier/minifrontier/training/deepseek_opd.py`，不要仅在CLI把mopd改名opd。

需要同一tokenizer、特殊token和合法输出支持集。mini教师来自同一个base/SFT系谱；若外部官方旗舰词表不同，不可直接对齐logit编号。外部模型可生成经校验离线SFT数据，或单独研究跨tokenizer蒸馏，但那不是本方案的精确logits OPD。

### 10.4 3090上的完整分布计算

先学生生成短轨迹，按domain/mode分组，再顺序加载教师计算hidden或logits，最后学生训练。教师冻结，缓存绑定teacher hash、tokenizer/processor、轨迹hash，不能使用教师自己的不同答案替代同一学生prefix。

优先按**序列位置分块**计算full-vocab KL：例如64/128个位置一次，始终保留该位置完整65536词表的归一化，累计梯度。若进一步沿词表分块，必须用全局logsumexp的精确归一化与等价梯度，两遍/重计算实现；不能每块各自softmax，那不是原KL。top-k logits压缩只能作为明确的近似对照，不能宣称完成full-vocab OPD。

可缓存FP16/BF16 teacher最终hidden与固定LM head，在训练时重建完整logits；需包含teacher norm/head版本。hidden缓存比全词表logits小，但要验证重建误差、量化格式与teacher输出一致。单卡不要把所有teacher optimizer state驻留。

建议合并预算 **30M–80M学生response token**，LR1e-6–5e-6，domain/mode先均衡再按holdout调整。teacher必须在该任务上有优势；学生比教师强的区域不机械加大蒸馏权重。分别评估各域、各mode、文本/视觉，合并后若某域退步超过事先容限，调整数据权重或恢复该教师后再蒸馏，不用总均分遮盖遗忘。

### 10.5 DPO的独立对照

DPO仅在语言与任务能力达标后作为10K–30K偏好对的小实验，beta0.03/0.1、LR5e-7–2e-6，reference固定为进入阶段的checkpoint。[DPO][R12] 同prompt同图像、无答案截断、长度偏差审计。保留SFT与OPD作为demo候选；DPO最新不等于最好，本次旧DPO的NLL退化必须写入选择规则。

## 11. QAT与DSpark完整补齐方案

### 11.1 DeepSeek的QAT范围不同于Kimi

V4后训练QAT不仅涉及MoE expert权重MXFP4，还涉及CSA indexer的QK路径FP4，以及index scores降为BF16；teacher、reference、student和rollout要一致。[报告 §5.2.1][R1] 不能只做expert INT4后宣称完整复现。

实现FP32 master、符合MXFP4格式的量化/反量化、STE；attention、norm/mHC、router敏感路径保留适当高精度。官方FP4→FP8实现依赖硬件与scale条件，3090以BF16承载仿真计算并测试相同数值目标；没有原生FP4/FP8加速承诺。

先从SFT起点跑5M token BF16/QAT对照，覆盖文本、OCR、indexer选择与长上下文：记录CE、路由变化、top-k候选recall、QK误差、置信度和吞吐。index score BF16的ties需要确定性处理。过关后正式SFT→RL→OPD全程使用相同QAT；若尚不具备可靠仿真，则产物标BF16分支并保留后续重跑计划，不在最终导出临时量化后声称“已QAT”。

### 11.2 Vision-Exp的DSpark结构

发布Vision-Exp的`n_mtp_layers=3`、block_size5、target layer ids `[40,41,42]`、Markov rank256，以及专用noise token。`mtp.*`命名空间里实际有DFlash/DSpark attention、主干特征融合、Markov head、confidence head；不能等同于原文本MTP一个block。[官方推理源码][R5]

建议mini迁移保留3个draft阶段与block_size5，target taps改为12层主干的`[9,10,11]`，Markov rank先64；noise token从最终mini词表预留。它们都是缩配参数，必须与主干hidden/mHC维度匹配。主干完成OPD/QAT后冻结，再训练draft及其融合/Markov/confidence；不能让未学成的draft反向扰动主模型语言能力。

### 11.3 官方DeepSpec训练代码可以直接作为训练参考

本次已检查官方DeepSpec的DSpark trainer、loss和Markov实现。仓库有Qwen3/Gemma等目标模型适配，并不意味着已经提供MiniDeepSeek的adapter。应复用其训练目标、anchor采样和block mask，同时新增本项目mHC与Vision-Exp特征接口。[DeepSpec固定版本][R8]

在已读代码中，draft训练组合包括：

```text
L_draft = alpha_CE * CE(target_tokens)
          + alpha_L1 * sum_v |p_draft(v) - p_target(v)|
          + alpha_conf * BCEWithLogits(confidence,
                               stopgrad(1 - 0.5*L1_distance))
position_weight[j] = exp(-j / loss_decay_gamma)
```

target概率来自相同prefix，confidence标签是分布overlap构造的软目标；不是随手用“某次greedy是否相同”替代。已读公开Qwen3参考配置为CE0.1/L10.9/conf1、gamma4、block7；mini可把这些损失权重作为起点，但block5/三阶段和LR必须适配，不能声称直接复制了旗舰Vision-Exp训练超参数。[官方DSpark loss][R13]

Markov head在block内部依赖前一个token，用teacher forcing训练相应转移；draft backbone保持其并行mask。anchor不能跨文档/图像/答案终点；不允许从target未来hidden泄漏进当前draft输入。采样target回答并构建target cache后，建议 **10M–30M有效draft位置**，LR1e-4–6e-4短pilot，按留出acceptance、confidence calibration与真实延迟选择。

target cache优先存所需层hidden与head版本，不默认照搬官方多GPU大缓存配置；1M位置、3×512维BF16 hidden约3.07GB十进制，不含metadata，10M约30.7GB，必须先估磁盘。图片特征和positions也要可恢复；不同mode与视觉任务都在cache里，否则只在普通聊天训练出的draft不保证适合推理/看图。

### 11.4 调度与无损验证

先做固定draft长度1/3/5的accept/reject验证，再按置信度和3090实测成本选择prefix。遵循DSpark的prefix调度条件：不能查看后面的已采样token后，选择性丢弃前面不喜欢的结果而破坏目标分布；使用官方early-stopping/调度定义。[DSpark报告][R9]

测acceptance逐位置曲线、expected accepted prefix、confidence校准误差、draft/verify时间、最终token/s与首token延迟。实现拒绝后的SWA ring、CSA/HCA未完成块、mHC和视觉prefill状态回滚。若mini主干本来就小，draft开销超过节省，保留训练实验但demo默认普通增量生成，不将“理论无损”与“实测更快”混为一谈。

## 12. 单张3090的预算与训练器要求

当前短文本batch2、长度256的一次完整更新约 **4.36GiB** 峰值；不代表新视觉、4K、MTP、OPD全分布和DSpark同时训练可行。按FP32参数/梯度/Adam两状态粗略`16P`，现有文本约 **3.43GiB**，其余来自激活、logits、临时tensor和不同优化器分组；加入视觉与draft后应重新统计所有trainable/frozen参数。

4K×64K FP32 logits单份约1GiB，teacher/student/full-vocab KL会有多份；沿位置分块、head重计算、activation checkpointing、稀疏候选分块是优先项。SWA128相对64会增加局部计算，但应先保证覆盖正确；不能为了显存省一点保留已知语义缺口。

text microbatch1–2，image microbatch1；按总语言位置与视觉patch数分桶，gradient accumulation实现16K/32K有效输入batch。CE在整个累积窗口sum后除以有效target数，DDP考虑gradient averaging；router统计按真实非padding位置。MTP/seq loss/indexer KL各有自己的正确分母，不把不同loss的microbatch mean直接平均。

每个阶段/长度/模态配比，先50–100次warmup再200次真实updates，计CE token/s、图像/s、峰值allocated/reserved、optimizer/QAT/KL时间、I/O。单卡预留约2GiB余量；teacher/student/optimizer状态按阶段驻留，不按所有模型权重总和去占满显存。

假设500/1000/2000 CE token/s，2.5B文本PT约57.9/28.9/14.5天；300M视觉CPT约6.94/3.47/1.74天只是同速率算术示例，实际视觉速度应另测。加上SFT、领域rollout、full-vocab OPD和draft，完整计划按周到更长时间规划。旧双卡短训练约734 CE token/s不能当成新单卡长上下文速度。

checkpoint保存模型/optimizer/scheduler/RNG、sampler cursor、实际token counters、bias文本/视觉、hash表、stage、compressed attention config、processor/tokenizer、teacher registry hash。same-stage resume恢复全部；stage transition明确继承权重和重置状态。Anticipatory Routing若启用还需保存未来batch与indices，恢复时不得错配。

## 13. 评估、阶段门槛与demo

| 层级 | 必做检查 | 建议验收规则 |
|---|---|---|
| 文本结构 | window128覆盖、压缩边界、mHC行列和、hash表、dense/sparse同权重 | tiny FP32与参考一致，BF16单独容差，未来扰动不影响过去 |
| 数据/接口 | labels、EOS、完整答案、segment、图像sentinel/N-layout | 200条可读样本逐项检查，不能有silent越界/丢图 |
| 可学习性 | 文本与64–128图小样本记忆、换图/遮图 | 学得会且图像变化影响对应回答；不把记忆当泛化 |
| base | 按中英/代码/数学/长文的NLL和续写 | 100–200M后仍明显失败则诊断，禁止自动进入RL |
| SFT | 固定500文本问题与1K视觉组 | 基础指令完成率起始≥80%、严重重复/乱码≤5%，人工和规则共同判断 |
| 视觉 | OCR CER、文档ANLS、图表数值/单位、VQA、位置误差、多图指代 | 好于无图/shuffle对照；有冲突图时依据正确图，不靠题干猜答案 |
| RL/OPD | 每域每mode独立任务、text replay退化、KL/奖励/长度 | paired holdout提升，不能用overall分数掩盖某域显著遗忘 |
| QAT/DSpark | BF16对照、indexer recall、acceptance、校准、分布/回滚 | 格式一致且质量过关；加速需要真实端到端测量 |

阈值为本地release gate，须冻结数据后执行，不是旗舰榜单复现。NLL必须同词表同标签比较，关键方案多种子或bootstrap检查不确定性。困难公开榜单可以做诊断，不适合用几个全部失败的小样本决定训练方向。

demo分别列Text-v2与Vision-v1，默认加载通过capability的checkpoint，标注已训练长度、支持的图像数量和mode。导出tokenizer/encoding、image processor、hash routing表、bias_vl、量化配置、可选DSpark、权重hash与能力报告。输入坏图、图像span超限、cache失效都显式报错，不能回退纯文本后假装看过图。

## 14. 按文件的修改方案与依赖

“新增”路径均为设计，不表示当前文件或CLI已存在。

| 优先级 | 文件/模块 | 工作与完成证据 |
|---|---|---|
| P0 | `/workspace/MiniFrontier/configs/minideepseekv4.json` | v2显式window128、route_scale与theta选择；不覆盖历史run config |
| P0 | `/workspace/MiniFrontier/minifrontier/models/minideepseekv4/attention.py` | 覆盖边界、segment-aware压缩、dense/sparse teacher、cache协议 |
| P0 | `/workspace/MiniFrontier/minifrontier/training/runtime.py` | 一次update统计、padding排除、重算去重，分开Kimi QB与DS sign-bias |
| P0 | 新增 `minifrontier/training/minideepseekv4_optim.py` | 10步hybrid NS、独立矩阵语义、RMS缩放与参考更新 |
| P0 | `/workspace/MiniFrontier/minifrontier/data.py` | 新语料版本、模板/图像group去重、完整回复、真实预算 |
| P1 | 新增 `minifrontier/models/minideepseekv4/mtp.py` | 原文本MTP、shifted mask/损失；区别DSpark namespace |
| P1 | `/workspace/MiniFrontier/minifrontier/models/minideepseekv4/expert.py` | sequence balance统计、Vision bias_vl、image位置绕开tid2eid |
| P1 | 新增 `minifrontier/models/minideepseekv4/vision.py`、`processing.py` | 原生ViT/3×3aligner/N-layout/visible mask；迁移白名单 |
| P1 | `/workspace/MiniFrontier/minifrontier/models/minideepseekv4/modeling.py` | 图像逻辑id安全embedding、HC前注入、dtype/eps/config版本 |
| P1 | 新增 `minifrontier/training/deepseek_opd.py` | 学生轨迹、full-vocab reverse KL、position chunk、教师缓存与对照梯度 |
| P1 | `/workspace/MiniFrontier/minifrontier/training/rollouts.py` | 行为logprob一致、mode/media/tool masks、领域teacher registry |
| P1 | 新增 `minifrontier/training/deepseek_qat.py` | expert与indexer QK的MXFP4仿真、BF16 scores、同格式rollout |
| P2 | 新增 `minifrontier/models/minideepseekv4/dspark.py` 与 `training/dspark.py` | DeepSpec目标和mini adapter、Markov/confidence、3阶段block5 |
| P2 | 新增 `minifrontier/training/anticipatory_routing.py` | spike触发恢复、未来batch路由cache及hash校验 |
| P2 | `/workspace/MiniFrontier/minifrontier/inference.py` | 普通增量→视觉→DSpark回滚/调度，实测能力与速度 |

依赖顺序：固定基准与词表 → window/数据/优化/路由/MTP正确性 → Text-v2 PT → 视觉迁移与CPT → SFT/QAT → 领域GRPO → full-vocab OPD → DSpark → demo。文本SFT可以作为独立阶段产物交付，但不能用它替代视觉路线与后续算法的完成证明。

## 15. 数据版本与资料索引

| 数据集 | 固定revision | 数据卡许可入口 |
|---|---|---|
| Fineweb-Edu-Chinese-V2.1 | `a5b574efa48beb3a8f6887ef0b093becf004328b` | Apache-2.0 |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-BY |
| SmolLM-Corpus | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-BY，代码追溯原许可 |
| Infinity-Instruct | `bddc39a8feadbd679c30623197f4e736b7e75b48` | CC-BY-SA-4.0，自动gated |
| UltraChat200K | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT |
| OpenThoughts-114k | `bd093c3994fd54d2390985b66988ddf282a55eb6` | Apache-2.0 |
| ALLaVA-4V | `0fd42fce5c047d387a4bb5318d588eae9a9797f0` | CC-BY-NC-4.0 |
| FineVision | `3c380a731a3429c1d04693d6ec16d7e683def84c` | 逐子来源，无顶层统一许可 |
| Docmatix | `0725b65616e0e5f6024be10e38ddf8d8c48664fd` | MIT卡片，原文档来源单列 |
| UltraFeedback binarized | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT |

[R1]: https://arxiv.org/html/2606.19348v1
[R2]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1
[R3]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json
[R4]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/6821d6ad3681a4b137b066b76094fa82ebd0a380
[R5]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/model.py
[R6]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/vision.py
[R7]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/image_processor.py
[R8]: https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2
[R9]: https://arxiv.org/html/2607.05147v1
[R10]: https://arxiv.org/html/2412.19437v1
[R11]: https://arxiv.org/abs/2402.03300
[R12]: https://arxiv.org/abs/2305.18290
[R13]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py
[D1]: https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/tree/a5b574efa48beb3a8f6887ef0b093becf004328b
[D2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
[D3]: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f
[D4]: https://huggingface.co/datasets/BAAI/Infinity-Instruct/tree/bddc39a8feadbd679c30623197f4e736b7e75b48
[D5]: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a
[D6]: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/tree/bd093c3994fd54d2390985b66988ddf282a55eb6
[D7]: https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/tree/0fd42fce5c047d387a4bb5318d588eae9a9797f0
[D8]: https://huggingface.co/datasets/HuggingFaceM4/FineVision/tree/3c380a731a3429c1d04693d6ec16d7e683def84c
[D9]: https://huggingface.co/datasets/HuggingFaceM4/Docmatix/tree/0725b65616e0e5f6024be10e38ddf8d8c48664fd
[D10]: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/tree/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85
