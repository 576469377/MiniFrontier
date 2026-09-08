# MiniQwen4：Qwen3.8-Flash-Next 全流程训练与结构改造方案

调研日期：2026-09-08。检查对象：175、`docker_jilin`、`/workspace/MiniFrontier`。本文为下一轮实施设计，不表示文中新增模块已经实现或训练完成。

## 1. 核心判断与命名边界

MiniQwen4 当前已实现较完整的文本架构，以及按报告构造的 QSA 两阶段训练目标和语义分组 Muon。下一轮最重要的工作是：**重建训练数据并给出真实 token 预算；补原生视觉塔、三维位置编码和 PLE 输入连接；让 QSA 在有意义的长序列上学习；补 MTP 与完整推理验收。** 继续把 base、SFT、DPO 各跑少量步数，不能解决目前模型不说人话的问题。

上游公开模型名为 **Qwen3.8-Flash-Next**，相关 Transformers 类型名为 `Qwen4Exp`。项目沿用 MiniQwen4 可以帮助对应源码，但不宜把它写成“官方已正式命名的 Qwen4”。本方案固定实际模型与代码版本，不按名字猜结构。

本文的 **官方报告/发布源码/当前实测/本方案建议/待验证** 是不同证据等级。除明确注明官方的数值外，所有 mini 预算、数据配比、阈值、学习率搜索和视觉缩放都是本方案建议。公开报告侧重架构、稀疏化与优化研究，并未提供足以逐项复制的完整多模态数据清单和后训练生产配方；下文对这些空缺给出工程设计，不冒称官方训练复现。

### 1.1 固定证据

| 来源 | 版本与用途 |
|---|---|
| [Qwen3.8-Next 技术报告][R1] | arXiv `2608.30320v1`；架构、QSA、Muon、训练超参数研究 |
| [官方模型配置与 processor][R2] | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| [Transformers Qwen4Exp 源码][R3] | `4177486a9f199bd7be520eff14431071d5d41ec5`；项目当前移植来源 |
| [vLLM 项目维护的 Qwen4Exp MTP 推理实现][R7] | `96eccb8f49aff58aa2a11b431bf8331f4b368606`；补充核对MTP残差融合、多流输出和权重映射；不是Qwen公开训练器 |
| 本地配置 | `/workspace/MiniFrontier/configs/miniqwen4.json`；SHA256 `b7e7ba00e2df4d815ce044b1dce698c473275c63ee9dedfa7f58c498acc26d5b` |
| 实际训练与能力 | `/workspace/MiniFrontier/docs/training-failure-v1.md`、`docs/audits/miniqwen4-capability-v1.json`、`docs/audits/miniqwen4-memorization-v1.json` |

当前 Git 仍把源码列为未跟踪文件；长训练前需固化可追溯提交、配置和数据 manifest。本文不是基于一个已经提交且封版的训练仓库作出的完成声明。

## 2. 当前训练失败的证据与量级

当前包含 untied LM head 的参数总量为 **431,609,632**；去掉该输出头为 398,055,200。16 层、hidden 512；64 个 routed experts，top-4，另有 shared expert。embedding 与输出头合计 **67,108,864** 参数。PLE 的大 n-gram 表也占显著容量，不能只用激活专家数判断数据需要量或训练显存。

旧训练：base 1000 updates，8 sequences/update，长度 256，只有 **2,040,000 next-token 目标**。dense indexer distill 100 updates 冻结主干，不计入主干 LM token 预算。sparse CPT 200 updates、长度 1024，额外 **1,636,800** 个 LM 目标；合计 **3,676,800**。SFT 仅 4,000 条样本、532,176 assistant token；DPO 800 对。

完整同一留出集上，SFT 对话 NLL **6.453**，DPO 后 **8.041**，DPO greedy 退化为连续换行。数据有严重的首问模板偏置：29,449 条 SFT 仅 11,865 种首问；长度 256 又截断了 4,597 条。当前失败不能只归结为温度、demo 前缀或某个尚未实现的旗舰模块。

最新单独的 8 问答记忆实验已经完成：**8/8 精确复述、NLL 0.0002087**。早先复盘正文仍写 Qwen 在运行，实际应以新的 `miniqwen4-memorization-v1.json` 为准。这是可学习性证据，不是正常对话能力证据。

新的全文件抽样、预算显式化、均匀验证、DPO 默认关闭和 capability 标记已落实，但没有由此产生新可用权重。本方案新建数据版本和 run；旧失败 checkpoint 用于对照，不作为正式强基座。

## 3. 文本架构审查与修改方案

### 3.1 当前骨架

| 结构 | 当前 mini | 建议 |
|---|---|---|
| 混合注意力 | 12 GDN + 4 全局/QSA，3:1 | 保留；不要改成全 dense 或普通线性注意力后还宣称同架构 |
| GDN | key heads 4、value heads 12，head dim 64，conv kernel 4 | 检查 value/key head 扩展、门控、归一化、recurrent state 与 chunk kernel 的对应 |
| 全局 attention | 8 Q heads、2 KV heads、head dim 64，输出 sigmoid gate | 保留 Q/gate 的 fused 参数语义；优化器不能把 gate 当 Q head 一起正交化 |
| Gated Residual | 4 streams、lowrank 128 | 保留读/写门控及其归一化；这些低秩投影走 AdamW |
| PLE | 第 2 层，ngram 3，heads/gram 2，base vocab 32768，embedding 512 | 保留；不是普通位置 embedding；需额外记录 n-gram hash 和词表版本 |
| MoE | 64 选 4，expert mid 192，shared mid 192 | 保留源代码的 router/auxiliary loss 语义；不引入 Kimi QB 或 DeepSeek 固定 sign bias 作为默认替代 |
| QSA | 4 query heads、1 KV head、index dim 32、compress 4、token budget 512 | 保留第一版；用 ≥1024 的真实序列训练并统计稀疏程度 |
| RoPE | theta 1e7，partial factor 0.25，实际 rotary dim 16 | 文本成立不代表多模态三维位置正确；需要 §3.3 的适配 |
| MTP | 未实现 | 增加训练分支；config 中声明预测层不等于发布 Transformers 已实现 MTP |

当前 QSA 的 eager 实现已经对 teacher/input 做 stop-gradient，并把 teacher head sum/L1、block max pool/L1、KL 方向写入训练目标。应在此基础上做性能优化及视觉适配，不能重复把已实现目标列为“从零补齐”。

`minifrontier/models/miniqwen4/cache.py` 现在已有 **无梯度、无 padding 文本** 的 GDN/conv/KV/indexer cache adapter，并带 owner、失败状态检查。尚不覆盖视觉三维位置、beam、rollback、offload、序列化或 training-through-cache；下一轮应扩展现有协议，不再声称完全没有 cache。

### 3.2 容量与词表实验

正式 base 前做 32K/64K tokenizer 对照：比较同一批中文、英文、数学、代码、OCR 的 bytes/token、输出长度、冷门 token 使用率与实际训练吞吐。词表从 64K 改 32K 只减少一部分 embedding/head，**不会自动同比缩小 PLE n-gram tables**；两项是独立容量选择。

建议默认保留当前 431.6M 配置作为可对照主版本；另以等数据与等墙钟 pilot 比较：A 当前词表/PLE；B 32K 词表但 PLE 不变；C 在 B 上缩 PLE table。决定必须基于真实学习速度和留出生成，不能同时砍专家、层数、词表后无法定位收益来源。

新 tokenizer 会改变 PLE 的 token n-gram 哈希分布、特殊 token 和教师兼容性；如决定切换，重建数据，从新初始化开始正式 PT，不在旧 checkpoint 上部分加载后无说明继续。保留 untied head 作为源结构契约。

### 3.3 原生视觉结构与关键接口

官方配置是 27 层、hidden 1152、FFN 4304、16 heads 的视觉塔；patch 16、temporal patch 2、空间合并 2，输出到原模型语言维度。源码中包含 Conv3D patch embedding、可插值的二维位置表、视觉 rotary、patch merger；当前配置的 `deepstack_visual_indexes=[]`，不应从其他 Qwen-VL 版本随手补 DeepStack。[发布配置][R2] [源码][R3]

mini 建议值：

| 字段 | mini 初版建议 |
|---|---|
| vision depth / hidden / heads | 12 / 384 / 6（head dim 64） |
| vision intermediate | 1536 |
| patch / temporal patch / spatial merge | 16 / 2 / 2×2 |
| output hidden | 512 |
| learned position table | 保留原生二维插值实现，表大小可沿用 2304；不把它当 decoder 最大长度 |
| deepstack indexes | 空列表，与本次固定发布配置一致 |
| 图像课程 | 256→384/512，维持动态宽高比，以合并后 64/144/256 等视觉 token 桶管理 |
| 视频课程 | 4→8→16 帧；补齐时间 patch 的方式严格跟 processor，不人为省掉 temporal 维 |

这组配置尚未实际测完整训练显存；重点是保持计算关系，容量可以在 pilot 后固定。merger 应使用原生前置 LayerNorm、2×2 concat、MLP/GELU 和投影；它不是 Kimi 的输出 RMSNorm projector。[源码][R3]

**多模态必须补三维位置路径。** 当前 head dim 64、partial rotary 0.25，只有 16 个 rotary dimensions，对应 8 个基本频率。官方大模型的 mRoPE section `[11,11,10]` 不能原样用于此配置。本方案建议 `[3,3,2]`，以源码交错选择方式覆盖 8 个频率；它是缩放适配，不是官方发布数值。测 T/H/W 单轴移动、视频时间顺序、图片宽高互换、文本三轴一致、padding 与增量 `rope_deltas`。QSA indexer 必须使用同一个语义位置构造，而不是另给图像一个线性 position index。

**PLE 输入要在图像特征替换之前取得。** 上游 `Qwen4ExpModel.forward` 先从原 `input_ids` 保存 `ple_input_ids`，再把 image/video feature 写进 `inputs_embeds`，最后把两者交给语言模型。mini 应复现这个接口：保留原图像 placeholder ids 用于 PLE，不能从已经替换成任意视觉向量的 embeddings 反查最近 token。[源码 forward][R3]

视觉数据处理器输出 `input_ids/inputs_embeds或pixel_values/labels/attention_mask/position_ids/image_grid_thw/video_grid_thw/ple_input_ids/rope_deltas/segment_ids` 的明确契约。各图的 feature 数与 placeholder 一一相等，不相等直接报错。图像和 video embedding 位置的 LM labels 为 `-100`；assistant 答案正常监督。无图样本不构造黑图。

## 4. QSA：必须按两个训练目标与完整因果边界实现

### 4.1 为什么现在短训练不足以检验 QSA

`indexer_budget=512` 是保留的 token 预算，compress ratio 4 对应最多约 **128 个完整 block**。在长度 256 的序列里，所有可见 block 都容得下，大多数位置没有实际稀疏筛选。即使训练器执行了 indexer 的 forward，也不能说明检索选择学会了。

长度 1024/2048/4096 才能逐步观察预算约束；对每层统计 candidate blocks、selected blocks、未完成尾块、真实保留 token 比例。不同 query 的可见范围不同，报告一个固定 50% sparse ratio 不够。

### 4.2 QSA 目标与阶段

官方方法的 dense indexer warmup 与主模型基础预训练是两个概念。报告中的 1,000 步 dense indexer 阶段使用大批长序列，约 2B token 量级，**不是“整个从零模型预训练 1,000 步即可”**。[报告 §2.1.2][R1]

第一阶段：在已经形成语言/视觉表示的主干上，冻结主干和视觉塔，使用 dense attention 得到 teacher 分布，只更新 indexer：

```text
token_teacher = L1_normalize(sum_attention_heads(P_dense))
block_teacher = L1_normalize(max_pool_complete_blocks(token_teacher, r=4))
L_index = KL(stopgrad(block_teacher) || softmax(index_scores))
```

原始 index key 先按完整 block 做平均池化，再 norm/RoPE；pooling 的顺序不能调换。源码分数含 `1/sqrt(index_head_dim)`，报告对应公式未显式列出该项，mini 已选择遵循固定源码，应把此差别保留在对应测试和说明中。

第二阶段：启用稀疏 attention，LM loss 更新主干；在 selected blocks 上重新归一化 teacher/indexer 分布，KL 训练 indexer。teacher probability 和 indexer 的 backbone input 都 detach，不能把 KL 无意传回主干去“迎合检索器”。离散 top-k 不假装可微，梯度来自 KL。[报告][R1] [当前 QSA 实现的源码依据][R3]

保持当前 eager 实现为 golden reference，新增向量化/分块实现时，对随机序列、全 padding 尾部、不足 4 token 的尾块、文档边界、混合长短 batch、视觉跨度和 cache 做同权重对照。不能因为 eager 很慢就在完整 dense `L×L` mask 上做一次布尔筛选后宣称获得稀疏显存优势。

### 4.3 视觉与 QSA 的耦合

视觉 feature 是语言上下文中的有效位置，会参与稀疏检索，但它们自身没有 LM CE 标签。indexer query 的训练 mask 与 CE mask 应区别处理：有效视觉/prompt query 不应因 label=-100 被无条件丢掉。需要分别统计 text→image、text→text 和 image-position 的选择覆盖。

先遵循原生 block 选择和尾块处理规则，不能无依据强制“所有图像永远全可见”。针对 OCR/多图导致的信息漏检，比较 budget `{256,512,1024}`，而不是让模型暗中绕过 QSA。若需要图像保留策略作为新研究变体，另命名配置并做相同预算对照。

保持 causal 语义：尚未到达的图像或后续文字不可被 query 读取；图像内部视觉塔的双向编码不能变成语言序列跨未来答案的双向 attention。跨 packed 文档不能组成一个压缩 block，segment reset 还必须清理 GDN/PLE 卷积与 n-gram 历史。

## 5. 优化器与数值稳定性

当前 `/workspace/MiniFrontier/minifrontier/training/miniqwen4_optim.py` 已有 Qwen 专属语义切分和 Polar Express 实现，应保留并完善视觉/MTP 分组，而不是另写一个通用 Muon 覆盖它。官方采用 8 次 Polar Express、0.95 Nesterov、按矩阵形状缩放；输入/输出 embedding、router、Gated Residual 低秩等走 AdamW，n-gram table 不做 weight decay。Q/GDN 头、expert gate/up 的 fused 参数按语义拆开。[报告 §3.1][R1]

扩展要求：

1. 输出 gate、GDN decay/beta 等不属于普通 head 矩阵；一个 fused 参数可含 Muon 与 AdamW 子块，必须保存各自状态。
2. 视觉 Conv3D、位置表、norm 和 merger 的分组是本地新增选择：初版视觉塔与 merger 用 AdamW，稳定后再做 per-head Muon 的独立视觉对照；不得把报告对文本模块的结果推广为已经验证过视觉模块。
3. PLE table 使用无 decay 的 Adam；PLE key/value linear 可以用 Muon。报告学习策略与 mini 表容量要分开。
4. 比较一个 optimizer update 的参考结果、恢复状态后的下一步、DDP/累积一致性；新增参数必须被恰好一个分组覆盖。

建议用 20M–50M token pilot 搜索主干 Muon LR `{0.003,0.01}` 与 AdamW LR `{1e-4,3e-4}`，保持当前缩放实现；先单因素筛选再组合，避免一次跑完整大网格。视觉 AdamW 起点 1e-4，projector 3e-4，分别比较倍率。正式 LR 由等预算验证与稳定性选出。

固定有效 batch `{16K,32K}` 输入位置比较，不默认加 batch warmup；Qwen 报告对该设计支持恒定 batch，但仍需 LR warmup。建议 LR warmup 1%–2% 计划 token，再按选定 schedule 冷却。更换 sequence length 时维持近似有效 batch 和实际 CE 归一化，不意外将更新强度扩大四倍。

BF16 主计算，FP32 loss、router、norm 敏感统计与 optimizer state；grad clip 初值 1.0，记录触发率和每参数组 update RMS。遇到 spike 先定位 GDN、QSA、GR、PLE 和视觉梯度，不默认认为所有架构都应引入 Kimi QK-Clip。

## 6. 数据：建立多语言、多模态与长序列池

### 6.1 数据源与来源控制

| 数据源 | 主要用途 | 准入与限制 |
|---|---|---|
| [Fineweb-Edu-Chinese-V2.1][D1] | 中文 PT | 高分目录与实际 score 字段先核对；按来源随机取样，删除模板网页 |
| [FineWeb-Edu][D2] | 英文 PT | 固定 dump/子集；网页与长文同时采样，保留独立验证文档 |
| [SmolLM-Corpus][D3] | 合成教育材料、python-edu | fineweb 派生部分不与 D2 重复计算；代码按 repo 切分并保留许可 |
| [Infinity-Instruct][D4] | 中文与双语 SFT、代码/数学 | 当前自动 gated；源任务与首问模板聚类，未获得访问时不能在脚本中默认成功 |
| [UltraChat 200K][D5] | 英文日常对话 | 使用 train_sft；test_sft 封存，不混入主训练 |
| [SmolTalk][D6] | 约束、摘要、改写、工具 schema | 按子集保留来源/许可；人工审阅合成模板，防止少数风格垄断 |
| [ALLaVA-4V][D7] | 双语 caption/VQA | NC 来源隔离；caption/instruct 同图绑定 split；不重复计为独立视觉信息 |
| [FineVision][D8] | OCR、图表、文档、科学图 | 按子数据集做许可/质量/长度 allowlist，剔除与其他池重叠样本 |
| [Docmatix][D9] | 文档与多页 QA | 按 PDF 而非页随机切分；保留答案页映射，避免无证据问答 |
| [LLaVA-Video-178K][D10] | 短视频语义/时序 | 帧抽样与问题事件保持一致，按原视频 group 切分 |
| [OpenThoughts-114k][D11] | 推理课程 | 只选完整答案且 verifier 合格的长度桶；不直接截断长 CoT |
| [UltraFeedback binarized][D12] | 可选 DPO | chosen/rejected 在相同 prompt 下，去除长度/格式主导的伪偏好 |

这些数据替代未公开的官方完整语料，不表示来源于 Qwen 官方训练集。MiniMind 的加载/训练工程可作教学参考；其短训练的效果和视觉塔选择不能直接迁移为 431M、64K 词表、PLE、GDN/QSA 的预算依据。[MiniMind][R4]

### 6.2 混合、清洗与固定切分

文本预训练流按 token：中文自然教育 40%、英文教育/自然 25%、代码 20%、数学/科学 10%、对话式文本 5%。每桶记录来源唯一字节、token 数与重复轮次；后期增加代码/OCR 不能只记录为“高质量”。

多模态流按样本：caption 35%、OCR/文档 25%、图表/数值/空间 20%、图文交错/多图 15%、视频 5%。这是独立于文本 token 比例的分母。每次 update 记录 input token、CE token、image features、image count、frame count，防止大量无监督图像位置被误记为语言训练量。

正式主预算建议 **3.0B CE token、2.5M 图像出现次数、50K 视频样本**；caption/答案 token 已包含在 3B 内。图像唯一量目标 ≥1M；同源图片不同语言描述不能计为两张唯一图。与旧 3.6768M LM token 相比，这是约 816 倍的暴露量，但仍需要能力曲线来判断足够与否。

统一数据 schema 应有 `source/revision/item_id/group_id/license/lang/task/content_hash/quality/turns/media`。清洗先于 split：文本精确与近似去重、首问聚类、答复模板检测；图像 decoded RGB hash+pHash、视频 id、PDF id 做跨源 group。训练/验证/测试按 group 切分；跨语言同图、同一文档的不同问题不得泄漏。

建议 SFT 同一首问最多保留 3 个真正不同的优质版本，身份/意识类助手监督 token ≤0.5%；预训练删除大段 boilerplate，但不删除有价值的短文。中文先抽 300 条、代码 100 条、视觉每子类 100 条人工审阅，记录拒绝原因再决定是否扩大。

保留各域约 0.5% 文档作为验证池，固定约 5M CE token 做周期 PT 验证；SFT 固定 2K 独立问题组，多模态固定 1K 图像/文档组。最终任务测试另建封存集合，排除训练集和公开混合集中的已知同题。

### 6.3 Packing 与模板

PT 用真实长文和代码文件组构造 512/1K/2K/4K 桶，不以 padding 把短数据伪装成长数据。跨文档 packing 同时隔离 causal attention、GDN 状态、conv 状态、PLE n-gram 历史和 QSA 压缩组；尚未实现完整 segment reset 时，先每序列单文档，不能只插 EOS 就声称隔离。

图像/视频必须整段放入；长度预算是文字+视觉合并 token+角色/时间标记总数。过长多图样本改进 resize、转长桶或拒收，不能随机删图却保留依赖被删图的答案。SFT 保留完整 assistant answer；user/system/tool observation、padding 和视觉 token 无 CE，结束 token 有 CE。

tokenizer、chat template、视觉 processor 与 demo 同源。专门验收从 raw multimodal message 到 `input_ids`、`ple_input_ids`、`position_ids`、feature 数、label 数的对应；输出一份可读的样本审计，不只检查张量 shape。

## 7. 完整预训练阶段与 QSA 转换预算

Qwen 报告未公开足以证明唯一正确的视觉训练顺序。本方案选择早期即加入原生 mini 视觉塔的联合 PT，使 QSA 转换时已经覆盖图像；这是项目的教学训练设计，区别于对官方视觉训练史的事实陈述。

| 阶段 | CE 预算 | 图像/视频暴露 | 长度与注意力 | 训练对象与退出条件 |
|---|---:|---:|---|---|
| Q0 单元与记忆诊断 | 0.5–2M，单列 | 64–512 张 | 64/256/512，dense | 对齐源码、图像反事实、PLE/mRoPE、MTP 因果性 |
| Q1 dense 联合 base | 300M | 300K 图；0 视频 | 512→1024，GDN+全局 dense | 训练文本/视觉/projector/MTP；indexer 可初始化但不计已学成 |
| Q2 dense 主预训练 | 1.20B | 1.20M 图；20K 视频 | 1024→2048 | 先得到可用的 dense teacher attention，检查各域验证 |
| Q3 indexer dense distill | **20M–60M 输入位置，另列** | 图文任务均覆盖 | 2048；部分 4096 | 冻结主干/视觉/MTP，仅 indexer；KL、选择质量收敛才转换 |
| Q4 sparse 联合 CPT | 1.00B | 800K 图；20K 视频 | 2048→4096，budget 512 | LM 训主干，selected-block KL 训 indexer；检查稀疏化能力回退 |
| Q5 高质量冷却 | 500M | 200K 图；10K 视频 | 1K/2K/4K 混合，sparse | 各域与长文平衡，保存 base 与视觉评估 |
| 主干合计 | **3.0B CE** | **2.5M 图；50K 视频** | Q3 不计主干 LM 学习量 | 不含后训练 |

Q3 用输入位置计量，因为主干冻结且目标不是 LM CE；它与各阶段“图像暴露总量”会有额外重放，应单列重放计数，不假装是额外新图。20M 不是硬性成功标准，KL/检索覆盖未达标可到 60M 或回查 teacher/data；训练随机 teacher 的 indexer 不能替代基础 PT。

建议 Q3 indexer AdamW LR 先比较 `{3e-4,1e-3}`，warmup 2%，grad clip 1.0；其他参数要求 grad=None 且逐步哈希不变。Q4 主干 LR 为 Q2 稳定峰值的 0.3–0.5 倍短 warmup 重启，indexer LR 独立从 1e-4 起测，KL 系数初值 1，记录 LM/KL 的有效 query 分母。不能把 raw KL 大小和 CE 大小未经归一化直接相加。

```text
Q1/Q2: L = L_CE + lambda_mtp * L_MTP + lambda_router * L_router_source
Q3:    L = L_index_dense
Q4/Q5: L = L_CE + lambda_mtp * L_MTP
           + lambda_router * L_router_source + lambda_index * L_index_selected
```

当前本地模型和固定上游配置的 `router_aux_loss_coef` 默认均为 **0.001**，建议首轮保留并显式写入run配置；它是可配置默认值，不等于报告证明在本mini尺度最优。记录unweighted aux、weighted aux、有效token和专家负载，再比较0/0.001的短消融。禁止在Q3顺便更新router、PLE或视觉塔。

### 7.1 QSA 转换的验收

在相同 checkpoint 上对 dense 与 sparse forward 做对比，冻结 1K/2K/4K 的中文、代码、OCR、多图、检索集合。记录 index KL、teacher 在 selected blocks 上的质量、dense→sparse NLL 增量、真实延迟/显存、每模态任务分数。建议初始允许总体 NLL 增量 ≤0.1 nats 作为调查线，任一重点模态退化 >5 个百分点必须分析；这些是本地门槛，不是理论保证。

为了定位问题，保留 budget 256/512/1024 的等权重推理与短 CPT 对照，并对不足 block、tail、不等长 padding、跨图边界做定向测试。若 512 不足以保留图表/多图证据，应调整预算和训练数据，不能直接把预算回满后仍宣称验证了稀疏模型。

## 8. MTP、SFT 与完整后训练

### 8.1 MTP 的补齐边界

发布config含一个next-token prediction block的信息，但当前Transformers模型代码会忽略部分`mtp.*`权重，不能据此认为存在完整训练路径。本次进一步检查了vLLM项目维护的Qwen4Exp MTP实现，已能确定不能把它照写成普通`Linear(2H,H)`融合。[配置][R2] [Transformers][R3] [MTP推理源码][R7]

新增`miniqwen4/mtp.py`应保留以下计算关系：主干输出的是final mixer之前的4流hidden；将`4×512=2048`维flatten hidden做对应GemmaRMSNorm，再按每条512维流用共享`fc_hidden:512→512`投影；新token embedding做自己的norm和`fc_embedding:512→512`，经residual-linear-shared路径注入每条流。预测block使用全局/QSA attention和MoE、GR，关闭PLE。输出同时保存混合后的512维sample hidden供LM head使用，以及未压成单流的2048维hidden供下一draft步使用。不能先把主干压成一条流，再repeat四遍当等价实现。[MTP推理源码][R7]

embed权重按主词表映射；输出头按`mtp.shared_head.head`的发布键单列核对，不能仅凭名字shared就认定它与主LM head绑定。保存参数共享/复制关系、norm参数约定和权重键映射。draft内部没有第二个视觉塔，视觉条件通过目标模型的多流hidden/KV传入；图像生成范围和目标模型的position/cache语义仍需验证。

初版训练一个额外预测深度，使用允许看到的shifted token预测再下一token；建议lambda`{0,0.1,0.2}`做小预算对照。EOS、图片、packed document与padding附近的未来标签要mask，不能泄漏被预测token。上述推理布局有公开代码依据，而mini MTP损失权重和训练课程属于本地设计。

后续若做 speculative 推理，报告中 QSA 的索引复用跨多个预测步与“有一个 MTP block”是不同层面。可以在递归 draft 的多个步上复用已核定的 QSA indices，但必须保持 causal 可见集合、新 token/尾块更新与 rollback；不要把 `next_n=4` 解读为需要随意增加三个独立主干层。[报告 QSA 推理部分][R1]

最终SFT/RL/偏好权重选定后，另以10M–30M有效draft位置做适应训练：冻结目标，生成覆盖文字/图像prompt的目标回答，缓存对应多流hidden，用相同结构训练MTP的未来token CE；可比较加入完整词表teacher KL的变体，但不能把Kimi的LK loss或DeepSeek DSpark目标默认为Qwen原配方。短unroll逐步模拟自身draft输入，验收逐步acceptance和缓存回滚。先测无spec、1步和3步draft的端到端延迟，实际无收益时保留普通增量作为默认demo。

### 8.2 通用与多模态 SFT

建议清洗后池为 400K–800K 文本指令、400K–800K 视觉指令；先 1 epoch，必要时最多 2 epochs，目标 **120M–250M assistant token**。若唯一数据和 token 不匹配，以统计报告为准，不为了凑预算重复少数问题十几轮。

按 assistant token：日常解释/问答 20%、摘要改写翻译 15%、代码 15%、数学 10%、OCR/文档 15%、图表/VQA/多图 15%、工具与结构化约束 5%、视频 5%。优先完整短答案与多轮指令，再加可验证长推理。中文和英文按约 60/40 作为起点，最终以目标 demo 任务分布调整。

SFT 长度 512/1K/2K/4K；保留基础 PT 文本的 5%–10% replay 作为可测的抗遗忘选项，该比例按 CE token 单独记录，不与 SFT assistant 比例混淆。视觉塔继续训练，初始 LR 为主干 AdamW 组的 0.2–0.5 倍，projector 1–2 倍；主干 SFT AdamW 组 LR `{2e-5,5e-5}`，Muon 组从 PT 最优 LR 按相同倍率缩小。

保留三类 mode：直接回答、有限推理、工具调用。官方公开后训练细节不足，不强行宣称必须采用 Kimi 的九教师/effort 划分。若本项目提供 think/no-think，先定义完整模板和训练样本分布，分别验收；不要仅靠插一个 `<think>` 让模型获得推理能力。

### 8.3 可验证 RL

通过 SFT 基础门槛后，建议 30K–100K 独立训练 prompt，group 4 起步、最多 8；以 **10M–30M rollout response token** 作为首轮 RL 预算。覆盖算术/可验证推理、小函数、JSON/工具约束、OCR 数值与图表计算。困难题先做高质量 SFT 或难度筛选，不能在几乎全失败题组上期待 GRPO 自动创造语言能力。

GRPO 使用同 prompt 同 mode 的组内相对奖励与 clipped policy 更新；属于本方案选择的后训练算法，不是报告明确公布的 Qwen 全配方。[GRPO][R5] 初始 LR 1e-6–5e-6、clip 0.2、reference KL 0.01，零方差组跳过并报告，单批 rollout 最多 1–2 次更新。

当前通用 rollout 存在采样分布与 old_logp 重算分布不一致的问题：先统一 temperature、top-p 和特殊 token mask。建议初版 temperature 1、无 top-p 截断、合法 token mask 固定，并保存实际行为 logprob；训练重算必须与其同定义。rollout、reference、old policy 不更新 router 平衡状态，不启用训练 dropout；工具 observation 不计 action loss。

每任务存输入、图像/视频 id、策略 hash、mode、轨迹、结束原因、奖励拆解及 verifier 版本。代码在隔离环境执行，结构化输出先严格解析；图表答案需单位与容差归一化，不能只 regex 提取第一个数字。

### 8.4 偏好与蒸馏分支

DPO 保留为可选实验，使用 [UltraFeedback binarized][D12] 或本模型同 prompt 候选的合格偏好对，先 10K–30K 对，beta `{0.03,0.1}`、LR `{5e-7,2e-6}`。reference 固定为进入阶段的 SFT/RL 模型；chosen/rejected 共享 prompt 和图像，验证长度偏差、答案截断与偏好标签来源。[DPO][R6]

SFT、RL、DPO 每一阶段都在同一 heldout 上比较 NLL、回答完整性、换行/EOS 比例、重复度和任务成功率。DPO preference accuracy 提升不足以抵消对话能力退化，本次旧 DPO 已是直接反例。

若另做多教师 OPD，必须明确这是项目扩展：学生 on-policy 轨迹上按任务选择教师；同 tokenizer 教师可做 full-vocab KL，不同 tokenizer 的旗舰只能用离线答案蒸馏或专门的跨词表方法。不要把当前 Kimi 式 token reward 自动冠名为“Qwen 官方后训练”。首轮建议 10M–30M response token，且只在领域教师确有优势时启用。

### 8.5 量化与部署适应

本次报告和已检查的源码不足以把一套 Kimi/DeepSeek MXFP4-QAT 配方称为 Qwen 官方训练要求。项目仍需部署量化研究，但明确归为本地扩展：先 BF16 验收，再量化校准，若 PTQ 在目标任务退化，再做 5M–20M assistant token 的仿真 QAT。

对 routed experts、PLE table、GDN 状态、router 和 norm 分别评估敏感性，不一次把全部 4bit。3090 不具备这些新格式的原生高效算力时采用量化/反量化仿真，不能承诺训练加速。使用相同格式做 rollout 和训练对照，防止训练 BF16、推理近似量化导致结果不可复现。

## 9. 3090 上的执行与性能改造

当前 Qwen 短文本曾测得 batch 1、长度 512 完整训练更新约 **5.71GiB**；该数字不能证明新增视觉/MTP、4K QSA、RL reference 同驻留一定可行。旧双卡 sparse CPT 每次 update 约 96.84 秒处理 8,184 LM token，折合约 84.5 token/s，只是旧 eager 实现与该双卡场景的观测，不是单卡新方案速度预测。

431.6M 参数在 FP32 参数/梯度/Adam 两状态的粗略占用为 **6.43GiB**；实际混合 Muon/Adam 状态不同，还需加激活、临时 logits、attention/indexer teacher、视觉和副本。4K×64K 的 FP32 logits 单份约 1GiB。QSA dense warmup 为 teacher 概率保留 `L×L` 可能主导显存；不能同时 materialize 每层完整 logits、teacher probabilities 和多份损失中间量。

工程优先级：

1. LM head CE 分块/fused，MTP 分深度分块，不生成多份全词表 FP32 logits。
2. GDN 用可核对的 fused/chunk kernel；QSA indexer 从逐 query Python 循环改向量化/分块，保留 golden reference。
3. dense teacher attention 分块读取和在线统计 teacher target，避免永久保留全部层 attention；只对参与 KL 的层生成目标。
4. 激活 checkpointing 保持无副作用；PLE/GR/QSA 的输出和梯度对照通过才替换内核。
5. text microbatch 1–2，图文 1，按视觉 token 与总长度分桶；gradient accumulation 达 16K/32K 输入位置。统计真实 CE，不用 padding 填大吞吐。
6. RL/OPD 顺序驻留当前学生、reference/教师，预计算可复用 reference 分数；教师缓存绑定版本，不能跨模型改动复用。

每种阶段、长度和模态配比测 50–100 次 warmup 后的 200 个真实 updates，记录 CE token/s、图像/s、峰值 allocated/reserved、优化器时间、GPU utilization 与 I/O。单卡留约 2GiB 余量，失败不得通过跳 batch 偷偷完成预算。

墙钟按 `D / throughput / 86400` 估算：假设 500/1000/2000 CE token/s，每 1B 约 23.15/11.57/5.79 天，3B 约 69.4/34.7/17.4 天；**是假设情景，不是当前实测保证**。旧 eager CPT 若不优化，可能更慢；先优化并测量，再决定追加量。RL、dense indexer、视觉解码和评估另算，不能用一个 PT token/s 覆盖全部流程。

训练器按 accumulation window 的 CE sum/有效目标数归一化；DDP 考虑 gradient averaging。保存 optimizer 子组状态、sampler cursor、token counters、RNG、scheduler、QSA stage/budget、processor 与 tokenizer hash；indexer 转换阶段和恢复点要能复现实验。

## 10. 完整验收与发布范围

| 阶段 | 必须验证的内容 | 本方案建议判据 |
|---|---|---|
| 结构 | GDN recurrent/full，GR/PLE 与固定源码；QSA 全量/稀疏/缓存，视觉 merger/mRoPE | tiny FP32 数值与梯度参考；BF16 另设误差带；随机未来扰动不影响过去 |
| 数据 | role/labels、image feature、PLE ids、grid、三轴位置、分段 reset | 200 条可读样本审计全部通过；无 silent image dropping |
| PT | 分语言/领域 token NLL、完整续写、router 和 PLE 使用统计 | 随预算稳定改善；100–200M 仍严重乱码先诊断，不进入偏好训练 |
| QSA | dense→sparse 能力与 NLL、selected teacher mass、实际稀疏比例 | 重点模态不能被总均值掩盖；预算增大是否能解释损失 |
| SFT | 固定 500 独立文本任务和 1K 视觉组；人工/规则联合 | 基础指令完成率起始门槛 80%，严重重复/乱码 ≤5%；这是本地 release gate |
| 视觉 | OCR CER、文档 ANLS、图表数值、VQA、位置/帧序任务 | 优于无图与图像 shuffle 对照，图像相互冲突时能依据对应图回答 |
| RL/DPO | 成对生成、任务成功、参考 KL、EOS/换行/长度分布 | 不因 reward 或偏好单指标改善而接受整体退化 |
| 推理 | cache reset、batch owner、失败 forward 后重用、视觉 position delta、MTP rollback | 全量与增量对照、并发隔离和多轮图像更新通过 |

长上下文用真实检索、代码跨段依赖、图片前后引用和多图定位验证，不能只通过一条 needle-in-a-haystack 就宣称全功能 4K。若只测到 4K，则文档与 demo 上限写 4K；8K/更长需额外训练与性能记录。

demo 导出主权重、可选 MTP、config、词表、PLE hash 参数、processor、能力报告与源码/数据 hash。默认选择已验收 checkpoint，支持文字/图像/短视频与已训练 mode；输入超限和图片缺失要明确反馈，不无声降成纯文本回答。

## 11. 待实施清单

下表“新增”为建议路径，尚不是可调用的现成命令。

| 优先级 | 文件/模块 | 修改方案与完成证据 |
|---|---|---|
| P0 | `/workspace/MiniFrontier/minifrontier/data.py` | 首问/跨源图像 group、固定 split、完整回复、真实 token 预算、shard manifest |
| P0 | `/workspace/MiniFrontier/minifrontier/models/miniqwen4/modeling.py` | 原 input_ids→PLE、视觉 embeddings、三维 positions 的完整接口 |
| P0 | 新增 `minifrontier/models/miniqwen4/vision.py`、`processing.py` | 原生 Conv3D ViT/merger/processor，按固定源码提取并缩配 |
| P0 | `/workspace/MiniFrontier/minifrontier/models/miniqwen4/qsa.py` | 保留 golden；扩展视觉/segment mask，分块计算 teacher KL 与性能实现 |
| P0 | `/workspace/MiniFrontier/minifrontier/training/miniqwen4_optim.py` | 视觉/MTP 参数归属，分组覆盖与 resume；已有文本语义分组继续使用 |
| P1 | 新增 `minifrontier/models/miniqwen4/mtp.py` | 共享关系、shifted labels、因果 MTP；与主干 λ=0 对照 |
| P1 | `/workspace/MiniFrontier/minifrontier/training/train.py` | token 驱动 stages、Q3 冻结证明、Q4 selected KL、进入门槛、累计归一化 |
| P1 | `/workspace/MiniFrontier/minifrontier/training/rollouts.py` | 实际行为 logprob、mode/图像/工具 mask、验证奖励与可恢复轨迹 |
| P1 | `/workspace/MiniFrontier/minifrontier/models/miniqwen4/cache.py` | 在已有文本 cache 上补视觉位置、尾块状态、snapshot/rollback |
| P2 | `/workspace/MiniFrontier/minifrontier/inference.py` | 原生多模态 demo、MTP accept/reject、量化格式与能力范围 |

执行顺序：固定源码/词表 → 数据与视觉接口 → dense 基础能力 → indexer distill → sparse CPT → SFT → 可验证 RL → 可选偏好/蒸馏与量化 → 增量/MTP demo。不能把 QSA indexer warmup 当从零预训练的替代，也不能把 Qwen、Kimi、DeepSeek 的后训练算法混成一个通用阶段名。

## 12. 数据固定版本与资料

| 数据集 | revision | 卡片许可入口 |
|---|---|---|
| Fineweb-Edu-Chinese-V2.1 | `a5b574efa48beb3a8f6887ef0b093becf004328b` | Apache-2.0 |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-BY |
| SmolLM-Corpus | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-BY，代码追溯原许可 |
| Infinity-Instruct | `bddc39a8feadbd679c30623197f4e736b7e75b48` | CC-BY-SA-4.0，自动 gated |
| UltraChat 200K | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT |
| SmolTalk | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | 顶层无统一标识，逐子集 |
| ALLaVA-4V | `0fd42fce5c047d387a4bb5318d588eae9a9797f0` | CC-BY-NC-4.0 |
| FineVision | `3c380a731a3429c1d04693d6ec16d7e683def84c` | 逐子源许可 |
| Docmatix | `0725b65616e0e5f6024be10e38ddf8d8c48664fd` | MIT 卡片，文档来源单列 |
| LLaVA-Video-178K | `6d8c562dc26d70042a0d9704d1cae58c94b89098` | 逐视频来源 |
| OpenThoughts-114k | `bd093c3994fd54d2390985b66988ddf282a55eb6` | Apache-2.0 |
| UltraFeedback binarized | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT |

[R1]: https://arxiv.org/html/2608.30320v1
[R2]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540
[R3]: https://github.com/huggingface/transformers/blob/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
[R4]: https://github.com/jingyaogong/minimind
[R5]: https://arxiv.org/abs/2402.03300
[R6]: https://arxiv.org/abs/2305.18290
[R7]: https://github.com/vllm-project/vllm/blob/96eccb8f49aff58aa2a11b431bf8331f4b368606/vllm/models/qwen4_exp/nvidia/mtp.py
[D1]: https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/tree/a5b574efa48beb3a8f6887ef0b093becf004328b
[D2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
[D3]: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f
[D4]: https://huggingface.co/datasets/BAAI/Infinity-Instruct/tree/bddc39a8feadbd679c30623197f4e736b7e75b48
[D5]: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a
[D6]: https://huggingface.co/datasets/HuggingFaceTB/smoltalk/tree/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa
[D7]: https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/tree/0fd42fce5c047d387a4bb5318d588eae9a9797f0
[D8]: https://huggingface.co/datasets/HuggingFaceM4/FineVision/tree/3c380a731a3429c1d04693d6ec16d7e683def84c
[D9]: https://huggingface.co/datasets/HuggingFaceM4/Docmatix/tree/0725b65616e0e5f6024be10e38ddf8d8c48664fd
[D10]: https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/6d8c562dc26d70042a0d9704d1cae58c94b89098
[D11]: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/tree/bd093c3994fd54d2390985b66988ddf282a55eb6
[D12]: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/tree/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85
