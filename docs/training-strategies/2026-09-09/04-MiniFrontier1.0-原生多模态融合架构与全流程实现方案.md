# MiniFrontier1.0：原生多模态融合架构与全流程实现方案

名称：MiniFrontier1.0；状态：实现方案（尚未实现）；日期：2026-09-09；硬件环境：仅有 PCIe 互联的多卡环境。

**建议实施对象：约 228M 总参数、16 层、512 隐藏维度的原生图像/短视频语言模型。采用 KDA 循环状态、CSA 压缩历史和 QSA 索引的逐 token MLA 三种通路；配合四流 Gated Residual、LatentMoE 和受控的浅层 n-gram 查表。主预训练按 3B 有效 CE token 立项，后续完整覆盖多模态 SFT、可验证 RL、同词表多教师 OPD、MTP/draft、量化和 Demo。**

这是针对 MiniFrontier 当前量级提出的新模型设计，不是任何一家旗舰模型的缩放复刻，也不是已经验证优于现有三个模型的结论。本文中的模型尺寸、预算、配比、阈值和阶段安排，除明确说明来自上游者，均为本项目的候选实现决策。最终配置必须经过本文规定的正确性检查、短程对照和单卡实测后冻结。

**硬件原则：单个可容纳的训练任务优先单卡，多张 PCIe GPU 优先用于独立任务和低通信协作。** 本方案要求完整流程保留单卡执行路径，也允许独立实验、不同教师、评估和采样分配到不同 GPU；不要求整个项目始终只使用同一张卡。需要频繁传递梯度、激活、专家 token 或全词表 logits 的多卡方案，先测端到端收益再启用。具体安排见第 9.5 节。

## 0. 这次设计解决什么问题

用户分享的[架构讨论](https://chatgpt.com/share/6aa11f53-d4ac-83ee-bb8d-b05d414f394d)已经读取。最值得延续的是三种历史信息访问方式的分工，以及避免同时堆叠多种残差机制的判断。聊天里的大模型预算、专家规模、检索数量不能直接移植到几亿参数模型。

MiniFrontier1.0 的目标有三个层次：

1. **教学和复现价值**：能清楚观察各模块的计算、梯度、缓存、训练阶段和推理行为。
2. **可用的小模型**：中文和英文能连贯作答；能对简单图片、文档局部和短视频进行有依据的回答；能执行有限的工具与可验证任务。
3. **可检验的融合研究**：在相同数据、参数或计算预算下，回答三种通路是否互补，以及这种互补是否值得增加工程复杂度。

这里的“多模态”具体指：输入支持文本、单图、多图和抽帧短视频，输出文本和结构化工具调用。语音输入、音频理解、图像生成和视频生成不属于本版本的交付能力。视频必须经过真实视频任务训练和时间顺序评估，不能仅因为处理器能接收多帧就宣称具备视频理解。

### 0.1 证据与现有工程的关系

本方案参考了本地 2026-09-08 的三份训练策略，以及 2026-09-09 的代码只读副本。现有策略配置的参数量级分别约为 MiniKimi 205M、MiniDeepSeek 244M、MiniQwen 513M；这些数字用于确定设计尺寸，不代表同等效果或同等速度。

只读副本中可以定位到 KDA、LatentMoE、QSA、GR、CSA、视觉处理、MTP、draft、优化器和后训练相关实现。这说明存在可复用的构件；不能据此认为它们已经能在新组合中直接工作。本文没有重新运行现有训练、没有更改远端工程，也没有把旧模型的测试结果当成 MiniFrontier1.0 的验证结果。

代码副本来源标识：`3e9cf58`，检查日期 2026-09-09。正式实施时应重新记录实际基线 commit、工作区补丁和依赖版本。此前方案位于本地相邻的 `2026-09-08` 目录；本文件是独立的第四份方案。

### 0.2 最重要的判断

**应该融合信息访问能力，但不能假定旗舰模块越多，小模型越强。** 当前量级的主要风险，是数据不足、专家训练不充分、过多压缩造成信息损失，以及稀疏算子的固定开销超过节省的计算量。

SmolVLM-256M 的公开实现说明这个量级可以做有实际用途的视觉语言模型，但它使用已有语言和视觉底座；它不能证明一个新组合、全随机初始化的模型仅需相同后训练成本。MiniFrontier1.0 的主路线明确包含视觉编码器从随机初始化开始的训练成本。[SmolVLM-256M 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

## 1. 从哪些模型取什么，哪些不放入默认结构

| 来源 | 有依据的设计启发 | MiniFrontier1.0 的实施选择 | 必须重新验证的部分 |
|---|---|---|---|
| Kimi K3 | KDA、稳定的潜空间专家、原生视觉、MTP 与训练稳定性处理 | KDA 占 12/16 层；LatentMoE；视觉从预训练开始参与；一个 MTP 块 | 与 GR、CSA、mRoPE 的交互；小规模专家覆盖 |
| Qwen3.8-Flash-Next / `qwen4_exp` | 微块索引、GR、浅层查表和多模态处理 | 四流 GR；QSA 风格索引；小预算 n-gram；图像/视频位置元数据 | 将索引接到 MLA 上是新适配，不能当成官方 QSA 原样复现 |
| DeepSeek-V4 | CSA 压缩历史、局部细节通路及高效后训练思路 | 两层 CSA，压缩率 4；全词表 reverse-KL OPD 路线 | CSA 与其前后 KDA 层的协作；适合本模型的稀疏预算 |
| Engram | 条件查表补充参数记忆 | 仅一个浅层、约 8.8M 参数的可关闭模块 | 是否值得占用本已很小的参数预算 |
| MiniMind / MiniMind-V | 易读训练链路、可运行小模型和演示组织 | 借鉴数据入口、阶段脚本和 Demo 的可理解性 | 不把教学短训预算当成本模型完整训练预算 |

机制出处见 [Kimi K3 报告](https://arxiv.org/html/2607.24653v1)、[Qwen 架构报告](https://arxiv.org/html/2608.30320v1)、[DeepSeek-V4 报告](https://arxiv.org/html/2606.19348v1)、[Engram 论文](https://arxiv.org/abs/2601.07372)、[MiniMind-V 项目](https://github.com/jingyaogong/minimind-v)。上表的具体组合与取舍属于本方案。

以下机制列入研究清单，但不在默认结构同时启用：

- **AttnRes、mHC**：作为 GR 的替换对照。三者承担的深度信息流作用有重叠，同时叠加会破坏归因并增加状态管理。
- **HCA-128**：在 2K～8K 训练长度下过于粗糙，尤其可能损害 OCR 和精确查找。只有 8K 验证完成后才研究它是否值得加入。
- **Loop / 动态深度**：改变训练和解码控制流，且本模型只有 16 层，优先级低于把基本能力训练充分。
- **超大专家数、CPU 查表卸载、跨 GPU 专家并行**：不作为当前小模型的默认实现。专家尽量留在一次训练使用的同一张卡上，避免经 PCIe 频繁交换 token；独立模型和教师训练可以使用其他 GPU。
- **DSpark 与另一套并行 draft 同时上线**：先交付一种正确且可测的 MTP 派生 draft，再做独立替换实验。
- **混合多种视觉编码器**：本版只有一套原生视觉塔，避免重复参数和不一致的视觉 token 语义。

完整方案不意味着所有论文中的模块全部开启。完整意味着每个选用模块的训练、推理和验收都有闭环，每个舍弃机制都有明确原因和后续实验入口。

## 2. 建议冻结的模型规模

### 2.1 主配置

| 项目 | `mf1_228m_native` 候选值 |
|---|---|
| 隐藏维度 / 主干深度 | 512 / 16 |
| 词表 | 32,768，包含特殊 token；重新训练并冻结 |
| 输入 embedding / 输出 head | 不共享；MTP 共享主模型输出 head |
| 残差 | 4 流 GR；每流 512；读门低秩维度 64 |
| 主干注意力 | 12 KDA + 2 CSA-4 + 2 QSA-MLA |
| KDA | 8 头；key/value 维度 64；短卷积核 4；完整输出门 |
| QSA-MLA | 8 头；内容维 64，位置维 32，value 维 64；KV latent 128；query latent 128 |
| CSA | 8 个 query 头；共享 KV 维 64；位置维 16；query latent 128；压缩率 4 |
| 索引器 | 4 query 头，head dim 32；微块 4；独立参数 |
| 专家 | 每层 32 个 routed，top-4；latent 256；专家中间维 256 |
| shared expert | 每层一条全宽 512 通路，中间维 768 |
| 浅层查表 | 第 2 层一次；2/3-gram，各 2 个 hash 头；每表 32,768×64 |
| 视觉编码器 | ViT 12 层，宽 384，6 头，MLP 1536 |
| 视觉 patch / 时间 patch / 空间合并 | 16 / 2 / 2×2 |
| 视觉投影 | 拼接后的 1536 → 512 → 512，两层 MLP |
| MTP | 1 块，GR + dense MLA + 同规格 LatentMoE；推理可卸载 |
| 最终训练上下文 | 8,192；先训练短序列，再逐步提升 |
| 主精度 | BF16 前向；关键归约、状态和优化器按模块使用 FP32 |

这里的 32K 不是把原模型的 150K/160K 词表截断。新 tokenizer 与新 checkpoint 是独立体系。若 tokenizer 试验显示 32K 的中文/OCR/代码切分明显恶化，应在正式训练前改为 64K；不共享 embedding/head 时增加约 33.55M 参数，总量约 262M，仍处于当前项目量级。

### 2.2 参数账本

以下是按照本方案矩阵尺寸进行的解析估算，**不是已经实例化模型的实测参数统计**。少量 norm、bias、位置参数及最终接口调整可能使数字变化。

| 模块 | 估算参数 M | 说明 |
|---|---:|---|
| 输入 embedding + LM head | 33.554 | 两个 32768×512 |
| 16 层 LatentMoE | 123.998 | 包含 shared、router、down/up 与 norm |
| 12 层 KDA | 16.663 | 含短卷积与完整输出门 |
| 主干 GR 与最终 read | 8.980 | 32 个子层注入模块 + 最终读取 |
| 2 层 QSA-MLA | 2.037 | 含两个独立索引器 |
| 2 层 CSA | 1.284 | 含 compressor 与索引器；使用直接输出投影 |
| 视觉塔 + merger/projector | 22.935 | patch embedding、12 个 block、两层 projector |
| 浅层 lookup | 8.784 | 表本身约 8.389M |
| 一个 MTP 块 | 9.985 | 不重复统计 embedding/head |
| **总计** | **约 228.221** | 工程冻结允许首先在 225～235M 范围核对差异 |

一个 LatentMoE 层的主要计数为：

```text
32 × 3 × 256 × 256             # routed expert matrices
+ 2 × 512 × 256               # latent down/up
+ 512 × 32                    # router
+ 256                         # latent norm
+ 3 × 512 × 768               # full-width shared expert
= 7,749,888 parameters
```

总参数包含所有专家。若按每个文本 token 实际选择的专家、完整输出词表投影和常规投影粗略计数，主干激活参数约 82M；这里排除了视觉编码和 MTP，也只计入本 token 使用的 embedding/lookup 行。这一口径不是 FLOPs，不能据此声称训练显存等于 82M 模型。

实现后必须输出：total / trainable / frozen / vision / main / mtp / embedding / routed / shared / lookup 分项，以及去重后的权重共享关系。不得仅显示 active 参数，把数亿参数的优化器状态隐藏掉。

## 3. 三种注意力通路的组合

### 3.1 层序与统一残差接口

```text
Text IDs ───────────────┐
                       ├─ unified embeddings → 4-stream residual
Image / Video → ViT ───┘

 1 KDA   2 KDA+lookup   3 KDA   4 CSA-4
 5 KDA   6 KDA          7 KDA   8 QSA-MLA
 9 KDA  10 KDA         11 KDA  12 CSA-4
13 KDA  14 KDA         15 KDA  16 QSA-MLA

Each layer: GR read → attention → GR inject
            GR read → LatentMoE → GR inject
Final GR read → RMSNorm → shared output head
                         └─ one auxiliary MTP block during training
```

图中的第 2 层 lookup 实际在该层 attention 前注入，具体接口见第 5 节。最后一层保持逐 token 的全局访问能力。主模型不另加第 17 层，MTP 是独立辅助路径。

初始四流由 embedding 广播建立。GR 的各流 norm 和注入参数独立初始化，不能把所有流与门完全绑成同一个参数，使其永久退化为单流。每个子层只返回自己的更新量；是否加入 residual 只能由 GR 外壳负责，避免拷贝上游完整 decoder 导致二次 residual addition。

### 3.2 KDA：固定大小的循环状态

KDA 的单头参考递推为：

\[
S_t=(I-\beta_t k_t k_t^T)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_t k_t v_t^T,
\quad o_t=S_t^Tq_t.
\]

采用逐通道保留系数，衰减发生在 delta 更新之前；输出使用独立的完整矩阵门。K3 的下界衰减形式采用 `g=-5*sigmoid(exp(A)*z)`、`alpha=exp(g)`，其中 `A` 初始为 0。不能直接沿用旧 Kimi Linear 的无下界 softplus 映射，或者把逐通道衰减退化为每头一个标量。[Kimi K3：Hybrid Attention](https://arxiv.org/html/2607.24653v1#S2.SS1)

本方案接口约束：

- q、k、v 都经过各自短卷积；q/k 的归一化、query scale 和输出 norm 在 reference 与 kernel 中只执行一次。
- recurrent state 使用 FP32；12 层、8 头、64×64 状态约 1.5MiB/序列，另加短卷积状态。训练激活不包含在这个数字中。
- 不给 KDA 直接拼接 Qwen mRoPE。图像的空间信息由视觉编码及后面的 token MLA 提供，KDA 根据统一输入序列递推。
- packed document、独立对话之间必须同时重置 KDA state 和短卷积历史；单个对话中的多模态块不重置循环状态。
- reference 使用逐步 FP32 递推；优化实现使用 chunk kernel。任何移植先对齐输出和梯度，再比较性能。
- 长度不是 chunk 整数倍、padding、空序列、前缀复用和 speculative rollback 都需要明确语义。

不要依据固定 state 容量声称整个模型具有固定 KV 显存，另外两类通路仍有随序列增长的历史。

### 3.3 CSA-4：历史 KV 本身被压缩

本方案保留两支投影与门控池化的压缩思路，4 个新 token 对应一个压缩位置，并允许同一段内与前一块重叠。局部窗口保留最近 128 个逐 token KV。CSA 选中的历史单元直接作为压缩 KV 参与注意力，**不会再展开成原 token**。

每层设置：共享 KV 维 64，4-token compressor，压缩历史 top-64，局部窗口 128。到 8K 时压缩历史约 2048 项；索引器从中选择 64 项。短序列候选不足时读取全部可见项。

主模型 phase 分两种：

1. `dense_reference`：局部窗口 + 所有已完成、可见压缩块；不进行 top-k 剪枝。
2. `sparse`：局部窗口 + 索引选择的压缩块。

这里 `dense_reference` 不是恢复成全长逐 token attention。CSA 与 QSA 的 dense reference 定义不同，训练器和日志要区分。

压缩器的 pooling 在 FP32 中执行，然后归一化，再对指定位置维施加 1D RoPE。默认使用块起点的序列位置，位置维 16，theta 10000；该选择是本版长度和接口适配，不沿用百万上下文扩展参数。局部 query/KV 和压缩 KV 必须使用同一层的频率体系。[DeepSeek-V4：CSA](https://arxiv.org/html/2606.19348v1#S2.SS3.SSS1)

**多模态修改：** compressor 分段不得跨 packed document、媒体实例或 text/vision 边界；重叠仅发生在同段内部。遇到不足 4 个 token 的段尾，边界到达后将其作为短块 flush，池化的无效槽置 `-inf`；在边界尚未到达的流式处理中，它仍保留于 pending/local 通路。必须记录真实 `member_indices`、起止位置和 `complete_at`，不能用全局 `index*4` 猜成员。

这会使压缩项数量不再严格等于 `N/4`。短媒体段可能提高实际缓存开销，需要按真实 block 数统计。不能把修改后的 compressor 宣称为官方 V4 原封不动的实现。

### 3.4 QSA-MLA：压缩索引，逐 token 读取

这一模块是本方案最需要单独实现和验证的新接口：借用 QSA 的微块筛选方式，连接 decoupled-position MLA 的逐 token latent KV。

张量设计：

```text
x:                  [B, T, 512]
q_latent:           [B, T, 128]
q_content:          [B, T, 8, 64]
q_position:         [B, T, 8, 32]
kv_latent c:        [B, T, 128]
k_position:         [B, T, 32]       # shared across heads
up(c) -> k_content: [B, T, 8, 64]
up(c) -> v:         [B, T, 8, 64]
```

位置支路独立于 latent 内容分支。注意力 logits 是内容内积与位置内积之和，除以 `sqrt(64+32)`；softmax 在 FP32 中归约。输出加入 sigmoid gate 后投影回 512。默认显式实现，验证通过后才做投影吸收与 latent-space decode 优化。

索引器从 **该 attention 的 GR read 结果** 获得独立投影，不能错误接到 attention 输出、MoE 输出或另一个层的 residual。用 4-token 微块产生轻量 index key，top-64 块展开后最多 256 个 raw token。再并入局部窗口 128、可见的当前未完成块，以及媒体保留集合；去重后进行 core MLA。

QSA 微块是索引目录，必须保留逐 token latent KV。不能为了省内存把微块摘要当作 core value，否则实现已经变为另一种压缩注意力。这个区别是融合设计成立的前提。[Qwen 架构报告：QSA](https://arxiv.org/html/2608.30320v1)、[Qwen 实现参考](https://github.com/huggingface/transformers/blob/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

QSA 目录也采用真实 block registry，不跨 segment/media 边界；只有 `complete_at <= query.cache_position` 的块允许参与索引。当前未完成块由 local raw 路径提供。全序列训练即使提前算好了未来块的 index key，也必须在 score/selection 前屏蔽，不能只在最终 raw attention 上补 mask：否则选路已泄漏未来信息。段尾短块的有效成员与 padding 分别记录。

`dense_reference` 阶段直接对所有因果可见逐 token KV 做 MLA；索引器只学习选路，不能改变参考输出。切到 sparse 后，MLA 投影和 gate 延续同一 checkpoint。

### 3.5 必须保留视觉细节的规则

如果当前图像有 833 个视觉 token，而 raw retrieval 仅 256 个，随机初始化的索引器很容易把 OCR 需要的局部区域全部排除。因此 QSA-MLA 使用显式 `protected_media_indices`：

- 当前完整请求中已提供的图像/视频视觉 token 总数，默认不超过 1024；都作为可保留集合。
- 文本 query 只能读取因果上已出现的媒体；packed 的其他样本不可见。
- 保护集合并入 top-k 和 window，而不是挤占后再被截断掉。
- 保护逻辑只能使用输入媒体的位置与配置，不能读取答案、标注框、教师 attention 或未来问题。
- 保护的是 latent KV 中对应 token 的位置，不是复制另一份视觉隐状态缓存。
- 超过媒体预算时，processor 根据公开规则调整帧数、分辨率或分成多轮；不能在 attention 内静默丢图。

默认单层每个文本 query 的读取上界约为 `256 retrieval +128 local +1024 protected`，实际去重后通常更少。这意味着视觉场景的稀疏收益比纯文本弱，是为保留细节接受的代价。必须分别报告 text-only、单图 OCR、多图、视频性能。

CSA 不额外保留全媒体 raw KV；这让它继续承担压缩历史角色。最终两层 QSA-MLA 提供原粒度视觉访问。是否还需在第 12 层 CSA 加有限视觉锚点，作为后续消融，不默认引入第四种长期缓存。

### 3.6 复杂度与缓存账本

| 通路 | 历史存储 | 每次 decode 的主要访问 | 不能声称什么 |
|---|---|---|---|
| KDA | 固定 state + 短卷积历史 | 当前投影 + state update | 不代表全模型固定内存 |
| CSA | 压缩 KV 与 index key + 窗口 + pending | 扫描压缩索引、读取 top-k 与窗口 | 索引扫描没有消失 |
| QSA-MLA | 所有 raw latent KV + block index key | 扫索引、取选定 raw KV，含媒体保护 | 稀疏读取不等于稀疏存储 |

按候选 BF16 缓存，2 层 QSA-MLA 的原始 KV 约为 `2×N×(128+32)×2 bytes`。N=8192 时约 5MiB；还需索引、metadata、KDA 状态、CSA 和运行时工作区。若采用显式解压缓存，则要另计 expanded k/v，不得仍只报 latent 字节数。

聊天中的“若四类历史项同宽，将部分 N 变成 N/4，条目数减少”只能作为条件算式。此处 KDA、CSA、MLA 的维度、精度和索引不同，应按实际字节重算。训练 prefill 若为每个 query 扫完整 index history，仍有约 `O(N²/4)` 的索引成本；Python mask 实现也可能分配 `N×N`。只有真实稀疏 gather/kernel 才可能兑现收益。

## 4. GR、位置编码与因果边界

### 4.1 四流 GR 的适配

残差张量统一为 `[B,T,4,512]`。GR 对每流独立 RMSNorm；flatten 后用低秩门生成 elementwise read 权重，沿 stream 维归约出 512 维子层输入；再为各流计算注入比例，把 attention/FFN 更新写回原 residual。

建议直接复用现有 GR 原语的 `read/inject` 数值语义，包括流数的缩放位置，而不是根据文字说明重新写一个近似版本。源实现中低秩中间量与注入门有与流数有关的 rescale；多一次或少一次除以 4 都会改变优化尺度。输出 head 的最终 GR 只读、不再计算注入。

本版 GR 不兼容 Kimi decoder 内部维护的 AttnRes block list，也不兼容 V4 mHC 的 Sinkhorn 状态。移植边界是 attention/FFN 原语，decoder 外壳单独实现。Qwen 参考代码定位见上节链接，工程复用路径见第 15 节。

### 4.2 位置体系必须明确分开

| 模块 | 位置设计 | 原因 |
|---|---|---|
| KDA | 不额外施加 RoPE | 保持递推定义 |
| CSA core/index | 1D，rope dim 16，块采用真实起点 | 压缩后的空间混合不直接套 mRoPE |
| QSA index | 1D 序列位置，head dim 32 中的 16 个旋转实维 | 保持目录有时间顺序，最终精确位置由 core 处理 |
| QSA-MLA core | decoupled mRoPE，32 实维 | 为图像网格/视频提供多轴位置 |
| Vision ViT | 独立空间位置体系；时间 patch 有明确语义 | 不能与 decoder 缓存共用频率表 |

QSA-MLA 的 32 个旋转实维分为 temporal 8、height 12、width 12；对应频率对数为 4、6、6。若 helper 使用 half-dimension sections，应传 `[4,6,6]`；若使用实际维度，应传 `[8,12,12]`。必须测试而非只检查总数能否整除。

文本位置三个轴使用同一个连续 offset。视觉块使用 processor 生成的 `(t,h,w)`；下一段文本从前一块三个轴最大值加一开始。独立 `cache_position` 始终是 flattened 序列的单调位置，不能用空间坐标替代缓存长度。图块附带原图坐标和 tile 标识；视频 temporal 坐标基于真实时间间隔量化，不能把不等间隔抽帧都当成等时间距离。

位置基数、轴划分、patch、merge 和坐标 offset 算法都进入配置哈希。8K 外推先不承诺；后续 YaRN/位置扩展必须增加单独训练与长程评测。

### 4.3 全序列、packing 与媒体掩码

本版 decoder 主路径采用严格因果顺序；视觉塔内部可对一张图或一个时间 patch 组做非因果空间编码。图像在文本回答之前完整编码，因此回答能访问整张图。decoder 不为视觉 token 额外开放跨未来文本的双向 attention，避免三个通路使用不一致的可见性定义。

每个 token 至少有 `segment_id、position_ids、cache_position、modality、media_id、label_mask`。attention 可见性、KDA reset、lookup reset、compressor flush、MTP shift 和 loss mask 都由同一份 segment 元数据派生。特别检查：不能只是 CE mask 隔离，attention 和循环状态仍跨样本泄漏。

视觉 placeholder 按实际特征数展开。展开后的序列才是预算、causal mask、MTP 和 labels 的依据。若一个特殊 image token 被替换为 196 个向量，则必须同步更新所有这些张量，不能只改 embeddings。

## 5. LatentMoE 与浅层 lookup

### 5.1 专家路径

新模块按以下结构实现：

\[
y_{routed}=W_{up}\operatorname{RMSNorm}\left(\sum_{e\in top4}p_e E_e(W_{down}x)\right),
\quad y=y_{routed}+E_{shared}(x).
\]

router 从 512 维输入计算分数，latent expert 在 256 维运算，聚合与权重归约用 FP32。归一化位于 expert 聚合之后、up projection 之前；不能误放在每个专家内后就认为等价。共享专家保持全宽，是针对 latent bottleneck 的细节通路。

默认 routed/shared 都采用现有适配中可定位的 SiTU-GLU：令 `g=W_gate x、u=W_up x`，激活为 `[4*tanh(g/4)*sigmoid(g)] * [25*tanh(u/25)]`，再经过 down projection；归约前在 FP32 中计算。4/25 是本方案起点，沿用已读适配的参数化，正式 M0 仍须与固定上游逐项核对；普通 SwiGLU 保留为消融。shared expert 的中间维独立设为 768，不能借用 `num_shared_experts=3` 对外宣称有三个独立 shared experts。

router 默认 `sigmoid(W_router x)`，top-4 由加 correction 的分数选择，混合权重从原分数取出并重新归一化到和为 1，初始 routed scale=1。不同 activation/scale 应进入配置，不能依赖上游默认值悄悄变化。

### 5.2 为什么选择 32/top-4

每个 token 激活 12.5% routed experts，明显高于旗舰极稀疏设置。单卡 microbatch 的有效 token 较少，数百专家会造成低覆盖、小 GEMM 和路由噪声。32/top-4 是本次训练充分性与教学可观察性的折中，不是由官方缩放律推导出的最优值。

在 3B 监督 token 下，输入 token 实际更多；按理想均匀路由，每个专家收到总输入 token 的 1/8。但图像、中文、代码分布可能不均，应实际统计不同模态和领域的专家覆盖，而不是用理论值替代。

### 5.3 平衡策略

默认采用 quantile balancing 的实现路线，路由 correction bias 只用于选择，真正混合权重仍来自未加 correction 的分数。其目标是在小模型上减轻热门专家垄断；是否优于简单 loss-free 更新需要消融。

单卡适配要求：

- 统计覆盖一个 optimizer step 的全部 gradient accumulation microbatch，不能每个 microbatch 独立大幅更新 bias。
- 首 100～500 optimizer steps 使用平滑启动；每次更新 bias 最大绝对改变量候选 1e-3，先记录直方图和分位数，再根据 pilot 调整。
- balancing 更新不通过反向梯度；bias/直方图/EMA 必须写入 checkpoint 并精确恢复。
- 不在默认目标中同时叠加大系数 load-balancing auxiliary loss。router z-loss 如需稳定 logits，从 1e-4 单独试验并注明它不是负载平衡项。
- no-token-drop：所有已选 token 必须经过专家；capacity overflow 不得悄悄丢弃。训练精度先由排序、分桶、gather/scatter reference 保证。

每层记录 `max/mean load、min/mean load、zero-use fraction、router entropy、top-k margin、各模态覆盖、shared/routed output RMS`。设预警而非自动判死：连续 200 个 optimizer steps 中 max/mean >2.5，或某专家零使用，应检查数据和路由实现。专家因领域专化偏载不一定有错，必须联系验证损失判断。

### 5.4 单点 lookup 的具体定义

仅第 2 层 attention 前使用：读取该 attention 已有 GR 的聚合输入 `h`，从不跨 segment 的 token ID 前缀提取 2-gram/3-gram；每阶两个固定 hash，各表 32768 行×64 维，拼接为 256 维。投影 `256→512`，与 `h` 生成的 sigmoid 门相乘，再经短深度卷积调制。结果乘可学习系数 eta 后广播加入四流，然后用同一个 attention GR 重新读取，进入 attention。此处不新建一套额外 GR 参数，独立记录 lookup 的注入 RMS。

为便于 228M 账本，门使用 `512→512`，卷积 kernel=4；默认注入系数从 0.01 初始化，不把全部模块硬置零导致长期无梯度。hash 函数、seed 和映射规则版本化，碰撞率通过真实语料测量。

媒体 token 不参与 n-gram token-ID 查表：图像在 placeholder 展开后不应几百次查询同一个 image ID。遇到媒体、padding、segment 边界重置 n-gram 历史；控制 token 可按固定白名单保留，但不能把 role boundary 当成普通连续词串。这是针对多模态输入的本地设计。

必须有 `lookup=off` 的等参数对照：把约 8.8M 参数转给 shared FFN 或 backbone 宽度后比较。若 lookup 只提高训练集短语记忆而损伤新组合、OCR 字符精确性，则关闭它并重新记录架构版本，不能为了“包含 Engram”保留无收益模块。

## 6. 原生视觉与短视频实现

### 6.1 初始化路线

**主路线所有核心权重随机初始化**：语言主干、视觉塔、projector、专家、lookup、MTP。视觉从主预训练 P0 起参与联合训练，不先训练纯文本数十亿 token 再给随机视觉塔几千张图。

另设一个已训练视觉塔的工程对照，用于判断视觉学习是否成为瓶颈；该对照必须标为 `pretrained-vision`，不能混进全随机主路线的训练成本或能力结论。若最后选择已有视觉权重作为产品版，也应分别命名，保持训练来源清楚。

### 6.2 视觉塔与 token 数

patch embedding 使用 `Conv3d(3→384,kernel=(2,16,16),stride=(2,16,16))`。静态图复制为两个相同帧，只产生一个时间 patch 组；视频按真实相邻帧配对。空间合并 2×2 后，每个视觉 token 代表 32×32 像素对应区域。

| 输入方式 | 合并后视觉 token | 使用阶段 |
|---|---:|---|
| 静态图 224×224 | 49 | P0/普通图像，低成本预热 |
| 静态图 320×320 | 100 | P1 普通图像 |
| 静态图 448×448 | 196 | P2/SFT 单图细节 |
| 静态图 512×512 | 256 | 分辨率消融 |
| 全局 224 +4 个 448 tile | 49+4×196=833 | 文档/OCR 高分辨率 |
| 8 帧 224 视频 | 4×49=196 | 时间 patch=2，短视频 |
| 16 帧 320 视频 | 8×100=800 | P3/SFT，受 1024 预算约束 |

表内未计入媒体边界、时间戳等文本控制 token。不同宽高的图按 32 的倍数调整，保留纵横比并记录变换。不能先强制方形拉伸再用 OCR 精确率评价原图。

高分辨率文档不先缩成 224 后再裁 tile；应从原图进行有重叠的局部裁切。全局图和 tile 的顺序、来源框、尺寸、原图坐标进入 metadata。最多 4 个高分辨率 tile 是初始产品预算，超过范围时用户可选局部区域；不能承诺任意整页小字均可辨认。

### 6.3 防止多模态“表面接通”

以下证据缺一不可：

1. `pixel_values` 改变时，输出 visual embeddings 和回答确实变化。
2. 图像/文本错配时，视觉验证任务显著下降。
3. 图像置黑、打乱 patch 或隐藏时，模型不能维持几乎相同答案。
4. 图像编码器与 projector 的梯度在应训练阶段非零，参数实际更新。
5. 输入图像与文本答案的分组划分避免同图、同页、相邻视频片段泄漏。
6. 不使用文件名、图片 URL、caption 或原始标注字段作为额外输入，意外把答案交给模型。

视觉塔学不好时，先检查数据、label 与图像错配、分辨率和梯度，再考虑增加视觉蒸馏。不能直接用更多 DPO steps 补救视觉编码器未训练。

### 6.4 短视频训练

数据必须含对象变化、先后关系、动作、计数和至少两帧才能回答的问题。先用 4/8 帧，后加 16 帧；记录原始 FPS、采样时间、时间 patch 组合。快动作任务避免时间 patch 跨过过长间隔。

视频 benchmark 同时运行：正常顺序、倒序、只给首帧、打乱帧、帧数 4/8/16。如果时间问题在只看首帧时同样正确，多半是在测场景先验。视频能力发布以正常顺序相对这些对照的收益为证据。

初始视频能力定位为十几秒到几十秒片段的粗粒度理解，实际时长取决于采样密度；不宣称实时视频、音画理解或长电影记忆。跨多图推理与时间推理分别验收。

## 7. Tokenizer、数据结构与数据工程

### 7.1 先冻结 tokenizer

准备与正式训练同分布、已排除验证数据的 5～10GB 文本样本，比较 32K 与 64K BPE。候选采样按中文 45%、英文 25%、代码 15%、数学 10%、结构化/OCR 文本 5% 的字节配比组织；这只是 tokenizer 建表分布，不等同于训练 token 配比。

保留 byte fallback、空格与缩进、换行、数字和公式中的重要符号。不做会破坏代码、单位、正负号、小数点、全半角意义的过度归一化。特殊 token 预留空间必须计入 32768，而非先生成 32768 个普通 token 后再随意增加。

最低控制协议：`bos/eos/pad、message_start/message_end、system/user/assistant/tool、final/thinking、tool_call/tool_result、image_start/image_end、video_start/video_end、frame/time`。具体字符串可沿用已有可维护的协议，但本模型保存独立 `chat_template` 和版本。普通用户文本中与控制 token 同形的字符串必须转义或安全编码。

冻结前报告中文每字 token、英文每词 token、代码每行 token、OCR 数字/罕见字切分、解码可逆率和输出 head 成本。两个 tokenizer 不能只比较 token-level NLL；跨词表比较使用同一文本的 byte-normalized NLL、任务正确率、吞吐和序列长度。若 32K 使核心中文/OCR样本长度相对 64K 增加超过约 15%，把 64K 列为正式候选，由 pilot 决定，阈值是工程决策。

正式训练前冻结 tokenizer SHA。后续所有教师、SFT、OPD、draft、量化导出、Demo 使用同一版本；词表更换应视为新训练系列。

### 7.2 样本 schema

```json
{
  "sample_id": "stable-id",
  "source": {"dataset": "...", "revision": "...", "record_id": "..."},
  "split_group": "document-or-image-or-video-group",
  "language": "zh",
  "domain": "ocr",
  "messages": [
    {"role": "user", "content": [{"type": "image", "media_id": "m0"}, {"type": "text", "text": "读取图片中的编号。"}]},
    {"role": "assistant", "channel": "final", "content": [{"type": "text", "text": "..."}]}
  ],
  "media": [{"media_id": "m0", "uri": "local-or-shard-location", "sha256": "...", "width": 0, "height": 0}],
  "supervision": {"type": "answer_ce", "verifier": null},
  "provenance": {"license_record": "...", "teacher": null, "transform_version": "..."}
}
```

预处理后另存 token IDs、loss mask、media spans、grid、segment IDs、位置元数据、有效 CE 数、视觉 token 数，不把一次 decode 后不稳定的 URL 作为唯一媒体来源。示例中的占位值不是可用训练记录；validator 对空 hash、零尺寸、缺失媒体必须拒绝。

### 7.3 候选文本来源与用途

| 来源 | 本模型用途 | 筛选与限制 |
|---|---|---|
| [FineWeb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | 中文知识、解释性预训练 | 去模板、去乱码、检查简繁和教育质量；不能假定全部中文都是高质量 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 英文高质量网页 | 保留完整段落；控制源域名重复和极长列表 |
| [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 教育文本、短故事、代码候选 | 分子集取样；故事仅作为早期语言学习补充 |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 数学解释与推导 | 限制难度和长度；先基础算术/代数，后复杂推理 |
| 自建可执行代码样本 | Python 基础、小函数和 API 使用 | 固定运行环境、保留测试、排除评测题与近重复 |
| 自建结构化文本 | JSON、表格、单位、OCR 样式转写 | 程序可验证；模板族分组切分 |

训练文本 pool 建议准备至少 5B 去重候选 token，供质量、领域和重复率筛选，最终主训练消费 3B CE 中的文本份额。不能把下载行数当作清洗后 token 数，也不能因流式下载方便而忽略最终重复统计。

### 7.4 候选视觉与指令来源

| 来源/方式 | 用途 | 关键处理 |
|---|---|---|
| [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | 多源视觉指令、识别、推理 | 按 source/subset 挑选；不整包盲采；不同源可能存在同图 |
| [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) | 文档问答、版面文本 | 按原始文档划分；自动生成答案要抽查；避免学习错误 OCR |
| 自建中文合成 OCR/表格/图表 | 补足中文、编号、单位、版面精确性 | 字体、模板、词表和渲染 seed 分离；不全部使用同一背景 |
| 经过逐项确认的公开图文子集 | 自然图像 caption、对象关系 | 保留图片来源与许可记录；caption 不能作为额外隐藏输入 |
| 合法取得的短视频 + 可验证合成视频 | 动作、顺序、变化 | 按原视频分组；时间戳完整；测试模板独立 |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | 短视频问答候选 | 优先 0～30 秒子集；过滤仅描述场景的长答案；按原视频/source 排除重叠 |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | 视频描述与事件候选 | 按片段和时间元数据筛选；不将音频才能回答的问题用于无音频模型 |
| [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 通用多轮指令候选 | 不整段照收；筛冗长、错误回答和格式不符样本 |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | 推理步骤候选 | 先验证 final；长链按能力与长度筛选，不强迫小模型模仿长篇文本 |

数据集卡的总体许可不等于每张图、每个文档或每段代码都具备相同使用条件。落地时为实际采用的子集保存 revision、原始来源、许可/用途约束和可再分发性；无法确认的源进入隔离池。此处列候选数据，不宣称已经下载、获准再分发或完成清洗。

FineVision 的量级足够大，按子集流式筛选比全量下载更合适。视频以 LLaVA-Video 的短片子集和 FineVideo 的可用片段为候选主池，配合可验证合成视频；须先完成具体 source 的使用条件、可下载性和抽查，才进入 manifest。初始视频暴露按真实公开视频 60%、可验证合成 30%、人工核对的授权短片 10% 组织；最后一项不足时暂分给前两类并记录变更。当前查到数据卡不等于已拥有可用的本地训练池。中文视频监督可以从经过核对的英文答案翻译并复核数字/时序，不能只替换提问语言后仍默认答案正确。

### 7.5 清洗、去重与验证分组

推荐顺序：source allowlist → 解码与 schema 检查 → 基础质量 → 精确去重 → 近重复 → 组级切分 → 评测污染检查 → tokenize/media 处理 → shard manifest。

- 文本同时计算规范化 hash 和原始 hash；近重复以文档为单位，不能把相关段落随机分到 train/val。
- 图片使用内容 hash + 感知 hash；同图不同问句视为同一 split group。裁剪图和原图也应归在同组。
- 文档按文档 ID，视频按原视频 ID 划分；图表模板、合成代码题型生成器使用分离模板族和 seed。
- 去重跨数据集执行。FineVision 中的记录与独立下载的文档/图文源不能重复计入 unique coverage。
- 默认按 group 做 98.5/0.5/1.0 的 train/validation/test，再对低频领域设置最低数量；最终以 manifest 中的精确计数为准。
- 正式 test 不用于选择数据混合、LR 或 early stopping。固定 demo prompt suite 也不能回灌训练。
- 一个 media pair 可生成多个问题，但不能把它们当成多张独立图。报告 unique media、unique text、seen examples 和 repeats。
- 源文本长度和图片尺寸异常须有剔除理由。OCR 不做一般图像训练里的水平翻转、裁掉文字后仍沿用原答案等增强。

每个 shard 保存 SHA256、记录数、CE token、input token、image count、video count、领域分布和清洗版本。训练前随机抽查每个主要源至少 100 条；对答案自动生成的视觉源增加至 300 条，记录错配、幻觉、OCR 错字和低分辨率比例。

## 8. 全流程训练路线与预算口径

### 8.1 三种 token 必须分别记账

1. `input_tokens`：实际进入 decoder 的文本、控制符和展开视觉位置；决定主要计算量。
2. `ce_tokens`：`labels != -100` 且经 shift 后有效的监督位置；主预训练 3B 指这个数量。
3. `generated_tokens`：RL/OPD/draft rollout 实际生成的 token；另记 teacher forward、verification 和训练 replay 成本。

同时记录 `vision_tokens、unique_media、media_exposures、mtp_target_tokens、index_query_tokens`。MTP 的第二预测目标不能再计入主 CE 预算；indexer-only 阶段也不能伪装成新增语言学习 token。

预训练纯文本通常大部分可见 token 都受 CE 监督；caption/VQA 只监督文本目标，图像本身不作为离散 token 预测。SFT 只监督 assistant 指定 channel；user、tool observation 和图片位置默认 mask。EOS 应在可学习的结束位置保留监督。

### 8.2 完整阶段图

```text
tokenizer / data contracts / unit reference
  → smoke + 20M controlled pilots
  → P0 native multimodal bootstrap       0.20B CE
  → P1 dense-reference pretraining       0.80B CE
  → I  indexer-only distillation         40M input (separate)
  → P2 sparse multimodal pretraining     1.40B CE
  → P3 long-context + cooldown           0.60B CE
  → S  multimodal / tool / effort SFT    120M assistant CE
  → R  verifiable curriculum RL          10M–30M generated
  → T  qualified domain/effort teachers   2M–5M generated per teacher
  → O  multi-teacher on-policy KL         10M–30M student generated
  → A  preference alignment, if needed    2M–8M response CE-equivalent positions
  → D  MTP-derived draft training         5M–15M rollout tokens
  → Q  optional quantization calibration/QAT
  → end-to-end evaluation / export / Demo
```

P0/P1/P2/P3 合计 **3.00B 主 CE**。pilot、indexer、SFT、RL、蒸馏和部署训练不计入这 3B。A/Q 是完整流程中的有条件分支：无必要时记录 `not_selected` 及证据，而不是伪造一个已完成训练阶段。

全流程表示全部阶段都有实现规格和验证方法，不表示基础能力没达标也必须按时进入 RL。3B 是立项预算，不是保证充分训练的缩放定理；如果 P3 之后 held-out loss 仍明显下降且基本生成弱，应评估追加到 5B/10B 的成本，不能宣布“预训练完成”后靠 DPO 修复语言能力。

预算确实按本模型定制：主干去掉视觉塔和 MTP 后约 195M；3B CE 相当于约 15.4 个监督 token/主干总参数，但 MoE 的访问分布、非监督视觉位置、lookup 和 LM head 使这个比值不能直接套 dense scaling law。选择 3B 是在当前单卡量级下给语言与约 630M 视觉文本监督留出完整学习过程的首轮预算；是否充分由验证曲线、专家覆盖和视觉测试决定。不能按约 82M active 参数就把全部训练量缩到几十 M。

### 8.3 主预训练阶段表

| 阶段 | CE 预算 | 序列长度分布，按 batch 数 | 视觉 CE 目标份额 | 媒体暴露目标 | 主要改变 |
|---|---:|---|---:|---|---|
| P0 | 200M | 512 为主，少量 1024 | 20% | 约 0.3M 图；暂不要求视频 | 全随机联合学习；低分辨率 |
| P1 | 800M | 1024:2048 = 70:30 | 20% | 约 1.2M 图、20K 视频 | 完整 dense reference；提高语言与视觉质量 |
| I | 40M input | 1024:2048:4096 = 20:50:30 | 沿用 P1 领域采样 | 不作为新增视觉学习统计 | 主干冻结，训练索引器 |
| P2 | 1.40B | 2048:4096 = 70:30 | 20% | 约 2.5M 图、60K 视频 | 稀疏联合训练；高分辨率 OCR |
| P3 | 600M | 2048:4096:8192 = 25:45:30 | 25% | 约 1.0M 图、40K 视频 | 长程训练、质量提升、学习率下降 |

媒体数字是数据准备目标，不是与 CE 独立的硬停止条件。约 5M 图像暴露、至少 1.5M unique 图，以及约 120K 视频暴露，需要实际按目标文本长度重新核算。视觉 CE 总目标为 630M，其中视频和图片共同占用；如果平均视觉答案长度短，达到媒体暴露目标时仍不足 CE，就需新增高质量文本目标或更多独立样本，不能无限复制同一张图凑 token。

按经验设单图最多约 3～5 次有效曝光的观察范围，特殊稀有任务可例外但要披露；不要把强数据增强当成增加了 unique media。全部阶段累计统计重复，不在阶段切换后清零。

### 8.4 各领域混合

在每一阶段的**纯文本 CE**内部，起始目标为中文通用 42%、英文通用 28%、代码 12%、数学 10%、结构化/检索/工具文档 8%。P0 可将一半代码/数学份额替换为浅显解释、短故事和基础题；P2 后恢复上述分布。所有变更保留确切比例与有效 token 统计。

视觉 CE 内部起点为 caption 30%、OCR/文档 35%、VQA/关系 20%、图表/表格 10%、视频 5%。P0 尚未引入视频时，将该 5% 分给 caption；P2/P3 补足视频学习量，避免文字任务全部吞掉视频配额。

以上分布按 CE token 计算，不能直接作为 sample sampling probability。数据构建器应通过每个 bucket 的实测平均有效长度换算采样权重，运行时每 1M CE 校准一次比例。跨阶段 checkpoint 保存 bucket 已消费计数，恢复时继续。

### 8.5 P0：语言与视觉一起启动

P0 不执行独立视觉冻结：全随机视觉塔冻结会使 projector 学习固定随机特征，违背主路线。全模型参与训练，但 vision LR 较小，图片优先使用 224，任务优先是短而正确的 caption、对象/颜色/数字识别和基础 OCR。

前 2M CE 线性 warmup；初始 image batch 避免难图和长答案。前 20M 用于检查基础语法趋势、视觉梯度、各模块 RMS、router 覆盖与 EOS。小批量过拟合成功只是正确性证据，不是 P0 的能力结果。

如果随机视觉塔导致语言优化完全失稳，可在 pilot 中比较视觉特征辅助蒸馏或视觉预热，但改变主路线前应明确：额外目标、教师、数据和成本。不能临时替换预训练 vision 权重而仍沿用 all-random 的名称。

### 8.6 P1：建立可用的 dense reference

P1 延续 P0 权重和优化器状态。QSA-MLA 对因果可见 token 做全量注意力；CSA 对窗口和全部已完成压缩块做注意力。索引器可以前向记录，但尚不能控制主 attention。

训练数据加入普通网页、解释文本、基础数学与可执行短代码，多模态从 caption 过渡到 OCR/VQA。MTP 从早期即参与，系数见第 11 节。该阶段后必须能出现连贯的文本续写，且图像错配明显影响视觉题效果。

P1 不要求 base 模型像聊天助手一样熟练遵循 role 协议，评估时使用与 base 训练匹配的 continuation/短问答格式。不能把尚未 SFT 的 chat-template 输出差直接定性为架构失败。

### 8.7 I：索引器单独学习

冻结主干、视觉塔、LM head、GR、MoE、主 attention 投影与 compressor；只训练四个 indexer 自身参数。CSA indexer 自有 compressor 可训练，core compressor 冻结。采用 P1 checkpoint，保存其 hash。

对 QSA-MLA，teacher 为本层 dense reference 的逐 token attention 分布，将 token 概率按真实 `member_indices` 汇总成块概率。对 CSA，teacher 为本层窗口与全部压缩项的 dense reference 分布，仅取压缩候选质量并重新归一化。分别计算：

\[
L_{index}=KL(\operatorname{stopgrad}(p_{dense,block})\|p_{index}).
\]

窗口与 protected media 通常无需索引决策，对应块在评价“检索候选召回”时单独列出；不能把它们加入分子让索引器看似接近满分。候选为空或 teacher 对历史候选质量极小的 query 跳过，并记录比例。

每个样本只抽 64～128 个有效 query 计算 reference 分布，query 分块重算，避免为全部层保存完整 `T×T` FP32 attention。sampling seed 和有效 query 数写入日志。40M 是输入 token 预算，另记实际 KL query 数。

indexer LR 候选 `1e-4`，warmup 1M input，目标 `captured_attention_mass@budget ≥0.90`，至少分别在纯文本、代码、OCR、视频验证。0.90 是工程准入目标，不是官方阈值。未达标时先提高 top-k/分析错误，不直接切 sparse。

### 8.8 P2：从 dense 平滑切到 sparse

恢复所有应训练参数；main optimizer moments 延续 P1，indexer moments 延续 I，明确合并规则。禁止把 indexer-only checkpoint 的“冻结参数组状态”误当成主干新训练的空状态。

前 20M CE 采用逐 batch 稀疏概率从 0 增至 1；同一 batch 的执行 phase 确定且可复现。这个过渡是本地策略，避免突然改变检索集合。CSA top-64 compressed，QSA top-64 blocks；indexer auxiliary KL 系数起点 0.01，梯度仅更新 indexer，teacher 分布 stop-gradient。KL query 以固定采样比例计算并记录额外计算。

切换前后用同一 checkpoint、同一验证样本比较 dense/sparse NLL 和视觉任务。候选准入：总体 NLL 相对增加不超过 1%，OCR/VQA 绝对准确率下降不超过 2 个百分点；若不满足，先试 top-128 和延长 I 阶段，不能用继续训练掩盖选路实现错误。

P2 增加 448 图和 OCR tile、多图与 8 帧视频。视觉输入比例提升时同时记录 CE/s 和 input/s，不能把因为视觉 token 增加造成的 CE/s 下降误认为 kernel 退化。

### 8.9 P3：训练长程与收敛

P3 的 8K 数据必须包含有效的长距离监督：跨段资料回答、长文摘要、分散编号定位、多图引用、文档表格联合问答、短视频与文本说明。只是把无关文本拼成 8K，不等于学会长上下文。

从 4K 扩到 8K 时先做 10M CE 过渡，逐步提高 8K batch 占比。完整输入长度包括视觉 token 和答案预留。正文、图片或工具结果不能静默截断；采用可追踪裁剪，目标依赖的内容被裁掉时整条样本丢弃或重建答案。

最后 600M 采用平滑衰减至峰值 LR 的 10%；高质量数据占比适度提高，仍保留至少 50% 来自 P2 的通用领域，以防长程/OCR 样本覆盖掉语言能力。记录 2K/4K/8K 分层 NLL 和 fixed prompts，不只观察混合后的平均曲线。

## 9. 优化器、数值与单卡训练配置

### 9.1 主 reference 先用 AdamW，Muon 做有预算的比较

新组合同时改变残差、注意力、MoE 和视觉，直接套用某个旗舰的 Muon LR 缩放不可靠。默认正确性/reference 路线用 AdamW；在 20M pilot 中比较 per-head Muon 适配，胜出的配置才进入正式训练。

| 参数组 | AdamW 初始 peak LR | weight decay | 备注 |
|---|---:|---:|---|
| 主干矩阵、shared/routed、projector | 3e-4 | 0.1 | 试 1.5e-4/3e-4/6e-4 |
| embedding、LM head、lookup 表 | 3e-4 | 0.01 | 检查罕见 token 和 lookup RMS |
| 视觉塔 | 1e-4 | 0.1 | 试 1e-4/2e-4；按视觉验证选 |
| norm、bias、GR gate 标量、KDA 衰减参数 | 1e-4 | 0 | 矩阵门仍按其形状独立分组 |
| I 阶段 indexer | 1e-4 | 0.01 | 不改主干参数 |

AdamW 起点 `betas=(0.9,0.95), eps=1e-8`，全局 gradient norm clip=1.0。以上是本量级搜索中心，不是声称官方采用这些 LR。

Muon 试验要求：二维矩阵明确分类；head projection 按 head 分组正交化；embedding/head、norm、bias、router correction 不进入同一矩阵更新规则。显式记录 Q/K/V、MLA factor、GR factor、vision qkv、expert matrices 各自适配方式。每个 trainable 参数必须恰好进入一个 optimizer group；新增模块不能因命名不匹配被漏掉。

Muon LR 候选 0.003/0.01 的前提是适配器已经固定其更新归一化定义，不能把不同实现的同名 LR 直接比较。用相同 seed、数据顺序、CE 和近似计算预算比较验证损失、吞吐、梯度异常率、视觉指标；20M 只能筛稳定性，最终前 200M 保留观察窗口。

### 9.2 初始化与尺度

- embedding/普通投影从标准差 0.02 的分布起步；输出残差分支候选按 `1/sqrt(2L)` 缩放。与 GR 流缩放共同检查，不能同时重复应用两套初始化规则。
- norm 权重初始 1；GR 低秩投影使用非完全对称初始化，输出及注入均记录初始 RMS。
- KDA 的 `A=0`，衰减 bias 按已核对映射生成目标 retention 分布，记录 0.1/0.5/0.9 分位数；禁止照搬旧字段名却改变实际函数。
- router logits 初期较小，correction bias=0，top-k tie 使用稳定规则；不能用全零 router 永久固定同四个专家。
- compressor gate 初始化要产生近似平滑池化；无效位置始终 masked，边界单元素块不能出现 NaN。
- 新图像投影检查输出 RMS 与文本 embedding 同量级；视觉向量过大可能使四流 GR 门饱和。

每个新模块必须在第 0、1、10、100 step 保存梯度与权重更新范数。很多“loss 下降但学不会”的错误来自某条支路没有训练、head/tokenizer 错位或 mask 只剩少量有效目标。

### 9.3 Batch、累计与 token-normalized loss

目标每个 optimizer step 约 16,384 **输入** token，单卡起始配置如下；具体 microbatch 由包含最大媒体输入的实测决定。

| 序列长度 | microbatch | gradient accumulation | 名义 input tokens/step |
|---:|---:|---:|---:|
| 512 | 2 | 16 | 16,384 |
| 1024 | 1 | 16 | 16,384 |
| 2048 | 1 | 8 | 16,384 |
| 4096 | 1 | 4 | 16,384 |
| 8192 | 1 | 2 | 16,384 |

不同 microbatch 有不同有效 CE 数，不能简单平均每个 microbatch 的 mean loss 再除以 accumulation。正确做法是先累计 CE 的和，再按整个 optimizer step 的有效 CE 数归一化；MTP、indexer 与 RL objective 分别使用自己的分母和系数。全 mask batch 应跳过并报警，不能执行无意义 optimizer step。

LR scheduler 依据累计主 CE 驱动；I 阶段依据 input/query 自己驱动；不能把同一个 step 数套在所有阶段。保存 cumulative tokens、epoch/shard offset、采样 RNG、优化器和 scheduler，支持严格 resume。

主预训练默认一条连续 WSD 风格日程：P0 前 2M CE warmup 到表中 peak，P0 余量/P1/P2 保持该 peak，P3 用 cosine 衰减到 peak 的 10%；indexer-only 暂停主干 scheduler，不消耗主 CE 计数。P2 初段的 dense/sparse 概率过渡不额外重置主 LR。SFT/RL/OPD 是新优化任务，默认重新建立各自优化器和日程；与同一阶段内严格 resume 的语义分开。

### 9.4 数值与显存

按 228.2M 参数和每参数 16 bytes 的保守常见训练状态账本，权重/梯度/master/moments 约 3.40GiB。具体实现可能更低或更高，Muon 状态也不同。该数字不包括激活、视觉中间量、attention workspace、完整 logits、allocator 碎片和数据 buffer。

8K×32K logits 的 BF16 张量约 0.5GiB，FP32 约 1GiB。不能为了 CE 便利同时持有主头、MTP、教师与学生的多份全长 FP32 logits。采用分块 loss/必要位置 projection，但校验与完整 CE 的等价性。

默认启用 block gradient checkpointing、BF16 autocast、FP32 softmax/logsumexp/路由聚合、KDA FP32 state；视觉 block checkpoint；序列/媒体 bucket。避免把全 dense attention weights 返回给训练器。Flash/稀疏 kernel 不支持的组合先用正确 reference，不能偷偷切成不同结构。

单张 RTX3090 24GB 的准入线建议为峰值 reserved ≤22GiB，留出运行波动空间。需要分别测 P0/P1/P2/P3、833-token 文档、800-token 视频、SFT、RL update、OPD 和 draft verification。**目前这里只能判断参数状态预算有希望，不能证明最终 8K native multimodal 前后向已能放入 3090。**

### 9.5 PCIe 多卡环境的任务放置

默认把“一个模型的一次更新”放在一张卡上，以梯度累计获得第 9.3 节的 batch。多个互不依赖的任务可以各占一张卡；每个任务默认独立进程、独立 optimizer、独立输出目录，不建立跨任务梯度同步组。CPU 数据预处理、工具 verifier 和磁盘加载可以并行，但仍须观察共享内存、CPU 和存储是否成为瓶颈。

| 工作 | 首选放置 | 多卡使用条件 |
|---|---|---|
| 主预训练、indexer、SFT、DPO、QAT | 单个任务单卡；vision、专家、GR、MTP 与主干同卡 | DDP 只有测得有效增益后启用；不自动因为可见两张卡就启动 |
| 消融、不同 seed、不同领域教师训练 | 独立任务分卡并行，每个任务保持单卡 | 各任务资源与数据读写独立；不需要每 step 同步权重 |
| 评估、离线数据生成 | 可使用另一张空闲卡 | 读取完整且固定的 checkpoint；不读取正在覆盖中的权重文件 |
| RL rollout 与 learner | 单卡分阶段路径始终保留；可另卡放 rollout worker | 传 token IDs、logprob、reward、媒体引用与版本；按批刷新权重，限制 rollout 陈旧度 |
| OPD student 与当前 teacher | 能放下时优先同卡，分时 forward，logits 分块 | 异卡 teacher 必须核算完整条件分布的传输，不能仅看 teacher forward 加速 |
| draft 训练/推理与 target | 能放下时优先同卡 | 异卡必须实测验证往返；不能假定接受率高就能抵消 PCIe 延迟 |

默认不采用 tensor/expert/context parallel，也不把 pipeline parallel 或 FSDP/ZeRO 分片作为 228M 级模型的起步依赖。确有单卡内存压力时，先检查 checkpointing、microbatch、视觉 bucket、分块 logits 和临时张量；仍无法满足时，再比较分片/异卡角色的通信与重计算成本。减少 microbatch 后用累计补齐 batch；调整分辨率/长度/稀疏预算会改变任务或模型行为，须重新验收，不能当作无损开关。

**PCIe 不等于不能使用 DDP。** DDP 可以将梯度通信与反向计算重叠，实际收益取决于拓扑、传输路径、梯度大小和每步计算量。将其作为候选，比较同一全局 batch、相同有效 CE 的完整 optimizer step；同时比较“两卡一个 DDP 作业”与“两卡两个独立实验”的研究产出价值。[PyTorch DDP 设计说明](https://docs.pytorch.org/docs/2.14/notes/ddp.html)

若启用 DDP，第 9.3 节的全局 input batch 为 `microbatch × length × accumulation × data_parallel_world_size`。例如 2K、两卡、每卡 microbatch=1 时，accumulation=4 才与单卡 accumulation=8 对齐。非末次累计的 forward/backward 使用正确的 `no_sync` 作用域；按全部 rank 的有效 CE 数归一化，并补偿 DDP 的梯度平均语义。新增这些规则需先对齐单卡参考梯度，不能直接沿用微批 mean loss。

**OPD 特别关注传输量：** 本方案 32K 词表下，512 个回答位置的 BF16 全词表 logits 为 32MiB，16,384 个位置为 1GiB；FP32 再翻倍。这是按张量尺寸计算的体积，不是实测链路流量。仅发送所采样 token 的 logprob 无法保持本文精确全词表 KL 目标。若采用异卡 teacher，需要分块传输、限制队列，并测 teacher 计算、传输、student 更新的合计耗时；不能为省通信静默改成 top-k KL。

额外 rollout 卡与 learner 按轮/批传模型 snapshot，而非每个生成 token 同步参数；snapshot 的准备/传输时间也计入吞吐。异步队列必须限制 policy version 差距并沿用第 12 节的 on-policy 控制，不能用严重陈旧的样本换表面采样速度。多个领域教师从同一合格基础 checkpoint 分卡训练，最终只传资格指标和完成的权重产物，无需互相同步梯度。

启动时记录实际 GPU UUID、PCIe 拓扑、P2P 可用性、传输路径及有效带宽。仅有 PCIe 不能推定 GPU 间一定能直接 P2P，也不能依据标称 PCIe 代数认定实际带宽。NCCL 的路径选择与拓扑有关，不将强制开启/关闭 P2P 的调试参数写成所有机器通用默认值。[NCCL P2P 配置说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level)

多卡候选在完成单卡 profiling 后测至少 100 个稳定 optimizer steps，报告有效 CE/s、实际通信字节、未被计算覆盖的等待、峰值显存与 GPU 小时/百万 CE。是否启用由节省墙钟时间的幅度和额外 GPU 机会成本决定。保留可复现的单卡基线，约 228M 结构和 3B 主 CE 预算不因调度方式自动改变。

## 10. 多模态 SFT 与能力控制

### 10.1 SFT 预算与数据结构

主预算 **120M assistant CE**，初始 LR=2e-5，vision LR=5e-6，projector=2e-5，warmup 2%，cosine 到 2e-6，通常控制 1～2 个有效 epoch。准备 0.5M～1M 条去重候选指令，再按有效答案长度核算；条数与 token 数不是两个独立保证。

| 领域 | assistant CE 份额 | 目标行为 |
|---|---:|---|
| 中英一般对话/解释 | 30% | 简洁、连贯、多轮一致；中文优先 |
| 工具与结构化调用 | 10% | 合法参数、使用 observation、失败后修正 |
| 基础代码 | 15% | 短函数、测试、纠错 |
| 数学与可验证推理 | 15% | 难度递增、正确 final |
| OCR/文档/表格 | 15% | 字符和数值精确，保持格式 |
| 单图 VQA/关系/图表 | 10% | 回答与图像内容一致 |
| 多图与视频 | 5% | 指代、比较、时间顺序 |

SFT 前 10M assistant CE 可以偏短答案，之后按目标分布训练；视觉塔继续低 LR 更新，不默认全部冻结。保留约 5%～10% 的通用预训练 replay 作为防遗忘实验，其 CE 另行记录，不混入 assistant CE 预算。

### 10.2 direct / thinking 控制

采用明确的控制 token/metadata，而不是只在 system prompt 写一句“请思考”。同一道题可以生成 direct 与短 thinking 两个监督版本，但 split group 必须相同，避免一个版本进 train、另一个版本进 test。

初始 direct 占 70%、thinking 占 30%；thinking 目标主要为 64～256 token 的可验证步骤，复杂题逐步到 512。不要把大模型几千 token 的冗长过程原样当成小模型主食。高 effort 的质量必须相对 direct 有收益，不能只增加长度。

`thinking` 与 `final` loss mask、结束符、生成预算和模板须一致。Demo 显示可选择的模式和预算；若模型没有完成控制能力评测，不宣称“自适应思考”。

### 10.3 真实工具链数据

实现 3～5 个本地可验证工具：calculator、受限 Python、JSON/table query、离线文档检索、图像 crop。优先无外部网络依赖的小环境，结果可重放。

每条轨迹保存模型 action、工具参数、工具实际输出、执行状态和最终答案。模型只能学习 assistant action/final；tool observation 作为条件，不受 assistant CE 监督。必须包含至少两轮工具、参数错误修正、空结果、超时、无法回答等情况。

不能把教师写出来的虚构 tool_result 当成工具真的执行了。SFT 轨迹使用经过环境重放验证的数据；RL reward 来自实际 verifier。工具调用只输出严格解析的结构，用户可见回复不泄漏内部样本路径和训练字段。

### 10.4 SFT 准入与退出

进入 SFT 前，base continuation 应有明显的连贯语言与视觉依赖；如果连基本词句都不稳定，应先回到预训练诊断。SFT 退出至少报告 300 条固定 prompt 的分项结果：通用 100、代码/数学 80、视觉 80、工具 40；覆盖中文，设固定解码与抽样解码两套。

工程候选线：对普通短问答人工盲评的可读比例 ≥90%，角色格式与 EOS 正常率 ≥98%，图像错配对视觉任务有显著负面影响。阈值只是进入更昂贵后训练的最低目标，不能当成公开 benchmark 成绩。

## 11. MTP、draft 与完整推理缓存

### 11.1 从预训练开始保留一个 MTP

本版不把 Kimi 的 AttnRes MTP 或 Qwen 的整个 decoder 直接套入。定义独立 `MF1MTPBlock`：读取主干最终四流 residual `R_t`，对每流归一化后用共享的 `512→512` 投影；下一 token embedding 经另一 `512→512` 投影，广播加入四流；再经过 GR/dense MLA/GR/LatentMoE、最终 GR read，与主 head 共享输出权重。

在 teacher forcing 中，主模型位置 t 预测 `x[t+1]`；MTP 使用 `R_t` 与已给出的 `x[t+1]` 预测 `x[t+2]`。默认 `L = L_main + 0.1 L_mtp + λ_index L_index`，MTP 系数 pilot 比较 0.05/0.1/0.2。各项分别按自己的有效目标数归一化。

MTP 的移位必须在展开视觉后的序列上定义。默认只保留相邻两个未来位置均为可监督文本、同 segment、没有越过媒体/tool/role 边界的目标。不能删除 masked 位置后将剩余文本强行拼在一起 shift，制造不存在的连续预测。

MTP 参与主干梯度；专用 draft 训练时才冻结 target。SFT 保留较小 MTP 系数 0.05；RL/OPD 若不训练 MTP，阶段 manifest 显式记录其状态，不能因为 checkpoint 中有权重就声称 draft 已适配新策略。

### 11.2 专用 draft 训练

主策略 checkpoint 最终冻结后，再以它为 target 训练 draft。初版采用 MTP 派生的单块自回归草稿，候选 draft 长度 2/4/6。与主模型共享 tokenizer/head 的语义，但导出时可以复制小规模必要权重以独立加载，参数计数需去重与非去重两套。

原始 MTP 只见过 teacher-forced target hidden，不能直接宣称它能连续生成多个可靠草稿。专用训练必须加入 target 生成的真实前缀以及 draft 自己生成的错误前缀，进行 2～6 步展开；每步输入目标 hidden 可得性与线上完全一致。若后续步无法取得新的 target hidden，就使用草稿自身状态/固定 target anchor，并在训练中采用同样的规则，不能读取尚未验证的未来 target hidden。

预算 5M～15M rollout token，LR 1e-4 起点，训练 CE/soft-target KL，检查 direct/thinking、tool、图像和视频前缀。target 固定 SHA，数据保存 draft/target 版本、采样参数和前缀来源。训练时用不到多模态时也应喂 target 的多模态前缀状态，不能仅靠纯文本 draft 数据推断视觉场景同样有效。

初版不要求 7 步 LK 或并行 mask draft。它们进入单独研究配置：需要独立写明 hidden taps、循环状态、mask token、训练展开和线上缓存协议后再实现，不能只修改 `draft_steps` 就称为完成对应方法。

### 11.3 采样正确性

先实现 greedy speculative，与逐 token target greedy 输出逐字一致。再实现随机 speculative：接受概率 `min(1,p(token)/q(token))`，拒绝时从规范化的 `max(p-q,0)` 修正分布采样；temperature、top-k/top-p 等变换必须在定义 p/q 时一致，不能只比较 argmax。

验证模型一次处理 draft token 后，KDA state、卷积状态、CSA pending block、QSA KV、index key、位置和媒体 metadata 都可能已经前进。拒绝后必须回到最后接受前缀：

- 初版保存最近 verification chunk 的状态快照；需要时回到快照并重放接受前缀。
- 后续才优化为增量 checkpoint；KDA state 不能像 KV list 一样简单 `truncate(length)`。
- CSA 压缩块如果包含被拒绝 token，必须移除并从 pending 重建；重叠块的后继也可能受影响。
- QSA block index 与 raw KV 一起回滚，tie-breaking 与重放保持稳定。

默认 Demo 在 speculative 未通过分布检验或没有实测加速时使用 target-only。必须报告 acceptance、平均接受长度、prefill/decode 分段延迟、吞吐、显存和端到端收益；高 acceptance 不必然意味着速度提升。

### 11.4 前缀缓存与多模态复用

缓存 key 包含模型/adapter/checkpoint SHA、tokenizer、模板、输入 token、图片内容 hash、预处理配置、grid/time positions、attention phase。不同图像即使 placeholder 相同也不能命中同一缓存。

文本前缀复用必须保存全部三种通路状态。修改系统提示、图片分辨率或 tile 顺序时缓存失效。生成时不反复编码同一媒体；但训练视觉塔更新后不得复用旧 encoder features，除非该阶段明确冻结塔并使用带权重 hash 的特征缓存。

## 12. 可验证 RL、领域教师与 OPD

### 12.1 不用 RL 补基本语言能力

进入 RL 的前提是 SFT 模型已经有非零且可重复的任务成功率。先为每个领域建立难度分桶，使初期 `pass@1` 大致在 10%～70% 的可学习区间；极低成功率时，继续进行有验证答案的 SFT、降低难度或提供短提示。若同组全部答错，标准组内相对优势几乎没有学习信号。

第一轮 unified RL 的范围包括：基础算术、短 Python、结构化输出、本地工具、OCR 短串/数字、可验证图表和合成视觉计数。开放式 caption 的质量暂不交给未经验证的小 reward model；用 SFT 和独立人工检查维持。

### 12.2 GRPO 目标与单卡执行

算法参考 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)，本方案采用长度和难度受控的实现。每个 prompt 初始 `G=4` 个 rollout，必要时试 8；组内奖励标准化形成 advantage，使用 clipped policy ratio 与 reference KL，记录所采用的具体 token/sequence 归约定义。

起始超参：policy LR=5e-7～2e-6，clip epsilon=0.2，KL beta=0.01 起步，自适应范围 0.001～0.05；每次 fresh rollout 只做 1～2 次更新，避免严重 off-policy。初始 response cap=256，之后 512；少量工具/代码任务可到 1024。每个 optimizer step 累计约 32 prompts×4 responses，微批顺序执行。

默认单卡流水线如下；有额外 GPU 时可按第 9.5 节将 rollout 与 learner 分离，保持目标、版本与记录一致：

1. 冻结当轮 policy snapshot，按样本序列或小 batch 生成。
2. CPU 本地 verifier 执行并保存环境记录；移走不需要的 rollout KV。
3. 保存 old logprob、reference logprob 或按小块重算；不同时保留完整旧策略梯度状态。
4. policy 使用带梯度微批更新，按有效回答 token 归一化。
5. 保存 policy version 和完成的 rollout shard，下一轮使用新 snapshot。

reference 可以只加载一份 BF16 权重并分时前向；不得把 teacher/reference 误放进 optimizer。frozen teacher 的 router balancing 和 cache 更新也必须正确区分：允许本次前向构造缓存，不允许改权重、correction bias 或训练统计。

主统一 RL 预算 10M～30M generated tokens；3B 预训练的吞吐不能用于估算它的时长。记录采样、verifier、update 和等待四部分耗时，以及 zero-variance groups、truncation 和有效 update token 比例。

### 12.3 奖励设计

| 任务 | 主要 reward | 次要信号与陷阱 |
|---|---|---|
| 数学 | 规范化 final 与程序/符号 verifier | 对单位、分数和数值容差显式定义；不能匹配答案子串就给分 |
| 代码 | 独立隐藏测试通过率 | 代码解析成功不是任务成功；限制超时、内存、文件与网络 |
| 工具 | 环境终态、参数有效性、答案与结果一致 | format 奖励占比应小，避免无意义调用刷分 |
| OCR | CER/精确匹配，按数字/字符单独记录 | 文本规范化不可删除待评估标点、小数点、正负号 |
| 图表/表格 | 与数据表计算结果核对 | 不把底层答案表暴露给模型；图表单位也计入 |
| 合成视频 | 可验证事件顺序/计数 | 模板族和物体外观必须跨 split 分离 |

建议 success reward 主导，格式项总权重不超过 0.1。长度惩罚只在已正确/达到基本要求时小幅启用；过早惩罚长答案可能让模型学会空回答。裁剪 reward、处理异常、不可验证样本状态均在日志中保存原值和最终值。

组内全对或全错时，不伪造优势。可以重新抽取更合适难度的 prompt，但要记录被过滤比例与难度分布；不能只报告保留下来的高成功组。

### 12.4 领域 × effort 教师

为了保留多教师训练的教育和能力整合价值，设计 4 个领域、2 个模式，共最多 8 个教师槽位：

| 领域 | direct 教师 | thinking 教师 | 训练与资格指标 |
|---|---|---|---|
| 一般约束/工具 | 短答案、结构和工具直达 | 多步工具、检查与纠错 | 真实环境成功率、格式有效率 |
| 代码 | 短函数与直接修复 | 计划、测试、迭代 | 隐藏测试 pass@1、资源开销 |
| 数学 | 直接计算/简短解法 | 逐步推理与核验 | 正确率、effort 收益 |
| 视觉 | OCR/单图事实回答 | 图表、多图、视频推理 | 视觉依赖性与任务正确率 |

所有本地教师从合格的 SFT/RL 公共 checkpoint 复制，各自继续训练 2M～5M generated tokens；最多合计 16M～40M。它们使用同架构、同词表、同模板、同媒体处理。此处 8 个是小模型的项目配置，不是复制 Kimi/DeepSeek 官方教师数量。

每个教师必须在不用于训练的领域验证集上，显著优于公共模型或在同等正确率下更省生成成本，且非目标领域不发生严重退化。没有达到资格的槽位标为 `unqualified`，不进入蒸馏。不能为了凑满八个，把同一个弱模型复制八份。

单卡回退路径依次处理教师；多张 PCIe GPU 可并行训练不同领域/effort 教师，每张卡各自保存完整 policy、必要 reference 和独立 optimizer，不同步教师间的梯度。各任务基于同一合格基础 checkpoint，完成后汇总权重与资格评估。视觉教师可低 LR 更新视觉塔；其余领域先冻结视觉塔并混入少量视觉回放来监测遗忘。教师训练是否冻结某模块必须写入各自 manifest。

### 12.5 全词表 reverse-KL OPD

最终 student 从公共模型初始化，按领域与 effort 采样 prompt，使用 **当前 student** 生成前缀，再调用合格教师在同样输入和前缀上计算条件分布。主目标为：

\[
L_{OPD}=\mathbb{E}_{x,\,y\sim q_{rollout}}
\left[\frac{1}{|y|}\sum_t\sum_{v\in V}q_\theta(v\mid x,y_{<t})
\big(\log q_\theta(v\mid x,y_{<t})-\log p_T(v\mid x,y_{<t})\big)\right].
\]

这是在已采样前缀上优化当前 student 的全词表 `KL(student || teacher)`。前缀离散采样不在这一步求导，teacher logits stop-gradient；student 的概率与 logprob 均参与正确梯度。若另用采样 token 的 log-ratio 做 policy gradient，它是不同估计器，不能在日志中仍称全词表 KL。

实施细节：

- tokenizer 和输出词表索引必须完全相同，校验 SHA 与 vocab mapping。不同分词器的外部旗舰教师用于离线生成并筛选 SFT 数据，不能直接按 token ID 相减 KL。
- student 和 teacher 在同一图像/视频、同一 processor、同一问题前缀上打分；只蒸馏回答位置，mask 掉 user、tool result、视觉特征与 padding。
- 对当前 student batch 只使用指定领域的教师，默认与 student 同卡，按 prompt bucket 执行若干微批后切换。允许其他卡驻留其他教师，但跨卡服务必须通过第 9.5 节的传输成本检查；保存 teacher selection 与 coverage，不默认让八个教师同时向 student 发送全词表分布。
- logits 按时间位置 64～256 token 分块，在每块完整词表上精确 logsumexp；需要按 vocab 分块时做数值稳定的两遍归约，不能分别 softmax 子词表。
- default temperature=1；如用其他温度，明确目标与梯度缩放。top-k-only KL 属于有偏近似，需要独立配置和对照。
- LR=1e-6～5e-6 起点；10M～30M student generated tokens。加入独立 10% 的优质 SFT replay 实验可减少能力漂移，单独计 CE 与权重。
- 跨轮 rollout 陈旧度受控：一个 shard 不超过约 2 次 student updates 后重生成；实际阈值按 KL/重要性比率监控。

本版选择全词表路线是因为 32K 词表和 228M 模型使精确条件分布比较有实现空间；它仍须实测显存与速度。多教师组织方式参考旗舰后训练思路，但目标、教师数和预算属于本方案的适配设计。

### 12.6 外部强教师数据的角色

用户已有或合法可用的旗舰模型可以为小模型生成高质量解释、视觉问答与工具轨迹。此类数据必须记录模型版本、prompt、生成设置和 verifier，不把教师输出默认视为真值。代码/数学/工具优先自动验证，图像答案抽查原图。

外部教师生成不要求在这张 3090 上部署旗舰权重；若采用外部服务，其费用、数据使用范围和生成成本单列。MiniFrontier1.0 的核心训练可以依靠公开数据与本地可验证合成任务完成，不能把“单卡训练”包装成整个项目无需额外数据与算力投入。

## 13. 偏好训练与量化部署

### 13.1 DPO 是针对具体缺陷的可选阶段

如果 OPD 之后仍存在回答冗长、忽略明确指令或视觉幻觉，可以在 matched prompt/media 的 preference pair 上做短 DPO。使用经核验的 chosen/rejected，先准备 20K～50K 高质量 pairs，2M～8M 有效响应位置作为预算范围。

LR=5e-7～2e-6，beta=0.1 起点，1 epoch 上限，reference 固定为当前合格模型。只比较 assistant 响应的条件 logprob，prompt 和图像完全相同；不能用不同问题/不同图像凑 pair。标准序列 logprob 与长度归一化版本是不同目标，应选定后记录。[DPO 论文](https://arxiv.org/abs/2305.18290)

准入依据是目标缺陷的分层评测；退出同时检查代码/数学正确率、OCR、EOS、重复率和通用可读性。如果偏好 reward 改善却基本能力下降，回滚，不继续凑满预算。

### 13.2 BF16 为主版本，量化单独验收

首先交付完整 BF16 reference。228M 权重本身约 456MB 十进制，不是必须依赖激进量化才有 Demo。3090 上使用 MXFP4/FP8 的数值模拟不能自动获得旗舰硬件上的计算加速；部署收益由实际 kernel 与端到端测量决定。

可选路线按顺序验证：

1. BF16 精确基线与缓存优化。
2. weight-only INT8 或 INT4 的 PTQ；group size 64/128 对比，先覆盖 routed expert 与普通 linear。
3. 留下 vision、router、norm、KDA decay/state、GR gate、indexer 和 lm_head 的高精度对照，确定敏感层。
4. 若 PTQ 的质量下降无法接受，做 5M～20M assistant CE 的短 QAT，LR=5e-6 起点，再重新评估。
5. 仅在有对应运行时支持时研究 MXFP4 等格式；fake quant 与实际导出逐层对照，不能只验证 fake quant 模型。

校准数据覆盖文本、代码、数字、OCR、视频、工具和长前缀，按约 512～2048 条代表样本开始。量化后主要任务下降候选不超过 1 个百分点、OCR 字符错误率相对增加不超过 5%，且有实测的内存或速度收益，才发布量化版本。统计不稳定时扩大评估，而非依据两次生成决定。

QAT 改变 target 后，draft 必须重新测 acceptance 与输出分布；必要时重新训练。不能把 BF16 target 上的 draft 数据当成量化 target 的完整适配证据。

## 14. 评估体系：正确性、学习效果与真实收益

### 14.1 模块正确性是第一道门

| 模块 | 必做检查 | 初始容差/验收思路 |
|---|---|---|
| KDA | FP32 recurrent 对 chunk 输出/梯度；短卷积；reset；任意长度 | FP32 小张量先达到约 1e-5 量级；BF16 根据误差分布和 NLL 确认 |
| MLA | 显式 expanded 对 latent 优化；mRoPE；head 维度；gate | dense oracle 对齐，不能只比较 shape |
| CSA | pooling、重叠、partial flush、跨媒体与 segment、prefill/decode | 比较真实成员、可见性、输出和压缩缓存 |
| QSA | block→token 映射、tie、空候选、mask、保护集合 | full-budget 与 dense 相等；固定 top-k 与 reference 相等 |
| GR | read/inject 数值；四流梯度；无二次 residual | 与原语对齐，随机非对称输入 |
| MoE | top-k 权重、no-drop、FP32 accumulate、bias resume | 与逐专家 reference 输出/梯度相等 |
| lookup | hash 确定性、跨段/媒体 reset、训练/推理连续性 | 逐 token 与全序列一致 |
| vision | 图片复制、patch/merge、坐标、tile、时间 patch | 特征数与 metadata 一致；多图不串位 |
| MTP | 两步目标位移、边界 mask、共享参数计数 | 人工小序列核对监督位置 |
| cache/draft | prefill+decode 与 full-forward；拒绝回滚 | greedy 全部 token 一致；随机做统计分布检验 |

容差不能机械套用所有 kernel。对长序列 BF16 允许的数值偏差需同时检查 top logits、NLL 和实际解码；错误随长度单调爆炸、视觉块边界突变、cache 重排不一致均不可接受。

正确性小模型用 2～4 层、低维配置即可快速执行，但正式 16 层规格至少做一轮随机前后向和缓存检查。只测试没有开启 CSA/媒体/GR 的 tiny config 没有覆盖本方案。

### 14.2 学习闭环检查

训练初始先做 32 条文本 +32 条图像样本的可控过拟合，再用独立样本验证没有输入错位；视觉需要同图不同问题、同问题不同图的配对检查。过拟合成功只说明能更新参数和对齐标签。

正式训练每 5M CE 记录轻量验证；每 25M CE 运行 fixed prompt suite 与视觉依赖性检查；阶段结束执行完整分层评测。时间昂贵的评价可以降低频次，但所有正式阶段转换必须基于同一个真实 checkpoint 产物。

指标分层：

- 文本：held-out CE/NLL、byte-normalized loss、重复率、EOS、可读率、指令完成率。
- 代码：可执行率、隐藏测试 pass@1；不把 pass@k 当 pass@1。
- 数学：基础算术/代数/短应用题按难度正确率，direct 与 thinking 分别统计。
- 视觉：OCR CER、数字/编号 exact match、VQA/图表准确率、图像错配差值、幻觉率。
- 视频：时间问题正确率、正常/倒序/首帧差值，按帧数和时长分层。
- 工具：真实任务成功率、参数合法率、轮数、失败恢复率、执行成本。
- 长程：在 1K/2K/4K/8K 中按证据位置测试 exact retrieval、数字复制、多图引用、摘要和跨段推理。
- 系统：CE/s、input/s、generated/s、TTFT、decode latency、峰值显存、各模块耗时。

公开基准可以作为补充；实施时固定官方评测脚本、版本、prompt、解码和 split，检查与训练源是否重叠。对这个量级优先选能够分辨进步的基础题、OCR 和短任务，不能只用旗舰难题的接近零分结果选择全部结构。

### 14.3 防止“loss 在降，但不会说话”再次出现

每次生成评估保存 prompt、渲染后的模板、input IDs、checkpoint/tokenizer SHA、temperature/top-p/max_new_tokens、原始输出、终止原因。保留至少 greedy 和固定 seed sampling 两组。

当生成异常时按顺序排查：

1. checkpoint 是否真的是当前训练权重，`--init`/resume/export 是否指向同一 artifact。
2. tokenizer、特殊 token ID、模板、EOS 和 output head 尺寸是否对应。
3. labels shift、assistant mask、视觉展开、有效 CE 数是否正确。
4. train/eval、full-forward/cache、dense/sparse 输出是否一致。
5. 训练数据是否以噪声、模板、重复或长难题为主。
6. 验证与样本是否显示仍在明显欠训练。
7. 最后再归因于结构容量/压缩设计，结合消融证据调整。

一个 DPO/GRPO 的 step 标签或曲线下降都不是可用性证据。阶段不能以 `step >= target` 单独判定完成。

### 14.4 架构消融设计

所有研究配置保持原生视觉；不以纯文本模型的速度去宣传多模态主模型。采用同一 tokenizer、清洗版本、data order 和评测。

| 编号 | 对照 | 要回答的问题 |
|---|---|---|
| B0 | 16 层全 dense MLA + dense FFN，参数匹配 | 复杂融合相对简单可训练基线是否有价值 |
| B1 | 12 KDA +4 token MLA，GR/LatentMoE 不变 | CSA 压缩历史是否值得替换两层 raw access |
| B2 | 主结构，但 QSA 保持 full-budget | QSA 稀疏选择的质量与真实速度代价 |
| B3 | 主结构关闭 lookup，重分配约 8.8M 参数 | 查表是否胜过普通参数容量 |
| B4 | LatentMoE 改全宽 MoE，调整专家数/中间维 | latent bottleneck 是否伤害视觉/精确任务 |
| B5 | GR 改普通 residual；分别做参数匹配与不匹配版本 | 四流带来的收益是否只是参数更多 |
| B6 | GR 替换成 AttnRes 或 mHC，各独立实验 | 哪种深度信息流更适合 16 层 |
| B7 | QSA top-block 32/64/128；media protect 开/关 | 稀疏预算与视觉细节的关系 |
| B8 | native random vision 对 pretrained-vision 对照 | 原生从零训练的视觉学习是否成为瓶颈 |
| B9 | MTP 系数 0/.05/.1/.2 | 辅助预测是否改善主模型或只增加成本 |

预算分三层：全部候选先做正确性与 2M 机制检查；关键单因素对照做 20M pilot 筛失稳/吞吐；保留最有希望的 2～3 个做 200M，并至少对最终两者跑第二个 seed。长期效果必须在更充分训练后确认，不能凭 20M NLL 微小差异宣布最优架构。

比较同时给出：等主 CE、等训练 GPU 小时、近似等参数三种口径中实际采用哪一种。MoE 总参数相等未必算力相等；视觉分辨率变化也会改变 FLOPs。必要时用 Pareto 图呈现准确率/延迟/显存，而非强行合成一个总分。

## 15. 现有代码复用和必须新增的内容

以下路径以远端项目预期根目录 `/workspace/MiniFrontier` 表示实施位置，**是修改方案，不表示本次已经在这些路径添加文件**。实际操作前重新读取当前代码与项目规范。

| 现有模块 | 可复用内容 | 新模型必须做的改造 |
|---|---|---|
| `/workspace/MiniFrontier/minifrontier/models/minikimik3/upstream_layers.py` | KDA/LatentMoE 数学原语 | 解耦 Kimi decoder/AttnRes；明确 gate、norm、因果输入 |
| `/workspace/MiniFrontier/minifrontier/models/minikimik3/kernels.py` | KDA reference/加速入口 | 三类层共同 cache contract、变长与媒体 segment 验证 |
| `/workspace/MiniFrontier/minifrontier/models/miniqwen4/upstream_core.py` | GR 数值原语 | 统一四流 shape 与 read/inject 所有权 |
| `/workspace/MiniFrontier/minifrontier/models/miniqwen4/qsa.py` | 索引/选块思路 | 接 MLA latent、真实 block metadata、保护媒体与 cache |
| `/workspace/MiniFrontier/minifrontier/models/minideepseekv4/attention.py` | CSA/compressor/局部窗口 | segment/media-aware flush、独立位置、训练/增量一致 |
| `/workspace/MiniFrontier/minifrontier/models/miniqwen4/upstream_vision.py` | ViT/多模态处理参考 | 新宽度/patch/merger、统一 grid 和位置协议 |
| `/workspace/MiniFrontier/minifrontier/models/miniqwen4/upstream_ple.py` | hash/n-gram 原语 | 限定一次浅层注入，跳过媒体，独立 reset 规则 |
| `/workspace/MiniFrontier/minifrontier/training/kimi_quantile_balance.py` | QB 统计更新 | accumulation 粒度统计、新专家数、checkpoint 恢复 |
| `/workspace/MiniFrontier/minifrontier/training/deepseek_opd.py` | OPD 训练接口参考 | 新模型/视觉/control/teacher manifest，全词表精确 KL 验证 |
| 三个模型已有 MTP/draft/cache 模块 | reference 测试与状态管理经验 | 新 GR-MTP、三通路缓存及拒绝回滚，不能仅改类名 |

新模块建议结构：

```text
/workspace/MiniFrontier/minifrontier/models/minifrontier1/
    configuration.py       # model/phase/position/media contracts
    modeling.py            # independent four-stream decoder
    kda.py                 # thin checked adapter
    csa.py                 # media-aware compressor and attention
    qsa_mla.py             # new integration, own dense oracle
    indexer.py             # block registry, KL targets, stable selection
    moe.py                 # normalized latent + full-width shared
    residual.py            # checked GR adapter
    lookup.py              # shallow text-only hash memory
    vision.py              # native small ViT + merger
    processing.py          # tokenizer/media/grid/packing
    cache.py               # all states, snapshots, rollback
    mtp.py                 # GR-aware auxiliary block
    draft.py               # dedicated rollout-adapted draft
    generation.py          # target-only/speculative APIs

/workspace/MiniFrontier/configs/minifrontier1/
    model_228m_native.yaml
    tokenizer.yaml
    data_manifest.yaml
    pilot.yaml
    p0.yaml  p1.yaml  indexer.yaml  p2.yaml  p3.yaml
    sft.yaml  rl.yaml  teachers.yaml  opd.yaml  dpo.yaml
    draft.yaml  quantization.yaml  eval.yaml  demo.yaml
```

必须新增/完成的能力清单：

- 独立模型注册、配置校验、auto mapping 和参数账本。
- QSA-MLA 原语与 dense oracle；CSA 媒体分段、partial flush。
- 统一 GR 残差外壳；删除适配中的隐式重复 residual。
- 新 tokenizer/control protocol、视觉展开后的统一 labels/positions。
- 四类索引器训练/切换状态；阶段内部 warmup 的确切语义。
- 三通路完整 cache、prefix key、batch reorder、speculative snapshot/replay。
- GR-aware MTP 和不读取未来 target hidden 的 draft rollout。
- 模型/数据/阶段/评测 artifact 的 hash 绑定。
- 8 教师槽位的资格评估、顺序训练和精确条件分布 OPD。
- 本地工具实际执行/重放、reward 与视觉验证器。
- 3090 各阶段显存/吞吐测量、精确导出及 Demo 固定样例。

复用现有模块是减少重复劳动；是否可直接复用，须由新接口下的检查决定。文件存在、函数有同名参数和旧模型训练过，都不是兼容性的充分条件。

## 16. 配置、阶段门禁与产物可追溯

### 16.1 配置必须能表达实际结构

以下是**拟议 schema 示例**，尚未实现为可运行配置；字段应按最终代码调整并做 validator，不能直接作为现成启动命令使用。

```yaml
model_type: minifrontier1
model_version: "1.0"
initialization: random_all
execution:
  policy: single_gpu_first_pcie
  default_gpus_per_training_job: 1
  independent_jobs_across_gpus: true
  ddp: benchmark_gated
  tensor_parallel: disabled_by_default
  expert_parallel: disabled_by_default
  context_parallel: disabled_by_default
  pipeline_parallel: disabled_by_default
  sharded_training: memory_need_and_benchmark_gated
  rollout_placement: same_gpu_or_benchmark_gated_separate_gpu
  opd_teacher_placement: same_gpu_preferred
vocab_size: 32768
hidden_size: 512
num_hidden_layers: 16
attention_schedule: [kda, kda, kda, csa4, kda, kda, kda, qsa_mla,
                     kda, kda, kda, csa4, kda, kda, kda, qsa_mla]
residual: {kind: gr, streams: 4, gate_rank: 64}
moe: {experts: 32, top_k: 4, latent_dim: 256, intermediate_size: 256,
      shared_intermediate_size: 768, normalize_before_up: true,
      activation: situ, situ_beta: 4.0, situ_linear_beta: 25.0,
      router_activation: sigmoid, routed_scale: 1.0,
      balance: quantile, token_drop: false}
qsa_mla: {kv_rank: 128, q_rank: 128, heads: 8, qk_content_dim: 64,
          rope_dim: 32, value_dim: 64, block_size: 4, top_blocks: 64,
          window: 128, protected_media_tokens: 1024}
csa: {ratio: 4, heads: 8, kv_dim: 64, q_rank: 128, rope_dim: 16,
      top_compressed: 64, window: 128, media_boundary_flush: true}
vision: {depth: 12, width: 384, heads: 6, mlp_dim: 1536,
         patch_size: 16, temporal_patch: 2, spatial_merge: 2,
         projector_hidden: 512, pretrained_weights: null}
lookup: {layer: 2, orders: [2, 3], hashes_per_order: 2,
         rows_per_table: 32768, head_dim: 64, text_only: true}
mtp: {blocks: 1, share_output_head: true, coefficient: 0.1}
context: {max_train_length: 8192, extrapolation_validated: false}
```

`layer:2` 使用本文一基编号；代码转换为零基只执行一次。schema 必须拒绝 layer schedule 长度不等于 16、stream width 不一致、rope sections 不匹配、媒体预算超过序列余量、专家 top-k 大于专家数等错误。

### 16.2 每一阶段绑定实际 checkpoint

阶段准入文件至少包含：

```json
{
  "stage": "p2",
  "model_config_sha256": "...",
  "tokenizer_sha256": "...",
  "processor_sha256": "...",
  "actual_init_checkpoint_sha256": "...",
  "predecessor_stage": "indexer",
  "predecessor_eval_artifact_sha256": "...",
  "dataset_manifest_sha256": "...",
  "code_commit": "...",
  "working_tree_patch_sha256": "...",
  "optimizer_groups_sha256": "...",
  "attention_phase": "sparse_transition",
  "approved_metrics": {},
  "status": "qualified"
}
```

训练启动时读取**实际 `--init` 指向文件**的 hash 与该文件比较，检查阶段、配置、处理器及评测产物；不能只确认某个名为 `gate_pass.json` 的文件存在。resume 另外校验 sampler、optimizer、scheduler 与累计 token，warm-start 明确允许丢弃哪些状态并记录原因。

同样的约束覆盖 SFT→RL、领域 teacher→OPD、target→draft、BF16→quantized。任一实际文件 hash 不匹配即拒绝启动该阶段，并给出具体字段差异。

### 16.3 产物目录

每次 run 至少产出 `resolved_config、source_manifest、data_manifest、param_report、train_log、metrics、eval_samples、checkpoint_manifest、resume_state、environment、stage_gate`。对于 RL 还需 `rollout_manifest、reward_trace、tool_trace、policy_versions`；OPD 增加 teacher registry；draft 增加 acceptance/rollback 检查。

checkpoint 保存 tensor 权重、必要非参数路由状态、优化器、scheduler、RNG 与数据游标；部署导出可以剥离优化器，但必须保留能追溯训练 run 的 manifest。高效保存可按每 25M CE 一个正式 checkpoint、每 5M 一个恢复点设置，并按磁盘预算保留；不能删除当前阶段唯一可恢复点。

## 17. 3090 成本与实施资源预算

### 17.1 先测吞吐，再承诺时长

3B 主预训练的纯训练耗时公式为：

\[
\text{days}=\frac{3\times10^9}{\text{measured CE tokens/s}\times86400}.
\]

| 假设整段平均有效 CE/s | 3B 所需天数 | 解释 |
|---:|---:|---|
| 250 | 138.9 | kernel/视觉/数据处理效率较差时的情景 |
| 500 | 69.4 | 低吞吐情景 |
| 1000 | 34.7 | 中间情景 |
| 2000 | 17.4 | 高吞吐情景，不能提前假定能达到 |

这是算式情景，不是 MiniFrontier1.0 已测速度，也不是 RTX3090 的性能承诺。实际时间还需增加 evaluation、checkpoint、I 阶段、数据准备、故障恢复与所有后训练。正式预算应用各阶段 `CE_budget / measured_stage_CE_per_sec` 求和，不能用 P0 短序列速度覆盖整个 P3。

若计入多模型对照、8 个教师与 draft，完整研究可能需要多周到数月的单卡等效计算。多张 PCIe 卡可通过并行独立实验/教师缩短墙钟时间，但有依赖关系的训练阶段不能简单并行，不能把总天数机械除以 GPU 数。追求“可在消费卡上完整实践”合理，追求“所有旗舰机制从零在数小时内训练充分”缺乏依据。

### 17.2 后训练成本分开算

一个中等配置示例：R=20M generated，8 个合格教师各 4M=32M，OPD=20M，draft=10M，共 **82M generated tokens**。这些是不同阶段的采样总数，实际还需多次 scoring、verifier 和梯度更新。

如果有效串行采样吞吐是 50 generated/s，仅这 82M 采样就约 19 天；100/s 约 9.5 天。此处同样只是算式，真实吞吐取决于前缀长度、batch、缓存和工具等待。不得把 82M 与 3B CE 相加后统一除以预训练吞吐。

教师槽位不是无条件花完预算：资格筛选前先做较小试训，未获得增益时调整任务而非继续训练同质教师。teacher 数量较少是实验结果，不是隐瞒流程简化；最终发布记录实际合格集合。

本节 82M generated token 示例是单卡等效采样账本。教师可分卡并行，rollout 可使用额外卡；实际墙钟时间按各依赖阶段、GPU 分工、权重刷新和队列等待重新计算，采样 token 总预算与 GPU 小时分别保留。

### 17.3 单卡 profiling 最小矩阵

每种场景先 warmup 20 steps，再测至少 100 optimizer steps 或足够覆盖波动的窗口；记录设备型号、驱动、CUDA、PyTorch、kernel commit、功率限制与是否共享 GPU。

| 场景 | 长度/媒体 | 需要报告 |
|---|---|---|
| dense pretrain | 512/2048，无图与 196-token 图 | CE/s、input/s、reserved/allocated、算子耗时 |
| sparse pretrain | 2048/4096/8192 | index 与 core attention 分开耗时 |
| OCR | 4K/8K +833 visual token | vision encoder、protected-media 额外读取 |
| video | 8K +800 visual token | 解码/预处理 CPU 吞吐与 GPU 等待 |
| SFT | 不同 assistant 有效比例 | CE 分母、padding 浪费与实际 step 成本 |
| RL | G=4，回答 256/512 | rollout、reward、update 三段 |
| OPD | 同词表 full KL | teacher/student/logits 分块峰值 |
| draft | greedy/random、不同草稿长度 | correctness、acceptance、端到端速度 |

如果 CPU 图像/视频预处理成为瓶颈，可缓存重采样媒体或确定性预处理结果；视觉塔仍训练时不能缓存旧模型的 learned features。数据 loader 失败记录重试与跳过样本，不把无限重试造成 GPU 空闲误认为训练挂起。

### 17.4 磁盘与 checkpoint

单份 BF16 权重约 456MB，FP32 约 913MB；含优化器的训练 checkpoint 大小按实际保存格式测量，通常明显更大。8 个教师只保存部署权重也要数 GB；保留各自优化器、rollout、图像和视频后，主要磁盘成本会转向数据和轨迹。

3B token 的 uint16 编码理论上约 6GB，但那只是 token IDs，不含 offsets、labels、metadata、未消费 pool 和媒体；如果词表或特殊 ID 超出 uint16 范围必须改格式。不能据此把整个数据集预算写成 6GB。

建议实施前清点数据、媒体、运行产物、备份和可用空间，基于实际 subset 大小制定下载与保留策略。视频按采样需要缓存片段/帧并保存源定位，避免每个 epoch 反复解码全长文件。

## 18. 实施顺序、交付物与阶段验收

### 18.1 工程实施顺序

| 里程碑 | 完成内容 | 进入下一步的证据 |
|---|---|---|
| M0 设计冻结 | upstream 清单、tokenizer 对照、参数账本、schema | 无未解释尺寸；source/config/hash 固定 |
| M1 正确主干 | GR + KDA + dense MLA + LatentMoE + vision | FP32/梯度/文本图像过拟合；基本生成缓存一致 |
| M2 三通路 | CSA、QSA-MLA、media block registry、保护集合 | dense oracle、indexer 训练、边界与 rollback 检查 |
| M3 辅助模块 | lookup、MTP、训练参数覆盖、QB、resume | 计数、目标位移、精确续训通过 |
| M4 单卡 pilot | 20M 对照，P0 规格性能实测 | 24GB 预算、无持续异常、数据/视觉梯度正常 |
| M5 主预训练 | P0/P1/I/P2/P3 | 各阶段真实 checkpoint 绑定评测；基础语言/视觉改善 |
| M6 SFT | 多轮、工具、direct/thinking、OCR/video | 格式、可读、视觉依赖与任务正确率 |
| M7 RL/teacher/OPD | 可验证环境、合格教师、精确 KL | 真实 reward trace、无能力倒退、student 分项收益 |
| M8 推理与发布包 | target-only、draft、可选量化、Demo | 导出重载一致、缓存、吞吐、固定示例与 model card |

M1 暂时还没有全部目标机制，是实施顺序中的可运行中间态，不能命名为最终融合模型交付。只有 M2/M3 目标模块按配置完整接入后，才进入本方案的正式架构训练。

架构验证应在独立模块/配置下推进，现有三个模型的默认行为保持由各自配置决定。把旧模块的修复与新架构接入区分提交，方便检查差异与回滚；本文不要求当前运行中的旧训练停止或迁移。

### 18.2 主阶段准入表

| 转换 | 必备条件 | 失败后的动作 |
|---|---|---|
| smoke→pilot | 标签/视觉/梯度/因果 reference 通过 | 修实现，不能靠增加数据掩盖 |
| pilot→P0 正式 | 24GB 实测、无遗漏参数、可恢复 | 优化 activation/kernel 或调整确定的尺寸 |
| P0→P1 | 语言 NLL 下降，视觉路径有效，路由无异常 | 查数据与数值；必要时延长 bootstrap |
| P1→I | dense reference 已学到有效注意力与基本续写 | 增加有效预训练，不能蒸馏随机 attention |
| I→P2 | 分层 attention mass 与 dense/sparse 差距合格 | 延长 indexer/提高预算/查映射 |
| P2→P3 | 4K 稳定，OCR 和文本无显著退化 | 修稀疏/视觉数据，再扩长度 |
| P3→SFT | 合理 base continuation，held-out 证据与视觉依赖 | 追加预训练或回看 tokenizer/结构 |
| SFT→RL | 可读与格式达标，任务有非零可学习成功率 | 优先补 SFT/降难度 |
| teacher→OPD | 教师资格、词表/processor 一致 | 淘汰不合格教师，修数据或领域训练 |
| final→draft/QAT | target checkpoint 冻结且分项评测合格 | 先确定主策略，不训练漂移 target 的 draft |
| export→Demo | 重载一致、正确输出、媒体/缓存测试与性能 | 修导出与服务链路，保留 target-only |

### 18.3 Demo 的实际产品形态

一个清楚的小型本地页面即可：文本输入、图片/短视频上传、direct/thinking 控制、生成预算；输出文本，必要时展开工具调用及其实际结果。高级面板可显示模型版本、TTFT、decode tokens/s、输入/视觉 token、是否启用 draft。

内置 12～20 个固定样例：中文解释、英文简答、基础数学、短 Python、数字 OCR、图表、文档局部、多图比较、帧顺序、工具计算与错误恢复。样例来自独立 demo 集，不用于训练或超参选择；同时提供失败案例，避免只选一张记住的图片。

单卡演示默认 batch=1，优先 target-only BF16。上传媒体先显示将使用的分辨率、tile/帧数和上下文占用；超过预算给出明确调整，不能显示“已分析全部视频”但实际只看两帧。生成过程不暴露技术报告/训练文件路径等内部实现信息。

发布物包含权重、tokenizer、processor、配置、训练数据来源概述、分项指标、参数账本、运行依赖锁定、复现实验命令和 model card。命令必须在代码真实存在并跑通后提供，本设计稿不伪造尚未实现的 CLI。

## 19. 最可能失败的地方，以及预先规定的处理

| 风险 | 可观察症状 | 优先处理 |
|---|---|---|
| 多重压缩降低细节容量 | OCR/编号/代码复制差，普通 caption 尚可 | 增 QSA raw budget；比较 B1；检查 latent dim 与视觉分辨率 |
| 专家训练不充分 | 高频领域改善、其他领域差，负载极偏 | 提高领域覆盖；减少专家/提高 shared 容量做对照 |
| 视觉从零学习太难 | text 好、错图几乎不影响回答 | 修图文对齐；提高视觉数据质量/数量；比较预训练视觉对照 |
| GR 与现有 block 尺度冲突 | gate 饱和、激活随层放大、梯度异常 | 检查二次 residual、初始化、缩放与 norm |
| 稀疏 attention 在短序列更慢 | core FLOPs 少，端到端更慢 | profile index/gather；短序列可走数值等价 dense 执行路径 |
| lookup 记忆替代泛化 | train 短语很好、陌生表达/OCR 退化 | B3 参数重分配；关闭无收益 lookup |
| 视觉保护吞噬稀疏收益 | 文档输入大量 KV 读取 | 公开报告代价；优化当前媒体预算，不悄悄删细节 |
| all-random 3B 仍欠训练 | held-out NLL 尚快降，样本仍弱 | 对 5B/10B 扩展做成本评估，先补基础再后训练 |
| RL reward 被投机 | 格式高分、实际任务低成功 | 隐藏测试、环境终态、奖励分项和重放 |
| OPD 教师不强或不匹配 | student 向错误回答集中 | 资格门禁、同词表/媒体校验，取消弱教师 |
| cache 实现错 | full-forward 好、流式输出差 | 三通路状态对齐；拒绝回滚重放；关闭 draft 先修正确性 |
| 数据量虚高 | epochs/steps 很大但 unique 很少 | 以有效 CE、unique media、重复率与来源审计重算 |

如果最终 B1（KDA+token MLA）比三通路更好，应接受结果并把 CSA 作为研究配置；如果 GR/lookup 无收益，也应删去并更新型号和参数账本。MiniFrontier1.0 的价值是形成证据充分的小模型设计，不是把所有模块名字永久写在 README 上。

## 20. 官方参考与版本固定方法

### 20.1 本次已读取的主要公开资料

| 资料 | 用途 |
|---|---|
| [用户分享聊天](https://chatgpt.com/share/6aa11f53-d4ac-83ee-bb8d-b05d414f394d) | 融合方向与三种历史通路；概念讨论，不当作算法验证 |
| [Kimi K3 技术报告 v1](https://arxiv.org/html/2607.24653v1) | KDA/LatentMoE/原生视觉及后训练机制来源 |
| [Kimi K3 官方模型页](https://huggingface.co/moonshotai/Kimi-K3) | 官方发布入口与代码版本定位 |
| [Qwen3.8-Next 架构报告 v1](https://arxiv.org/html/2608.30320v1) | QSA/GR/lookup 的设计背景 |
| [Transformers qwen4_exp 固定版本代码](https://github.com/huggingface/transformers/blob/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) | 可读的实现参考与接口核对 |
| [DeepSeek-V4 技术报告 v1](https://arxiv.org/html/2606.19348v1) | CSA 与压缩历史机制来源 |
| [DeepSeek-V4-Flash 官方模型页](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 官方发布入口 |
| [Engram](https://arxiv.org/abs/2601.07372) | 浅层条件记忆的对照方向 |
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 组相对策略优化来源 |
| [DPO](https://arxiv.org/abs/2305.18290) | 偏好目标来源 |
| [SmolVLM-256M](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 小规模多模态可用性参照；初始化条件不同 |
| [MiniMind-V](https://github.com/jingyaogong/minimind-v) | 教学工程与 Demo 组织参照 |

数据官方页面已在第 7 节逐项链接，包含中英文教育文本、数学、指令、视觉、文档和视频来源。本文没有逐条核验这些数据的真实样本质量，也没有把数据集卡的信息当作清洗结果。

### 20.2 上游版本与本地适配的证据边界

现有三份方案/只读代码中标注的 upstream 包括 Kimi `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`、DeepSeek `60d8d70770c6776ff598c94bb586a859a38244f1` 和 Qwen Transformers 上述固定引用。Kimi/DeepSeek 的某些固定文件网页本次未能被网页工具成功展开，因此相应细节同时依据本地带来源标记的适配代码和公开报告核对；不能把本地文件头的声明等同于逐行复现证明。

正式落地 M0 要获取实际采用的 upstream 文件和许可证，记录完整 revision、每个文件 SHA、原始类/函数、适配差异及验证方法。若固定引用已不可用，使用经验证的新引用并标明变化；不静默切到 main。Qwen 的社区库实现也与其完整内部训练系统不同，必须区分发布推理代码、库适配与本文新增训练逻辑。

本方案中 QSA-MLA、CSA 媒体分段、统一 GR decoder、228M 配置、8 个教师槽位和 3B recipe 都是 **MiniFrontier1.0 的新设计**。官方报告支持各构件的机制来源，不为这些组合、预算和效果背书。

## 附录 A. 参数/预算核算应如何固化

实现后增加独立的只读报告入口，至少进行以下核对：

```text
1. instantiate model from resolved config
2. collect named_parameters with storage identity
3. count unique trainable/frozen parameters by module category
4. inspect optimizer groups: every trainable parameter exactly once
5. assert all module-specific shapes, tying and layer schedule
6. compute theoretical bytes for each dtype/state
7. run one representative forward/backward and record actual peak memory
8. emit machine-readable report bound to config/code hashes
```

主要解析式：

```text
embedding/head = 2 × V × H
routed matrices/layer = E × 3 × latent × intermediate
shared/layer = 3 × H × shared_intermediate
latent down/up/layer = 2 × H × latent
lookup tables = n_orders × n_hashes × n_rows × head_dim
GR read matrices = 2 × streams × H × gate_rank
raw MLA KV/layer = N × (kv_rank + rope_dim) × bytes_per_value
KDA recurrent/layer = heads × key_dim × value_dim × state_bytes
```

CSA 缓存按真实分段完成块数，不能一律取 `floor(N/4)`；MoE active 参数还需 shared、路由与输入/输出投影，不能只报 `total×topk/experts`。图片编码是一次输入成本，不能和每个 decode token 的主干成本混算。

预算自动核对：P0/P1/P2/P3 主 CE 必须等于 3B；视觉 CE 目标等于 630M；文本 CE 目标为 2.37B；P0 无视频份额已重分配；领域百分比各自为 100%；媒体曝光和平均答案长度满足可行性；SFT 与 replay、MTP、index KL 各有独立计数。

## 附录 B. 正式实施前必须得到的十项答案

1. 32K 与 64K 的中文/OCR/代码切分和相同文本成本差多少？
2. 228M 候选实际总参数、active 口径和 3090 各阶段峰值是多少？
3. QSA-MLA 的原始 latent KV 是否完整保存，full-budget 是否等价 dense？
4. CSA 跨媒体边界、partial block、重叠与推测回滚是否正确？
5. 四流 GR 是否有独立学习信号，是否存在多次 residual 或重复归一化？
6. 训练数据实际有多少有效 CE、unique 图片、unique 视频与重复暴露？
7. 随机视觉塔的错误图对照是否证明模型正在看图？
8. 20M/200M 消融是否支持三通路、LatentMoE、GR、lookup 的预算分配？
9. 每个阶段的 gate 是否绑定实际初始化权重，resume 是否恢复全部训练状态？
10. fixed prompts、公开/内部评测和 Demo 是否来自同一可追溯导出 checkpoint？

这些答案是实施过程中的交付物。本文件已经给出完整的设计、数据、训练、工程和验收方案；它不替代尚未进行的模型实现与训练验证。
