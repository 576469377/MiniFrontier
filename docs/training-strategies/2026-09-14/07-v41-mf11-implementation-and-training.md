# MiniDeepSeek-V4.1 与 MiniFrontier1.1：实现与训练

日期：2026-09-14。两条新实验从随机初始化开始，与正在训练的四个模型分别计量。目标是在单张 3090 上研究 V4.1 的结构和优化方法；完整模型能力需经训练、固定验证和后训练评估后判断。

## 来源与范围

参考 DeepSeek 2026-09-10 发布的 **DeepSeek-V4.1-Flash**，固定官方模型仓库 revision `dba1be0a40aa45a94ad051997016db3960a90277`：

- [发布说明](https://deepseek.com/en/news/deepseek-v4-1-flash/)。
- [技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)：§2 的结构与优化器，§4 的预训练与评估。
- [官方推理源码与配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference)：`model.py`、`engram.py`、`vision.py`。
- [本地源码快照和逐文件校验值](../../../third_party/upstream/deepseek-v4.1-dba1be0/source.json)。代码沿用原 MIT 条款；报告权利单独保留，未下载旗舰权重。

报告 PDF SHA-256：`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`。官方提供推理实现与训练说明，没有公开完整训练器及训练数据。本项目的数据、参数规模、训练目标补充和 CUDA 执行方式均单独记录。

## 结构选择

| 方法 | MiniDeepSeek-V4.1 | MF1.1 |
|---|---|---|
| CED 编码/解码分工 | 编码层压缩历史；解码层的全局 KV 来自最终编码状态，本层 SWA 使用自己的状态 | 暂不采用；现有 KDA 递推状态不能直接替换成跨层 KV |
| CSA2 | Full / Reindex / Reuse 显式共享主 KV、索引键和选中位置；压缩采用非重叠窗口 | 保留已有 CSA / QSA-MLA，维持媒体可见域和缓存契约 |
| 分层候选池 | 实现开关；主干预训练关闭，官方亦在后训练引入 | 暂不采用 |
| Single-Pass mHC | 按前一子层产生的系数混合下一子层输入 | 替换四路 GR，保留四路残差；不叠加两种残差混合 |
| Engram | 压缩 tokenizer ID、n-gram 哈希、多头查表与上下文门控；查表状态进入 checkpoint | 保留原 lookup；先隔离残差与优化器升级的影响 |
| 主干 MTP | 关闭；草稿训练另列 | 关闭；暂不支持旧 MTP 草稿入口 |
| 优化器 | 逐头 Q/K Muon、embedding/head/Engram Sinkhorn、标量 AdamW | 同类更新；保留融合投影的实际矩阵边界 |

V4.1 从随机初始化直接训练稀疏注意力。其索引器完整训练代码没有发布；本地使用停止主干梯度的注意力概率 KL 监督，并分别归一化主 CE、路由平衡和索引器损失。这是训练实现补充，不声称复现未公开的全部算法细节。

MF1.1 保留 dense → indexer → sparse。当前 QSA/CSA 的离散 top-k 不能从 LM 损失获得索引器梯度；删除索引器阶段会让未训练的选择器决定历史访问。新版本沿用原有 KDA、LatentMoE、ViT 与 mRoPE；不把来源模型的每项新机制都叠加进去。融合 KV 投影无法用连续行块单独拆出全部 V 时，按完整矩阵更新；卷积参数明确使用 AdamW。

实现采用可微的 BF16 前向与 FP32 优化器状态。官方部署中的 FP4、Mega-mHC 融合内核和有近似性的有界重放，不作为本项目的训练速度或计算等价承诺。

## 预训练安排

| 新版本 | 首阶段 | 后续主 CE | 独立计量 |
|---|---|---|---|
| MiniDeepSeek-V4.1 | D1：250M 文本 CE，直接 sparse，最长 1024 | D2 500M → D3 1.5B → D4 250M；合计 2.5B | 后续视觉暖身和联合训练单列，不混入文本完成度 |
| MF1.1 | P0：200M CE，20% 视觉 CE；512/1024 长度权重 90%/10% | P1 800M → P2 1.4B → P3 600M；合计 3B | P1 后训练 indexer 40M input；SFT/RL 等另计 |

MiniDeepSeek-V4.1：64K tokenizer，全局 input 目标 32,768，执行 microbatch 8，seed 42。Muon/Adam 峰值 LR 共用 `2.6e-4`；前 12.5M 主 CE warmup，保持到 1.55B，1.55–2.225B cosine 降至 0.1 倍，随后保持。该比例参照报告的 warmup/保持/冷却/末期保持形状，并按本地预算缩小；不照搬官方约 100.6M token 的更新 batch。

MF1.1：32K tokenizer，全局 input 目标 16,384，初始 microbatch 8，seed 42。主干峰值 LR `3e-4`、视觉 `1e-4`；保留 MF1 的 2M warmup、2.4B 后 600M 冷却，以便在相同课程下观察结构与优化器升级。参数量由配置实例化统计，配置见 [MiniDeepSeek-V4.1](../../../configs/minideepseekv41.json) 和 [MF1.1](../../../configs/minifrontier11.json)。

两者采用报告的 momentum 0.95、Muon 更新缩放 0.18；Sinkhorn 使用 11 次交替归一化，近零行阈值 `1e-3`，不做 embedding 权重衰减，MiniDeepSeek-V4.1 的 Engram 表 LR 为主 LR 的 5 倍，MF1.1 保留的 lookup 表使用主 LR。标量 AdamW 使用 `(0.9, 0.95)`、`eps=1e-20`。矩阵及 norm 权重衰减 0.1，bias/gain 不衰减。超参数是待实际结果检验的初始选择，不称为最优配方。

### 视觉阶段

官方 ViT 在主干联合训练前已完成大量图文对比和自回归预训练。不能据此冻结一个随机 ViT。MiniDeepSeek-V4.1 首阶段先训练文本；随后新增原生 2D RoPE、3×3 unshuffle 视觉塔，严格继承文本权重，冻结文本进行视觉与投影暖身，再联合训练。视觉数据、预算和解冻日程在接续前绑定；候选暖身 40M CE、联合 300M CE，不因候选计划而计为已启动。

MF1.1 从 P0 开始训练现有 ViT 和主干；保持已有图像控制 token、视频时间戳、mRoPE 和媒体保护规则。相同输入处理使原 MF1 的数据编码可以复用，但不代表旧 checkpoint 与新 mHC 架构兼容。

## 数据与复现

- 复用已冻结的文本来源、tokenizer、划分和污染排除结果。新实验不把相同语料的重复消费计为新增独立数据。
- MF1.1 数据视图只允许 `model_version`、`mtp_enabled`、`mtp_loss_coef` 三个架构字段不同；其余完整配置必须相同。视图同时绑定原 manifest/配置和新配置，读取原 token、像素与元数据并校验原 hash。
- 数据使用原有各来源许可。沿用维护者对学习项目人工审核的豁免，保留自动清洗、去重、留出和完整性检查。
- 每次运行保存代码提交、配置、命令、seed、tokenizer/data hash、优化器参数组、真实 CE/input 计量、验证和吞吐。与旧模型比较时使用相同验证样本，并说明 tokenizer 与实际监督量的差异。
- 后续数据绑定不改已训练阶段的身份。更换结构、目标或优化器产生新实验，不能伪装成原实验恢复。

## 启动、运行与存储

本次使用本机 GPU 3 和 GPU 5，各一项任务。现有 GPU 0/1/2/4 的训练及其源码副本继续运行。调度复用排他 GPU 队列：先取得锁，再核对实际进程与显存。首阶段显式声明随机初始化；后续阶段必须绑定真实父 checkpoint。

启动前执行新计算公式、因果性、共享 KV 梯度、媒体梯度及恢复的 CPU 回归。首批正式更新同时观察速度和显存；若需定位错误，独立性能检查限制在分钟内，不新增 20M pilot。吞吐用实际 CE/s 与 input/s 分列，microbatch 可在显存约束下调整，全局 input 目标和损失归一化保持不变。

固定验证复用已有库存：常规 1M CE、阶段末 5M CE，按累计主 CE 触发。训练状态和 TensorBoard 使用两个独立新名称；失败尝试保留诊断记录，展示目录只包含当前有效运行。训练 loss 下降不能替代生成与视觉依赖评估。

目录约定：模型实现仅新增 `models/minideepseekv41/`；MF1.1 仍在 `models/minifrontier1/`，共用层与优化器各一份。正文仅此联合方案，入口维护在[预训练主计划](../../pretraining-plan.md)。本地绑定与队列集中在 `outputs/strategy-v41-mf11-v1/`，正式训练沿用 `outputs/strategy-base-pretraining-v1/`。

不复制语料或下载旗舰权重。编码缓存维持 24 GiB 上限；本机至少保留 80 GiB 空闲，正式权重预算调整为 56 GiB。保留最新可恢复 checkpoint 与确需的阶段父产物；只有完成引用核对后才清理中间权重。外部文件传输使用直连，不通过代理。

## 启动记录

两项首阶段均已于 2026-09-14 完成正式优化更新，训练源码为 `95fd1ff`。MiniDeepSeek-V4.1 首次 microbatch 16 在首步反向 OOM，优化更新为 0；改为 8 后重新从随机初始化训练，全局 input 32,768 不变。MF1.1 使用 microbatch 8。固定起点验证的实际监督量分别为 1,000,320 和 1,000,915 CE。

排查发现 mHC 广播归约会展开四路残差的平方维度。当前源码已改为禁用 autocast 的 FP32 矩阵收缩，前向及反向经过官方公式对照；计算公式不变，归约舍入可能略有差异。这项修复尚未进入上述两项冻结运行，后续阶段独立绑定后使用。

## 后续判断

先比较固定验证的分域损失、早期学习速度、路由负载及实际显存/吞吐，判断新版本是否值得继续扩大训练。MF1.1 同时改了残差、优化器和 MTP，属于整体版本比较，不能将差异归因于单一方法；必要的单变量消融在出现具体问题后安排。视觉接续、SFT/RL、层级索引与独立草稿模型依次推进，并分别记录未完成项。
