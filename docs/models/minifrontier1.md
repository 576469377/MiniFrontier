# MiniFrontier1.0

独立原生多模态融合模型，代码在 `minifrontier/models/minifrontier1`。模型系列版本与 Python 包版本分开管理。

| 配置 | 当前实现 |
|---|---|
| 全模型参数 | 228,235,809，含视觉和一个 MTP；从实例化参数计数 |
| Decoder | 16 层、hidden 512、四流 GR，低秩门 64 |
| Attention | KDA ×12；CSA-4 在第 4/12 层；QSA-MLA 在第 8/16 层 |
| MoE | 每层 32/top-4、latent/intermediate 256；一个 full-width shared FFN，intermediate 768 |
| Lookup | 仅第 2 层前，2/3-gram 各两张 32768×64 表；媒体/控制/独立样本重置 |
| QSA-MLA | q/kv latent 128；content/rope/value 64/32/64；保留全部 raw latent KV |
| CSA | ratio 4、独立双分支重叠池化；短块在边界到来时刷新 |
| 视觉 | 随机 12 层 ViT384，Conv3D(2,16,16)，2×2 merge，1536→512→512 |
| 词表 | 配置候选 32K，独立 24 个控制 token；正式 tokenizer 尚未冻结 |
| 上下文 | 配置目标 8192；尚无 8K 3090 性能/能力验收 |
| 缓存 | KDA state/conv、CSA window/pending/compressed、QSA latent/positions/index，全状态快照与重放 |

QSA 的 top-block、local window 和 protected media 取并集；索引候选先按真实 `complete_at` 和 segment 屏蔽。稀疏读取不代表丢弃历史 latent KV，也不表示 Python reference 已获得加速。CSA 压缩后直接作为 KV，不展开回原 token。

MTP 用最终四流状态与下一 token embedding 预测再下一 token；仅保留同 segment、相邻未来位置均受监督且不跨媒体/控制边界的目标。输出 head 使用主模型权重，不额外计一个词表矩阵。draft 是独立固定目标、使用草稿自身前缀的适应路径，不把 teacher-forced MTP 当成已训练草稿。

已完成 CPU 模块、融合前后向、缓存及恢复测试；正式训练和可用语言、OCR、视频、工具能力均待验收。没有公开可用权重，没有宣称优于三个来源模型。具体证据及未完成项见[实现记录](../audits/minifrontier1-implementation.md)。

各模块的来源版本和校验值见[来源映射](../../configs/minifrontier1/source-map.json)。本项目原创部分采用 Apache-2.0，使用的 Kimi、Qwen 和 DeepSeek 组件分别保留上游许可，详见[第三方说明](../../THIRD_PARTY_NOTICES.md)。
