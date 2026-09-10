# 首版 base 执行记录

本轮执行依据为 [预训练主计划](../../pretraining-plan.md)。本页记录实际处理与缺口，不替代主计划、运行账本或阶段验收。

## 2026-09-10：首批文本构造

已增加 `python -m minifrontier.data.pretraining`，复用现有 Parquet 范围读取与 corpus 去重构造器。首批候选目标按现有 64K **参考** tokenizer 计数：中文 225M、英文 150M、OpenWebMath 50M、对话 40M。465M 是去重后、划分前的目标，不是已准备的训练 token；代码域仍缺少来源/许可证据，不能用其他域或重复样本补齐。首轮 ≥500M 的去重训练库存尚未完成。

来源版本沿用已核查的中文、FineWeb-Edu、UltraChat 固定提交；[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) 固定为 `fde8ef8de2300f5e778f56261843dab89f230815`。OpenWebMath 的 ODC-BY、Common Crawl 条款和底层页面权利单独记录。每次远程读取记录文件、文件校验值、行组、行号及采样顺序。

新构造器使用按组 train 98.5% / val 0.5% / test 1%；旧默认划分继续可复现。重复文档的别名组也连接到保留文档，避免重复副本被丢弃后，其同源变体跨划分泄漏。对话保留全部轮次，按 continuation 文本处理。

候选数据输出始终为 `formal_admission=false`。完成时输出实际分来源/划分 token 数、清洗/去重计数和训练集抽查样本；抽查、benchmark 污染排除、最终 tokenizer 与媒体仍需独立证据，不能由构造脚本自动认定通过。现有参考 tokenizer 不等于本轮最终 tokenizer 已冻结。

单个文本构造任务的磁盘上限 12 GiB，新分配保留 80 GiB；使用选中分片，不下载全库。运行状态与源文件哈希保存在该数据版本的 `source-audit.json`。数据原文和抽查样本不进入 Git。

验证：11 项相关 CPU 回归通过；Ruff 和相关模块 mypy 通过；已验证固定中文分片可按范围读取并取得 LFS SHA256。没有为此启动 GPU 测速或训练。

首次构造在 4,073 条、5,851,266 个候选参考 token 后中断。固定文件 `4_5/003932.parquet` 的 3,274 行中，一条 WuDao 子来源记录的 score 为 `1.0126953125`，超出现有读取器声明的 0–1 范围；该文件其余分数中位数约 0.817。修复仅在新候选构造入口将此类异常隔离并记录来源位置/原始分数，不裁剪分数或改写旧实验读取语义。首个未准入候选及失败报告保留，新版本重新构造，不将失败进度算作正式数据或训练量。

## MF1 紧凑二进制消费

`minifrontier mf1 encode --compact` 可生成按完整记录切分的二进制 shard，默认每片约 64M input token。训练器按 manifest 自动选择紧凑加载器；旧 JSONL 与参考导出继续可读。

纯 token 数据由 18 bytes/token（ids、labels、三轴位置）降为 uint16 的约 2.125 bytes/token（ids 加监督位图），另有每记录索引、来源/媒体描述和文件 manifest。超过 65535 的 ID 使用 uint32；编码前验证范围，不能溢出。位置、segment 和模态数组按现有模型规则恢复；原始图片/视频仍解码进入可训练视觉塔，没有缓存视觉 embedding。打开至多四个 shard 的 memmap；采样器建立长度/领域桶时直接读取索引，不重新编码或解码媒体。

验证覆盖跨 shard 的文本/图片/视频、token 与监督掩码、三轴位置、合并 batch 后的逐值相同模型输出、数据损坏拒绝、uint32 范围，以及二进制数据上的暂停/精确恢复。针对这些路径的 7 项测试通过。数据格式通过不代表正式媒体或生产吞吐已准入。

完整 CPU 回归为 362 passed / 1 skipped（常规 357 项及其余 slow/distributed 5 项），Ruff、格式检查及 173 个源文件 mypy 通过。数据构造任务已接入原有实验台账，候选参考 token 单列，不能显示成训练 CE 或正式准入。旧 Qwen batch16 作业已完成 20,016,632 CE；旧三模型 batch16 队列已无运行进程，不派生新实验。这些消耗不计入本轮从零主训练。
