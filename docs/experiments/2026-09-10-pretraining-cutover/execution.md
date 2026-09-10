# 首版 base 执行记录

本轮执行依据为 [预训练主计划](../../pretraining-plan.md)。本页记录实际处理与缺口，不替代主计划、运行账本或阶段验收。

## 2026-09-10：首批文本构造

已增加 `python -m minifrontier.data.pretraining`，复用现有 Parquet 范围读取与 corpus 去重构造器。首批候选目标按现有 64K **参考** tokenizer 计数：中文 225M、英文 150M、OpenWebMath 50M、对话 40M。465M 是去重后、划分前的目标，不是已准备的训练 token；代码域仍缺少来源/许可证据，不能用其他域或重复样本补齐。首轮 ≥500M 的去重训练库存尚未完成。

来源版本沿用已核查的中文、FineWeb-Edu、UltraChat 固定提交；[OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) 固定为 `fde8ef8de2300f5e778f56261843dab89f230815`。OpenWebMath 的 ODC-BY、Common Crawl 条款和底层页面权利单独记录。每次远程读取记录文件、文件校验值、行组、行号及采样顺序。

新构造器使用按组 train 98.5% / val 0.5% / test 1%；旧默认划分继续可复现。重复文档的别名组也连接到保留文档，避免重复副本被丢弃后，其同源变体跨划分泄漏。对话保留全部轮次，按 continuation 文本处理。

候选数据输出始终为 `formal_admission=false`。完成时输出实际分来源/划分 token 数、清洗/去重计数和训练集抽查样本；抽查、benchmark 污染排除、最终 tokenizer 与媒体仍需独立证据，不能由构造脚本自动认定通过。现有参考 tokenizer 不等于本轮最终 tokenizer 已冻结。

单个文本构造任务的磁盘上限 12 GiB，新分配保留 80 GiB；使用选中分片，不下载全库。运行状态与源文件哈希保存在该数据版本的 `source-audit.json`。数据原文和抽查样本不进入 Git。

验证：11 项相关 CPU 回归通过；Ruff 和相关模块 mypy 通过；已验证固定中文分片可按范围读取并取得 LFS SHA256。没有为此启动 GPU 测速或训练。
