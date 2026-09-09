# 架构与目录

```text
configs/                       唯一模型清单与三份容量配置
minifrontier/
  catalog.py                   校验模型、配置和固定源码 revision
  cli.py                       models / quickstart / doctor / train / generate / demo
  data.py                      来源记录、清洗、划分、tokenizer、mmap 数据
  inference.py                 checkpoint 载入、生成、浏览器 demo
  models/
    factory.py                 三个模型唯一构造入口；无替代主干
    common.py                  输出类型与输入验证
    miniqwen4/                 固定 Qwen 计算、PLE/GDN/QSA、缓存
    minikimik3/                 固定 Kimi decoder + 可反传 MoE / KDA 适配
    minideepseekv4/             固定 DeepSeek block + 函数式压缩注意力
  training/
    train.py                   单卡/DDP、阶段、评估、日志、精确恢复
    runtime.py                 全局数据游标、路由偏置平衡、原子检查点
    miniqwen4_optim.py          Qwen 语义分块 Muon / AdamW
    posttrain.py               DPO / GRPO / MOPD 目标
    rollouts.py                本地可验证任务、on-policy rollout、多教师映射
scripts/
  extract_training_sources.py  可审查的 Kimi / DeepSeek 源码提取规则
  launch_training.py           每模型独立 GPU 组、串行阶段、断点续跑
  training_status.py           持续训练状态
third_party/upstream/          固定原始源码与许可证
```

`upstream_layers.py` 是可读、可打包的源码派生文件；不是运行时下载或动态执行远程代码。提取脚本明确列出移除的推理限制、替换的依赖以及可反传分发的变化。格式化后重新运行源码对照测试。

三模型使用复制式 DDP：每张卡持有完整模型与优化器。分布式进程组不会被 DeepSeek 错当作 tensor parallelism。模型规模应先在单卡容纳，默认单卡独立训练；两卡 DDP 是否提高吞吐需要实测；没有把两卡显存合并成“单卡可训”的说法。

各阶段独立重建模型、DDP 和优化器；`--init` 只接入权重，`--resume` 恢复原阶段的全部训练状态。基础预训练冻结离散索引器；稀疏阶段训练 indexer；SFT/DPO/GRPO/MOPD 保留稀疏选择并冻结 indexer。

三个模型均有原生增量缓存；草稿推理包含接受/拒绝与缓存回滚，范围见 [草稿适应](draft-adaptation.md)。压缩注意力使用 PyTorch 显式张量作为教学后端，长上下文内存/速度不能外推到官方融合内核。

`.venv/`、`data/`、`outputs/`、tokenizer、检查点及构建产物均排除于源码发布。wheel 安装模型/策略 JSON 与清单，但正式策略训练首版要求 Git checkout；本地源码快照与审查报告随 sdist 提供，派生文件自身保留许可。
