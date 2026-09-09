# MiniFrontier1.0 使用与训练

本页对应 `minifrontier mf1` 独立入口。完整架构、reference 训练和诊断链路可以执行；正式数据、配方与能力门禁仍须逐项取得证据，详见[验收记录](../audits/minifrontier1-implementation.md)。

## 完全离线的小闭环

```bash
uv sync --locked --extra dev --extra monitoring
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 quickstart --output outputs/mf1-quickstart --device cpu
uv run minifrontier mf1 generate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --prompt 'Color?' --image outputs/mf1-quickstart/data/media/train-32-0.png \
  --max-new-tokens 12 --device cpu
```

该图用于展示输入链路，属于训练数据，不能作为视觉能力评测。生成可能为空或错误。示例生成 72 条训练记录：32 条算术、32 条图像色块、8 条带时间戳的视频；另有独立 val/test/demo 分组。它不下载网络数据，不使用已有视觉权重。输出包含全部通路的四层小配置、约 320 词表、训练日志、检查点、参数分组、路由统计与验证记录。

quickstart 顺序执行 pilot 暂停/恢复、indexer 两步、sparse 两步与 SFT 两步。提前暂停不改变原 update 预算；恢复会拒绝源码、配置、词表、数据或优化器组变化。示例的接受标记是 `diagnostic_only`，不是正式阶段资格。完整 228M 不能用这个微型实验的耗时估算。

## 数据和 tokenizer

原生样本使用方案第 7 节的 `sample_id/source/split_group/messages/media/supervision/provenance` schema。媒体必须在输入 JSONL 所在树内，包含真实 SHA256、尺寸；视频逐帧 hash 与单调 source timestamps 缺失时拒绝。文件名、URL、期望答案与 verifier 不进入模型输入。

```bash
uv run minifrontier mf1 prepare-data --input data/candidate/records.jsonl \
  --source-allowlist data/candidate/allowed-sources.json \
  --output data/mf1-candidate-v1 --max-gib 8
uv run minifrontier mf1 tokenizer --data data/mf1-candidate-v1 --vocab-size 32768
uv run minifrontier mf1 encode --data data/mf1-candidate-v1 \
  --config configs/minifrontier1/model_228m_native.json \
  --output data/mf1-encoded-v1 --max-gib 16
```

以上是准备真实候选数据时使用的接口，`data/candidate` 并非仓库自带数据。allowlist 是 JSON 数组，每项包含 `dataset/revision/license_record/status="admitted"`。未知来源进入隔离记录。准备器在 SQLite 中做精确去重、同媒体/规范化文本的传递分组，再按组切分；**感知/语义近重复、评测污染和逐来源 100/300 条人工抽查仍需额外审计**，不会自动把候选 manifest 标成正式合格。

编码产物分别保存 token IDs、shift 前 labels、三轴 positions、segment/modality/media IDs、媒体 span/grid、CE 和 input 计数、源记录定位与 shard hash。token IDs 在 32K/64K 词表下用 uint16；labels 和 positions 为 int32。编码文件用于审计及后续 loader 优化；当前 reference trainer 直接读取已固定 JSONL 并做原生处理，未声称具备大型分片高吞吐加载器。

文档图片可在对应 media 记录设置 `"representation": "document"`，从原图生成全局缩略图和最多四个裁剪，保存 source box；这些 view 合计消耗媒体 token 预算，但只记一次原图曝光。超过上下文/媒体预算时拒绝，不静默丢掉局部图。该路径尚未取得 OCR 任务能力结果。

正式 tokenizer 比较需要清洗后的同分布训练文本及独立留出样本，不能使用 quickstart 的小 BPE 代替 5–10GB、32K/64K 对比。普通用户输入里与控制 token 同形的字符串会插入 WORD JOINER 转义；代码、数字、空格及缩进不做 NFKC 破坏性规范化。正式 vocab mapping/SHA 未冻结前，不启动主训练。

## 阶段与恢复

```bash
uv run minifrontier mf1 params --output outputs/mf1-param-report.json
uv run minifrontier mf1 recipe
uv run minifrontier mf1 train --phase pilot \
  --config configs/minifrontier1/model_tiny.json \
  --data outputs/mf1-quickstart/data --output outputs/mf1-local-pilot \
  --device cpu --steps 20 --input-batch-tokens 64 --stop-after-updates 10
uv run minifrontier mf1 train --phase pilot \
  --config configs/minifrontier1/model_tiny.json \
  --data outputs/mf1-quickstart/data --output outputs/mf1-local-pilot \
  --device cpu --steps 20 --input-batch-tokens 64 \
  --resume outputs/mf1-local-pilot/checkpoint.pt
```

阶段配方在 `configs/minifrontier1/*.json`，可由 `mf1 recipe` 查看实际代码中的预算。正式入口使用 `--run-kind strategy --token-budget <该阶段预算> --evidence <阶段准入文件>`，要求 Git checkout。准入文件必须绑定原方案 SHA、实际 `--init`、tokenizer、processor、配置、数据清单、源码和评测文件；文件名叫 `gate_pass.json` 不代表有效。未获准的候选数据不能用于正式启动。

P0/P1/P2/P3 共 3B CE，WSD 的主日程累计贯穿四阶段；indexer 阶段暂停主 CE 日程，保留主干 moments，P2 合并 indexer 新状态。P2 前 20M CE 逐 batch 增加 sparse 概率，P3 前 10M CE 增加 8K 概率。正式训练按 CE deficit 采样领域，按阶段长度概率选择可容纳完整样本的 bucket，打包时同时隔离 KDA、lookup、attention、MTP 和 CE。缺少某个领域/长度桶时拒绝运行，不截掉答案。

默认全局累计约 16,384 实际输入 token；reference 路径逐微批处理，microbatch/packing 的最优选择仍须 3090 profiling。主 CE、MTP、index KL 使用各自分母。AdamW 和 Muon 各参数分组可检查；Muon 候选尚未获得胜出结论。

磁盘默认保留 **50 GiB**，保存采用带空间检查的原子替换；旧检查点在新文件写完前保持存在。quickstart 显式设 1 GiB 只为小型本地演示。不要对正式训练沿用该低保留值。当前未建立每阶段多级 checkpoint 自动保留策略；正在运行的旧实验不会被此入口接管。

## 后训练与草稿

```bash
uv run minifrontier mf1 posttrain --phase rl \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --data outputs/mf1-quickstart/data --output outputs/mf1-rl-check \
  --steps 2 --max-tokens 4 --device cpu
uv run minifrontier mf1 posttrain --phase draft \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --data outputs/mf1-quickstart/data --output outputs/mf1-draft-check \
  --steps 2 --max-tokens 2 --device cpu
```

随机/弱模型的 RL group 可能全错；代码记录 zero-variance 并跳过 optimizer update，不制造优势。环境工具复用受限 Python、表格查询和离线搜索等本地执行器，tool observation 被 mask。RL/teacher/OPD/DPO/draft 都是待能力验收的独立路径；少量运行不代表后训练已有效。

正式 RL/teacher 默认每轮 32 prompts × 4 responses，先用同一 policy 生成，再顺序反向；按整轮有效 assistant token 数归一化。诊断默认一个 prompt，可用 `--prompts-per-update` 调整。采样时实际行为 logprob 与每组奖励保存在 `rollouts.jsonl`；正式运行按 generated-token 预算结束，显式 `--steps` 可限制试运行更新数。

`opd` 使用当前学生生成的前缀和同图/视频，按位置块计算完整词表 `KL(student || teacher)`。teacher registry 必须给出领域/模式槽、权重 SHA、资格评测与相同词表/processor；正式 OPD 拒绝不合格教师。`train-teachers` 从同一个基础 checkpoint 依次诊断八槽，没有对应数据的槽记录为不合格。`qualify-teacher` 比较同一领域/模式留出集的候选与基础模型，检查 bootstrap 增益、非目标领域退化、视觉置黑依赖和 EOS；数据未准入或证据不足时保持不合格。阈值和原始分项结果随报告保存。`opd` 不指定 `--teacher-slot` 时按 prompt 域/模式选择已合格教师、同卡分时加载，保留同一个 student optimizer。试训完成不会自动取得资格。

draft 训练固定 target SHA，MTP 副本沿自身采样前缀展开 2–6 步，只用已验证 target anchor，不读未来 target hidden。推测采样复用全状态快照/拒绝重放；尚无 acceptance/速度收益结论，Demo 使用 target-only。

`mf1 generate --draft-checkpoint <draft/checkpoint.pt> --draft-steps 4` 可显式测试绑定目标的草稿；`--temperature 0` 验证贪心输出，`--temperature 1` 使用接受/拒绝采样。导出或量化改变 target SHA 后，旧 draft 会被拒绝，需重新适应和验收。

## 导出和诊断 Demo

```bash
uv run minifrontier mf1 export \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt --output outputs/mf1-export
uv run minifrontier mf1 demo --checkpoint outputs/mf1-export/model.pt \
  --device cpu --port 7861 --allow-unqualified
```

浏览器打开本地 `http://127.0.0.1:7861`。页面支持多图、浏览器采样视频帧、direct/thinking、生成预算和输入用量预览。上传后先确认实际处理的帧数、分辨率与视觉 token；不会承诺分析未输入的完整视频。`--allow-unqualified` 仅开放标明状态的诊断页面，不能作为发布验收。

默认导出保持 FP32 权重数值，剥离优化器。`--dtype bfloat16` 单列精度转换；`--int8 --group-size 64` 提供真实 INT8 expert/shared 权重存储、dequantized matmul reference。它没有 INT8 kernel 加速承诺，量化后需重新评测。INT4、QAT 和正式 BF16/量化发布包当前未选用；没有合格策略权重前，不假造校准和量化能力数据。
