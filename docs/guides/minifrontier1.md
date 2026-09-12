# MiniFrontier1.0 使用与训练

本页介绍 `minifrontier mf1` 的数据、训练、评估、推理和导出接口。首次运行从离线示例开始，架构见[模型说明](../models/minifrontier1.md)，正式阶段与进展见[预训练计划](../pretraining-plan.md)。

命令在 Linux 仓库根目录运行，需要 Python 3.11+ 和 uv；安装步骤见[首页](../../README.md#快速开始)。使用 CUDA 时加装 `--extra training`，并选择空闲设备。训练主机应使用独立开发环境，避免 `uv sync` 调整活跃任务的依赖。

## 离线最小示例

```bash
uv sync --locked --extra dev --extra monitoring
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 quickstart --output outputs/mf1-quickstart --device cpu
uv run minifrontier mf1 generate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --prompt 'Color?' --image outputs/mf1-quickstart/data/media/train-32-0.png \
  --max-new-tokens 12 --temperature 0 --device cpu
```

示例离线生成 72 条训练记录：32 条算术、32 条图像色块、8 条带时间戳的视频，另有独立 val/test/demo 分组。模型使用覆盖各模块的四层小配置和约 320 词表，视觉塔随机初始化。上述图片来自训练集，生成可能为空或错误，不用于评定视觉能力。

quickstart 先执行 `pilot`，在第 4 次更新暂停后恢复至第 8 次，再运行 `indexer`、`p2` 和 `sft` 各两步。输出包括模型配置、日志、检查点、参数分组、路由统计和验证报告，标记为 `diagnostic_only`。微型示例的耗时不能外推到 228M 配置。

独立评估使用相同数据与检查点，`--generation` 追加生成任务评分：

```bash
uv run minifrontier mf1 evaluate \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt \
  --data outputs/mf1-quickstart/data --split val --generation \
  --output outputs/mf1-evaluation.json --device cpu
```

## 数据和 tokenizer

每条 JSONL 记录遵循[样本校验器](../../minifrontier/data/minifrontier1.py)：

| 字段 | 要求 |
|---|---|
| `sample_id`、`split_group`、`language`、`domain` | 非空字符串；`split_group` 用于关联样本划分 |
| `source` / `provenance` | 固定 `dataset`、`revision`、`record_id` 及 `license_record` |
| `messages` | 角色及类型化 content 列表；文本用 `text`，图像/视频用 `media_id` 引用 |
| `supervision.type` | `answer_ce` 或 `continuation_ce` |
| `media`（有媒体时） | 媒体 ID、目录内相对路径、尺寸与 SHA256；视频另需逐帧 hash 和递增时间戳 |

每份媒体须在 messages 中恰好引用一次。来源地址、独立评测答案和评分规则仅作元数据保存，不送入模型。实际语料见[数据说明](data-sources.md)。

```bash
uv run minifrontier mf1 prepare-data --input data/candidate/records.jsonl \
  --source-allowlist data/candidate/allowed-sources.json \
  --output data/mf1-candidate-v1 --max-gib 8
uv run minifrontier mf1 tokenizer --data data/mf1-candidate-v1 --vocab-size 32768
uv run minifrontier mf1 encode --data data/mf1-candidate-v1 \
  --config configs/minifrontier1/model_228m_native.json \
  --output data/mf1-encoded-v1 --compact --max-gib 16
```

`data/candidate` 需自行准备。allowlist 为 JSON 数组，每项包含 `dataset`、`revision`、`license_record` 和 `status: "admitted"`，未知来源进入隔离记录。准备器执行精确去重，将同媒体/规范化文本关联成组后划分数据；近重复、评测排除和正式准入由后续流程负责，见[已做的处理](data-sources.md#已做的处理)。

编码保存 token、shift 前标签、三轴位置、segment/modality/media ID、媒体 span/grid 和来源计数。正式训练使用 `--compact`：32K/64K 词表保存为 uint16 IDs 与监督位图，按确定规则恢复位置和样本边界，媒体描述另存稀疏索引。加载器以有界 memmap 读取，图片/视频在线解码并经过可训练视觉塔；旧 JSONL 编码仍可读取。两种格式的计算对照见[执行审计](../audits/training-infrastructure.md)。

文档图片可设置 `"representation": "document"`，生成全局缩略图与最多四个裁剪，并记录 source box。所有 view 合计占用媒体 token 预算，只计一次原图曝光；超过上下文或媒体预算则拒绝样本。该路径尚无 OCR 能力结果。

正式训练冻结 32K tokenizer，quickstart 的小 BPE 只用于示例。词表只从训练划分构建；变更后必须重新编码，不能直接恢复原检查点。与控制 token 同形的用户文本插入 WORD JOINER 转义，代码、数字、空格及缩进不做 NFKC 归一化。

## 跨机器媒体读取

紧凑组件可从远端读取原始媒体，并在训练机器设置有界缓存。先复制编码和审计文件、核对 hash，再配置媒体访问；token、标签和 span 仍由同一份编码提供，视觉塔输出不缓存。

例如，在数据主机仅对本机开放图片目录，再从训练主机建立 SSH 转发：

```bash
# 数据主机：目录内是原始图片，不能指向包含其他资料的工作区
python -m http.server 18390 --bind 127.0.0.1 --directory /path/to/corpus/images
# 训练主机
ssh -o ProxyCommand=none -o ProxyJump=none -NT \
  -L 127.0.0.1:18390:127.0.0.1:18390 user@data-host
```

通过组合接口设置缓存。`uri_prefix` 对应原编码路径，`cache_dir` 相对于组合输出目录；组件顺序、访问配置及原 manifest 校验值写入组合 manifest。

```python
import json
from pathlib import Path
from minifrontier.data.minifrontier1_components import assemble_components
from minifrontier.models.minifrontier1 import MiniFrontier1Config

config = MiniFrontier1Config(
    **json.loads(Path("configs/minifrontier1/model_228m_native.json").read_text())
)
components = ["data/mf1-text", "data/mf1-natural", "data/mf1-documents"]
assemble_components(
    components,
    "data/mf1-combined",
    config,
    media_access={
        components[2]: {
            "base_url": "http://127.0.0.1:18390/",
            "uri_prefix": "images/",
            "cache_dir": "../media-cache-documents",
            "max_bytes": 6 * 1024**3,
            "max_file_bytes": 64 * 1024**2,
            "reserve_bytes": 80 * 1024**3,
        }
    },
)
```

SSH 示例直连数据主机，媒体读取器禁用环境代理。缓存按 SHA256 共用并校验原始字节，随后执行原媒体变换；容量包含索引、临时文件和图片，最多 32,768 个文件。仅驱逐本缓存的训练文件，验证/测试媒体持久保留，应在训练前预取。保留文件填满缓存、校验失败或空间不足时停止新增写入。缓存未命中的媒体仍需原服务和隧道可用。

## 阶段与恢复

```bash
uv run minifrontier mf1 params --output outputs/mf1-param-report.json
uv run minifrontier mf1 recipe
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 train --phase pilot \
  --config outputs/mf1-quickstart/model.json \
  --data outputs/mf1-quickstart/data --output outputs/mf1-local-pilot \
  --device cpu --steps 20 --input-batch-tokens 64 --stop-after-updates 10
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MINIFRONTIER_MIN_FREE_GIB=1 \
  uv run minifrontier mf1 train --phase pilot \
  --config outputs/mf1-quickstart/model.json \
  --data outputs/mf1-quickstart/data --output outputs/mf1-local-pilot \
  --device cpu --steps 20 --input-batch-tokens 64 \
  --resume outputs/mf1-local-pilot/checkpoint.pt
```

`mf1 params` 输出研究配置的模块参数量；实际优化器参数组、学习率与 weight decay 见训练目录的 `optimizer_groups.json`。阶段模板在 `configs/minifrontier1/*.json`，`mf1 recipe` 显示代码中的阶段预算。

正式训练要求 Git checkout，使用 `--run-kind strategy --token-budget <阶段预算> --evidence <准入文件>`。证据绑定方案、前驱检查点、tokenizer、processor、配置、数据与源码。恢复须保持预算和这些身份一致；`--stop-after-updates` 仅暂停，不缩短原预算。当前配方和阶段依赖见[主计划](../pretraining-plan.md)。

P0/P1/P2/P3 合计 3B CE，WSD 主日程贯穿四阶段。indexer 阶段暂停主 CE 日程并保留主干 moments，P2 合并新 indexer 状态；P2 前 20M CE 逐 batch 提高 sparse 概率，P3 前 10M CE 提高 8K 概率。领域采样按 CE 缺额平衡，长度按阶段概率选择可容纳完整样本的 bucket。打包隔离 KDA、lookup、attention、MTP 和 CE；缺少必需领域/长度桶时拒绝运行，不截断答案。

默认每次更新目标为 16,384 个输入 token，`--batch-size 8` 限制每个微批的最大样本行数。训练器先选择完整更新窗口，再按行数和补齐后的 token 容量合批，实际输入可略超目标。调整微批须使用真实长度与媒体分布测量显存、padding 和吞吐，参考[性能报告](../audits/minifrontier1-execution-performance.md)。当前正式配方使用 AdamW；主 CE、MTP 和 index KL 各用自己的分母。

写入默认保留 **50 GiB**，由 `MINIFRONTIER_MIN_FREE_GIB` 调整。检查点采用原子替换，新文件完成前保留旧文件，需预留重叠空间。正式训练的缓存总额、阶段权重保留与轮换见[存储安排](../pretraining-plan.md#resources)；示例的 1 GiB 仅用于微型本地运行。

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

RL、teacher、OPD、DPO 和 draft 为独立后训练路径，目前尚无完整训练结果。弱模型的一组回答可能全错，此时记录 zero-variance 并跳过更新。工具任务使用受限 Python、表格查询和离线搜索等本地执行器，tool observation 不计入损失。

正式 RL/teacher 默认每次更新 32 prompts × 4 responses，用同一策略生成后顺序反向，按整轮有效 assistant token 归一化。诊断默认一个 prompt，可用 `--prompts-per-update` 调整。`rollouts.jsonl` 保存采样 logprob 和分组奖励；正式运行按 generated-token 预算结束，`--steps` 可限制诊断更新数。

`opd` 在学生生成的前缀和相同媒体上，按位置块计算全词表 `KL(student || teacher)`。teacher registry 记录领域/模式槽、权重 SHA、资格评测及相同词表/processor。`train-teachers` 从同一基础检查点依次诊断八槽；`qualify-teacher` 在对应留出集比较候选与基础模型，检查 bootstrap 增益、非目标领域退化、视觉置黑依赖和 EOS，报告保存阈值与分项结果。缺数据或证据不足的教师不能进入正式 OPD。未指定 `--teacher-slot` 时，按 prompt 领域/模式选择合格教师，同卡分时加载并保留学生优化器。

draft 绑定 target SHA，将 MTP 副本沿自身采样前缀展开 2–6 步，只读取已验证的 target anchor。投机采样使用全状态快照与拒绝重放；尚无接受率或速度收益结论，Demo 默认仅用 target。

生成时加 `--draft-checkpoint <draft/checkpoint.pt> --draft-steps 4` 可测试草稿，`--temperature 0` 使用贪心验证，`--temperature 1` 使用接受/拒绝采样。导出或量化改变目标 SHA 后，旧草稿不再匹配，须重新适应与验收。

## 导出和诊断 Demo

```bash
uv run minifrontier mf1 export \
  --checkpoint outputs/mf1-quickstart/sft/checkpoint.pt --output outputs/mf1-export
uv run minifrontier mf1 demo --checkpoint outputs/mf1-export/model.pt \
  --device cpu --port 7861 --allow-unqualified
```

打开 `http://127.0.0.1:7861`。页面支持多图、浏览器采样视频帧、direct/thinking 和生成预算，并预览实际帧数、分辨率与视觉 token 用量。模型只处理这些输入帧。`--allow-unqualified` 开放诊断检查点，页面保留其未通过能力验收的状态。

默认导出保留 FP32 权重并去掉优化器，`--dtype bfloat16` 转换权重精度。`--int8 --group-size 64` 将 expert/shared 线性权重存为 INT8，前向时反量化并调用浮点矩阵乘法，**仅用于 CPU 参考路径；当前 CUDA 分组专家路径与此格式不兼容**。量化后需要重新评估，尚无 INT8 kernel 加速或能力结果。INT4、QAT 及正式 BF16/量化发布包均未选用。
