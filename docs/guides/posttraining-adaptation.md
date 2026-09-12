# 三个来源模型的后训练接口

本页说明 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4 的后训练数据格式、媒体输入、工具执行及恢复行为。完整 SFT、领域教师和强化学习训练尚未完成，当前进展见[预训练计划](../pretraining-plan.md)。MF1 使用[独立接口](minifrontier1.md#后训练与草稿)。

## 共享模板

新 SFT 记录显式设置 `mode` 和 `effort`，由 `chat_tokens` / `prepare_record` 编码。固定 token ID 为：12/13/14 对应 low/high/max，15/16 为 think 边界，17 为 final，18/19 为 tool call/result。CE 只监督 assistant 的 reasoning、输出结构、正文和 EOS；角色、effort、user/system、媒体和工具观察均被 mask。

```json
{
  "stage": "sft",
  "mode": "thinking",
  "effort": "high",
  "turns": [
    {"role": "user", "content": "2 加 3 是多少？"},
    {"role": "assistant", "reasoning": "从 2 起再加 3，得到 5。", "content": "5"}
  ]
}
```

`direct` 不接收 reasoning；`thinking` 的最终答案要求实际 reasoning。另支持 `non-thinking`、`thinking-high`、`thinking-max` 别名。去重纳入 reasoning 和 tool_calls，单纯更换标签不会产生新样本，同一首问最多保留三个不同答案。

未设控制字段的历史数据沿用旧模板。编码 manifest 标记 `legacy`、`control-v1` 或 `mixed`，正式 SFT/RL 拒绝混用。检查点及导出权重保存模板；新模板推理默认 direct/low，可用 `--mode thinking --effort high` 切换，实际效果需要分别训练和验收。

## 原生多模态 rollout

RL 数据目录包含 `train.jsonl`、`val.jsonl` 和标记 `chat_template: "control-v1"` 的 manifest。每条记录提供 prompt 或 turns、domain、effort、verifier；媒体提供原始资源引用与解码 RGB 哈希。`--rl-max-media-features` 默认 64，图像/视频按对应 processor 完整编码。prompt 上限为 `sequence_length - rollout_tokens`，超长样本直接拒绝。

同一 prompt 采样 G 条回答时同步复制媒体索引，学生、reference 和对应 teacher 使用相同图像与轨迹；选择教师子集时重排索引。视觉教师必须来自同一模型家族，使用相同视觉配置和模板；OPD/MOPD 另要求相同 QAT 配方。教师权重须独立通过资格评估，测试中的微型教师仅验证接口。

## 本地工具任务

支持 `lookup`、`search`、`read_file`、`replace_text`、`set_value` 和 `python`，操作范围限任务自身的模拟状态。数据库、文件与键值状态在内存中隔离；Python 由 seccomp worker 执行，禁止读文件、联网和创建进程。工具 schema 加入 user/system 上下文，标准答案仅供 verifier 使用。

```json
{
  "id": "finish-task-001",
  "prompt": "将任务状态 done 设为 true，然后回答 done。",
  "domain": "tools",
  "mode": "tool",
  "effort": "low",
  "environment": {
    "tools": ["set_value"],
    "state": {"done": false},
    "writable_keys": ["done"],
    "max_calls": 4
  },
  "verifier": {
    "kind": "tool_state",
    "state": {"done": true},
    "answer": {"kind": "exact_text", "answer": "done"}
  }
}
```

assistant 在 `<|tool_call|>` 后生成严格 JSON 和 EOS，例如 `{"calls":[{"name":"set_value","arguments":{"key":"done","value":true}}]}`；也支持 `type: "function"` 与嵌套 `function` 对象。每轮最多四个调用，整条轨迹最多八个；初始/终态上限 256 KiB，单工具结果上限 8192 字节。文件替换要求路径在允许集合内且仅命中一次，失败不保留部分修改。

工具返回后开始下一 assistant turn，观察及角色前缀的行为概率占位为零、标签为 `-100`。多轮共用 response token 预算，不人为补 EOS。终止原因包括 final、truncated、invalid、length、context_limit；工具报错单独记录，并使严格终态奖励失败。中间状态正确但没有完成 final，仍不算成功。

verifier 支持整数、精确文本、严格 JSON、小函数测试、带单位换算/容差的 quantity 和归一化 box 坐标误差。按任务指定的规则评分，不从自由文本中默认抽取第一个数字。

## 训练记录与恢复

更新前将完整训练/验证轨迹写入 `trajectories/`，保存策略参数 SHA256、源码/tokenizer/teacher、任务与媒体身份，以及输入、动作掩码、采样 logprob、工具输出、奖励分项和终止原因。文件按内容寻址且不可变。恢复后用恢复的策略与 RNG 重新采样，旧轨迹仅作记录。

轨迹与检查点共用磁盘锁和默认 50 GiB 保留线，长期运行须计入轨迹空间。RL 记录有效输入、response、图像、帧及 feature 数，零优势窗口也记录数据曝光。GRPO/MOPD 按有效 response 平均损失，DeepSeek OPD 按实际 response 位置平均，验证沿用对应分母。全零优势时跳过优化器与路由状态更新。

采样沿用调用者精度，更新前检查冻结策略的概率比偏差，上限为 CPU FP32 `2e-5`、CUDA FP32 `0.001`、低精度 `0.02`，超限则停止。CUDA 单列阈值是因为 KDA chunk/recurrent 的累积方式不同。微型视觉测量及失败记录见[精度检查](../audits/native-rollout-precision-v2.json)，长上下文需独立核验。

## 范围限制

当前同步处理完整短轨迹，未实现长任务 partial rollout、过期策略修正、真实仓库修复或工具动态返回图片。CLI 支持重复 `--image` 输入多图，训练端支持原生视频 processor；来源模型浏览器媒体上传尚未接入。九/十二教师、QAT 的 5M 校准、正式 RL 预算、人工奖励审查及完整视觉课程均待完成。
