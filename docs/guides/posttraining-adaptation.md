# 三个来源模型的后训练接口

本页介绍 MiniKimi-K3、MiniQwen4 和 MiniDeepSeek-V4 的对话模板、图像/视频输入、工具任务及训练记录。接口已完成工程测试，完整监督微调、领域教师和强化学习训练仍待推进。MiniFrontier1.0 的命令见[融合模型指南](minifrontier1.md)。

## 共享模板

新 SFT 记录显式设置 `mode` 与 `effort`，经 `chat_tokens` / `prepare_record`
统一编码。固定词表不变：12/13/14 是 low/high/max，15/16 是 think 边界，
17 是 final，18 是 tool call，19 是 tool result。角色和 effort 提示没有 CE；
assistant 的真实 reasoning、输出结构、正文和 EOS 有 CE。工具观察、图像和
user/system 没有 CE。没有显式控制字段的历史数据保留旧模板。

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

`direct` 不接收 reasoning；`thinking` 的最终答案必须有实际 reasoning，不能只改
effort 标签。支持 `non-thinking`、`thinking-high`、`thinking-max` 别名。
去重身份包含实际 reasoning/tool_calls，单纯换标签仍会去重。一个首问最多三个
不同答案的限制仍在。模式是否有用必须分别训练并验收，模板本身不提供推理能力。
编码 manifest 记录 legacy/control-v1/mixed；正式 SFT/RL 拒绝混用。

checkpoint、best-model 和最终导出保留模板；推理默认按导出模板编码。
新模板默认 direct/low，CLI 可显式 `--mode thinking --effort high`。

## 原生多模态 rollout

RL 目录包括 `train.jsonl`、`val.jsonl` 和包含 `chat_template: "control-v1"`
的 manifest。每行有 prompt 或 turns、domain、effort、明确 verifier；媒体有原始
资源引用及解码 RGB 哈希。`--rl-max-media-features` 默认 64，整个图像/视频按各家
processor 编码。prompt 预算是 `sequence_length - rollout_tokens`；超长样本拒绝，
不会截图或删除依赖图像后继续奖励。

同一输入复制 G 份后，媒体 batch 索引同步复制；学生、reference、对应 teacher
均看到相同图像和轨迹。教师子集排序后重排媒体索引。视觉教师要求同模型家族、
同原生视觉配置、同模板；OPD/MOPD 要求相同 QAT 配方。注册器仍要求独立且合格
的权重，代码测试中的小型测试教师没有真实教师资格。

## 本地工具任务

任务只操作自己的可重复模拟状态：`lookup`、`search`、`read_file`、
`replace_text`、`set_value`、`python`。数据库、文件和键值状态在内存中隔离；
Python 使用现有 seccomp worker，不能读文件、联网或创建进程。工具 schema
自动加入任务的 user/system 上下文，标准答案只留在 verifier。

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

assistant 采样 `<|tool_call|>` 后输出严格 JSON：
`{"calls":[{"name":"set_value","arguments":{"key":"done","value":true}}]}`，
并真实生成 EOS。也支持 `{type:"function",function:{name,arguments}}` 形式。
每轮最多四个调用，整条最多八个；初始/终态最多 256 KiB，单工具结果最多 8192
字节。文件替换必须命中一次且路径在允许集合内；失败不会留下该次半完成修改。
工具返回再接入下一 assistant turn，观察和角色前缀的行为概率占位为零、损失为
`-100`。多轮共享总 response token 上限；没有人为补 EOS，也不复用旧策略轨迹。

终止原因包括 final、truncated、invalid、length、context_limit；工具报错单独
记录并使严格终态奖励失败。中间状态正确但未完成 final 不算任务成功。支持整数、
精确文本、严格 JSON、小函数测试，以及显式单位换算/容差的 quantity 和归一化
box 坐标误差奖励，不从自由文本随意抽取第一个数字。

## 账本与可恢复性

训练与验证的每条完整轨迹在更新前写入 `trajectories/`，包括实际策略参数 SHA256、
源码/tokenizer/teacher 身份、任务与媒体身份、完整输入、动作掩码、真实采样 logprob、
工具输出、奖励拆解和终止原因。文件名按内容寻址且不可变；恢复时重新用当前
恢复的策略和 RNG 采样，不把磁盘旧轨迹当成当前 on-policy 数据。

轨迹与 checkpoint 一样遵守跨进程磁盘写入锁及默认 50 GiB 剩余下限；哈希分块
搬到 CPU，避免额外复制整个模型。正式长跑需把轨迹容量计入磁盘估算。
RL 记录实际有效输入、response、图像/帧/feature 数，即使零优势窗口也记录暴露量。
GRPO/MOPD 的损失按有效 response 平均，DeepSeek OPD 按实际 response 位置平均；
验证采用对应分母。全零优势不做 weight decay 或路由状态更新。

行为采样沿用调用者精度，避免 GPU 采样强制 BF16 却用 FP32 重算。每批在更新前
检查冻结策略概率比；显式本地最大偏差为 CPU FP32 2e-5、CUDA FP32 0.001、低精度
0.02，超过即失败。KDA 的 CUDA chunk/recurrent 累积不同，因此单列 CUDA 上限。
微型原生视觉测试中 BF16 最大约 0.00147；这不是正式上下文长度的通过证明。
完整测量及旧失败日志出处见 `docs/audits/native-rollout-precision-v2.json`。

## 范围限制

当前是同步完整短轨迹实现，未接入长任务 partial rollout/过期策略修正、真实软件
仓库修复或动态工具返回图片。CLI 支持 `--image` 重复传入多图，训练端使用原生
视频 processor；浏览器媒体上传、按已验收模式展示，以及完整视觉课程仍待完成。
九/十二教师、QAT 的 5M 校准、正式 RL token 预算与人工奖励审查均尚未完成。
