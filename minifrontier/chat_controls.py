"""Explicit local control-v1 template shared by SFT, rollout and inference.

Control metadata uses existing frozen token IDs. It does not grant reasoning or
tool capabilities: every enabled mode needs corresponding training and evaluation.
Calls without explicit controls retain the historical template exactly.
"""

import json


def resolve(mode=None, effort=None):
    if any(
        value is not None and (not isinstance(value, str) or not value) for value in (mode, effort)
    ):
        raise ValueError("mode/effort must be nonempty strings when supplied")
    aliases = {
        "non-thinking": ("direct", "low"),
        "thinking-high": ("thinking", "high"),
        "thinking-max": ("thinking", "max"),
    }
    if effort in aliases:
        inferred_mode, effort = aliases[effort]
        if mode is not None and mode != inferred_mode:
            raise ValueError("mode and teacher effort slot disagree")
        mode = inferred_mode
    if mode in aliases:
        mode, inferred_effort = aliases[mode]
        if effort is not None and effort != inferred_effort:
            raise ValueError("mode and effort disagree")
        effort = inferred_effort
    effort = effort or "low"
    mode = mode or ("thinking" if effort in {"high", "max"} else "direct")
    if mode not in {"direct", "thinking", "tool"} or effort not in {"low", "high", "max"}:
        raise ValueError("unknown control-v1 mode or effort")
    return mode, effort


def required_ids(tokenizer):
    values = {
        "<|effort_low|>": 12,
        "<|effort_high|>": 13,
        "<|effort_max|>": 14,
        "<think>": 15,
        "</think>": 16,
        "<|final|>": 17,
        "<|tool_call|>": 18,
        "<|tool_result|>": 19,
    }
    if any(tokenizer.token_to_id(token) != index for token, index in values.items()):
        raise ValueError("control-v1 requires the frozen strategy tokenizer IDs")


def semantic_content(turn):
    content = turn.get("content") or ""
    if turn.get("reasoning"):
        content = turn["reasoning"] + "\n" + content
    for key in ("tool_calls", "tools"):
        if turn.get(key):
            content += "\n" + json.dumps(turn[key], ensure_ascii=False, sort_keys=True)
    return content


def record_template(record):
    return (
        "control-v1"
        if record.get("mode") is not None or record.get("effort") is not None
        else "legacy"
    )


def update_manifest(manifest):
    from collections import Counter

    counts: Counter[str] = Counter()
    for record in manifest["stages"].get("sft", {}).values():
        nodes = (
            (record["text"], record["media"])
            if record.get("format") == "hybrid-native-v2"
            else (record,)
        )
        for node in nodes:
            counts.update(node.get("chat_template_counts", {}))
    manifest["chat_template_counts"] = dict(counts)
    manifest["chat_template"] = (
        next(iter(counts)) if len(counts) == 1 else "mixed" if counts else "legacy"
    )


def encode(messages, tokenizer, *, generation_prompt=False, mode=None, effort=None):
    from minifrontier.data import ROLES

    required_ids(tokenizer)
    mode, effort = resolve(mode, effort)
    effort_id = {"low": 12, "high": 13, "max": 14}[effort]

    def text(value):
        if not isinstance(value, str):
            raise ValueError("message content/reasoning must be text")
        return tokenizer.encode(value, add_special_tokens=False).ids

    control = f"MiniFrontier control-v1: mode={mode}; effort={effort}."
    ids = [1, ROLES["system"], *text(control), 2]
    labels = [-100] * len(ids)
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError("unsupported conversation role")
        prefix = [ROLES[role], *([effort_id] if role == "assistant" else [])]
        ids += prefix
        labels += [-100] * len(prefix)
        content = message.get("content") or ""
        if role == "assistant":
            payload = []
            reasoning = message.get("reasoning")
            if reasoning:
                if mode == "direct":
                    raise ValueError("direct-mode examples cannot contain a reasoning trace")
                payload += [15, *text(reasoning), 16]
            elif mode == "thinking" and not message.get("tool_calls"):
                raise ValueError("thinking examples require an actual distinct reasoning trace")
            if message.get("tool_calls"):
                calls = message["tool_calls"]
                payload += [
                    18,
                    *text(
                        json.dumps(
                            dict(calls=calls, content=content), ensure_ascii=False, sort_keys=True
                        )
                    ),
                ]
            else:
                payload += [17, *text(content)]
            payload += [2]
            labels += payload
        else:
            if message.get("tools"):
                content += "\n" + json.dumps(message["tools"], ensure_ascii=False, sort_keys=True)
            payload = ([19] if role == "tool" else []) + text(content) + [2]
            labels += [-100] * len(payload)
        ids += payload
    if generation_prompt:
        ids += [ROLES["assistant"], effort_id]
        labels += [-100, -100]
    return ids, labels


def parse_action(ids, tokenizer, *, require_eos=False):
    """Parse sampled structure; tool-result and role markers are never environment commands."""
    required_ids(tokenizer)
    ids = list(ids)
    ended = bool(ids and ids[-1] == 2)
    if ended:
        ids.pop()
    if require_eos and not ended:
        return dict(kind="truncated", text="", reasoning="", ended=False)
    reasoning = ""
    if ids and ids[0] == 15:
        try:
            boundary = ids.index(16)
        except ValueError:
            return dict(kind="invalid", text="", reasoning="", ended=ended)
        if any(token < 21 for token in ids[1:boundary]):
            return dict(kind="invalid", text="", reasoning="", ended=ended)
        reasoning = tokenizer.decode(ids[1:boundary], skip_special_tokens=False)
        ids = ids[boundary + 1 :]
    if not ids or ids[0] not in {17, 18}:
        return dict(kind="invalid", text="", reasoning=reasoning, ended=ended)
    kind = "final" if ids.pop(0) == 17 else "tool"
    if any(token < 21 for token in ids):
        return dict(kind="invalid", text="", reasoning=reasoning, ended=ended)
    return dict(
        kind=kind,
        text=tokenizer.decode(ids, skip_special_tokens=False),
        reasoning=reasoning,
        ended=ended,
    )
