"""Structured text histories accepted by the generation entry points."""

import json

from minifrontier.data import ROLES


def add_prompt_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="single prompt text")
    source.add_argument("--messages", type=json.loads, help="JSON array of text chat turns")


def validate_history(prompt, messages, *, chat=True, media=False):
    if prompt is not None or not chat or media:
        raise ValueError("messages requires a text chat without a separate prompt or media")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    for turn in messages:
        if (
            not isinstance(turn, dict)
            or not isinstance(turn.get("role"), str)
            or turn["role"] not in ROLES
        ):
            raise ValueError("each message needs a supported role")
        if turn.keys() - {"role", "content", "reasoning", "tool_calls", "tools", "channel"}:
            raise ValueError("unsupported field in text chat history")
        if "channel" in turn and turn["role"] != "assistant":
            raise ValueError("only assistant history may specify a channel")
        if not isinstance(turn.get("content"), str) and not (
            turn["role"] == "assistant" and turn.get("content") is None and turn.get("tool_calls")
        ):
            raise ValueError("each message needs text content")
    if messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("the last message must be a user question or a tool result")
