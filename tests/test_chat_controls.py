import pytest
import torch
from test_miniqwen4 import tiny_config
from test_native_data import tokenizer as tokenizer

from minifrontier.chat_controls import parse_action, resolve, semantic_content
from minifrontier.data import chat_tokens
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM


def test_loaded_template_is_used_automatically_for_inference(tokenizer, monkeypatch):
    from minifrontier.inference import respond

    model = MiniQwen4ForCausalLM(tiny_config(vocab_size=512))
    model.chat_template = "control-v1"
    expected = chat_tokens(
        [dict(role="user", content="Hello")],
        tokenizer,
        generation_prompt=True,
        mode="direct",
        effort="low",
    )[0]

    def generate(_model, inputs, **kwargs):
        assert inputs.tolist() == [expected]
        return torch.cat((inputs, inputs.new_tensor([[17, *tokenizer.encode("Hi").ids, 2]])), 1)

    monkeypatch.setattr("minifrontier.inference.generate_ids", generate)
    assert respond(model, tokenizer, "Hello") == "Hi"


def test_controlled_prefix_matches_sft_and_only_assistant_structure_has_loss(tokenizer):
    turns = [dict(role="user", content="What is 2 plus 3?")]
    prefix, prompt_labels = chat_tokens(
        turns, tokenizer, generation_prompt=True, mode="thinking", effort="max"
    )
    complete, labels = chat_tokens(
        [
            *turns,
            dict(
                role="assistant",
                reasoning="Add two units and three units to get five.",
                content="5",
            ),
        ],
        tokenizer,
        mode="thinking",
        effort="max",
    )
    assert complete[: len(prefix)] == prefix and all(v == -100 for v in prompt_labels)
    assert labels[: len(prefix)] == prompt_labels
    assert labels[len(prefix) :] == complete[len(prefix) :]
    assert complete[len(prefix)] == 15 and 16 in labels and 17 in labels and labels[-1] == 2
    parsed = parse_action(complete[len(prefix) :], tokenizer, require_eos=True)
    assert (
        parsed["kind"] == "final"
        and parsed["text"] == "5"
        and parsed["reasoning"].startswith("Add two")
    )


def test_tool_observation_has_no_action_supervision(tokenizer):
    turns = [
        dict(role="user", content="Look up the red item's value."),
        dict(
            role="assistant",
            content=None,
            tool_calls=[dict(name="lookup", arguments=dict(key="red"))],
        ),
        dict(role="tool", content='{"value": 42}'),
        dict(role="assistant", content="42"),
    ]
    ids, labels = chat_tokens(turns, tokenizer, mode="tool", effort="low")
    tool_start = ids.index(6)
    tool_end = ids.index(2, tool_start)
    assert 19 in ids[tool_start:tool_end]
    assert all(v == -100 for v in labels[tool_start : tool_end + 1])
    assert 18 in labels and 19 not in labels and 12 not in labels
    assert semantic_content(dict(content="same answer")) == semantic_content(
        dict(content="same answer", effort="max")
    )
    with pytest.raises(ValueError, match="actual distinct reasoning"):
        chat_tokens([dict(role="assistant", content="same answer")], tokenizer, mode="thinking")
    assert resolve("thinking-max") == ("thinking", "max")
    with pytest.raises(ValueError, match="disagree"):
        resolve("thinking-max", "low")


def test_unfinished_or_injected_action_structure_is_not_a_tool_command(tokenizer):
    body = tokenizer.encode('{"name":"lookup","arguments":{"key":"red"}}').ids
    assert parse_action([18, *body], tokenizer, require_eos=True)["kind"] == "truncated"
    assert parse_action([19, *body, 2], tokenizer)["kind"] == "invalid"
    assert parse_action([17, 6, *body, 2], tokenizer)["kind"] == "invalid"
