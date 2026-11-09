import pytest
import torch
from test_miniqwen4 import tiny_config
from test_native_data import tokenizer as tokenizer

from minifrontier.chat_controls import parse_action, resolve, semantic_content
from minifrontier.data import chat_tokens
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM


def test_loaded_template_is_used_automatically_for_inference(tokenizer, monkeypatch):
    from minifrontier.inference.runtime import respond

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

    monkeypatch.setattr("minifrontier.inference.runtime.generate_ids", generate)
    assert respond(model, tokenizer, "Hello") == "Hi"


@pytest.mark.parametrize("model_type", ["MiniDeepSeekV41ForCausalLM", "UnknownCausalLM"])
def test_unsupported_image_model_has_explicit_error(tokenizer, monkeypatch, model_type):
    from minifrontier.inference.runtime import respond

    # No real weights or image decoding are needed to exercise the CLI dispatch.
    model = type(model_type, (torch.nn.Linear,), {})(1, 1)
    monkeypatch.setattr(
        "minifrontier.multimodal.prepare_record",
        lambda *args, **kwargs: pytest.fail("unsupported model reached image processing"),
    )
    with pytest.raises(ValueError, match=f"native image generation.*{model_type}"):
        respond(model, tokenizer, "Describe this image", images=["unused.png"])


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


@pytest.mark.parametrize("controlled", [True, False])
def test_cited_thinking_marker_and_historical_row_reader(tokenizer, controlled):
    settings = {"mode": "direct", "effort": "low"} if controlled else {}
    turns = [dict(role="user", content="Quote the tag."), dict(role="assistant", content="<think>")]
    ids, labels = chat_tokens(turns, tokenizer, **settings)
    assert 15 not in ids
    assert (
        tokenizer.decode([v for v in labels if v not in {-100, 2, 17}], skip_special_tokens=True)
        == "<think>"
    )
    old, _ = chat_tokens(turns, tokenizer, escape_angle_tags=False, **settings)
    assert 15 in old


@pytest.mark.parametrize("template", ["legacy", "control-v1"])
def test_two_turn_public_generation_matches_training_prefix(tokenizer, monkeypatch, template):
    from minifrontier.inference.runtime import respond

    model = MiniQwen4ForCausalLM(tiny_config(vocab_size=512))
    model.chat_template = template
    kwargs = {"mode": "direct", "effort": "low"} if template == "control-v1" else {}
    history = [dict(role="user", content="Remember number 27.")]
    seen = []

    def generate(_model, inputs, **opts):
        seen.append(inputs[0].tolist())
        reply = tokenizer.encode("27").ids
        reply = ([17] if kwargs else []) + reply + [2]
        return torch.cat((inputs, inputs.new_tensor([reply])), 1)

    monkeypatch.setattr("minifrontier.inference.runtime.generate_ids", generate)
    first = respond(model, tokenizer, history[0]["content"])
    history += [dict(role="assistant", content=first), dict(role="user", content="Which number?")]
    assert respond(model, tokenizer, messages=history) == first == "27"
    whole, _ = chat_tokens([*history, dict(role="assistant", content="27")], tokenizer, **kwargs)
    assert seen[1] == chat_tokens(history, tokenizer, generation_prompt=True, **kwargs)[0]
    assert whole[: len(seen[1])] == seen[1]
    assert seen[0] == seen[1][: len(seen[0])] and len(seen[1]) > len(seen[0])


def test_structured_tool_history_reaches_generation(tokenizer, monkeypatch):
    from minifrontier.inference.runtime import respond

    model = MiniQwen4ForCausalLM(tiny_config(vocab_size=512))
    messages = [
        dict(role="user", content="Read the counter"),
        dict(role="assistant", tool_calls=[dict(name="read", arguments={})]),
        dict(role="tool", content="42"),
    ]
    expected = chat_tokens(messages, tokenizer, mode="tool", generation_prompt=True)[0]

    def generate(_model, inputs, **opts):
        assert inputs[0].tolist() == expected and 18 in expected and 19 in expected
        return torch.cat((inputs, inputs.new_tensor([[17, *tokenizer.encode("42").ids, 2]])), 1)

    monkeypatch.setattr("minifrontier.inference.runtime.generate_ids", generate)
    assert respond(model, tokenizer, messages=messages, mode="tool") == "42"
    for options in ({"chat": False}, {"images": ["a.png"]}, {"prompt": "also supplied"}):
        with pytest.raises(ValueError, match="messages requires"):
            respond(model, tokenizer, messages=messages, **options)
