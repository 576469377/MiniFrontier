import copy
import json
from pathlib import Path

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_native_data import tokenizer as tokenizer

from minifrontier.models.factory import configure_posttraining
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.training.distributions import forbidden_actions
from minifrontier.training.posttrain import token_log_probs
from minifrontier.training.rollouts import RolloutObjective, TaskDataset
from minifrontier.training.teachers import state_hash
from minifrontier.training.tool_environment import ToolEnvironment
from minifrontier.training.trajectory_log import persist
from minifrontier.training.verifiers import reward


def call(name, **arguments):
    return json.dumps(dict(calls=[dict(name=name, arguments=arguments)]))


def test_resettable_tools_and_strict_state_rewards(tmp_path):
    specification = dict(
        tools=["lookup", "read_file", "replace_text", "set_value", "search"],
        files={"app.py": "return 1"},
        writable_files=["app.py"],
        state={"done": False},
        writable_keys=["done"],
        tables={"stock": [{"name": "apple", "count": 3}, {"name": "pear", "count": 8}]},
        documents=[dict(id="one", text="apples are red"), dict(id="two", text="pears are green")],
    )
    env = ToolEnvironment(specification)
    assert env.execute(call("lookup", table="stock", where={"name": "pear"}, columns=["count"]))[0][
        "output"
    ] == [{"count": 8}]
    assert env.execute(call("search", query="red"))[0]["output"][0]["id"] == "one"
    env.execute(call("replace_text", path="app.py", old="1", new="2"))
    env.execute(call("set_value", key="done", value=True))
    verifier = dict(
        kind="tool_state",
        files={"app.py": "return 2"},
        state={"done": True},
        answer=dict(kind="exact_text", answer="done"),
    )
    assert reward("done", verifier, environment=env) == 1
    assert reward("done", verifier, environment=ToolEnvironment(specification)) == 0
    assert specification["files"]["app.py"] == "return 1"
    assert not (tmp_path / "app.py").exists()
    for path in ("/etc/passwd", "../app.py", "a/../app.py"):
        assert "error" in env.execute(call("read_file", path=path))[0]
    assert reward("done", verifier, environment=env) == 0
    with pytest.raises(ValueError, match="relative paths"):
        ToolEnvironment(dict(tools=["read_file"], files={"/etc/passwd": "x"}))


def test_isolated_python_tool_never_receives_expected_answers():
    env = ToolEnvironment(dict(tools=["python"]))
    result = env.execute(
        call(
            "python",
            code="def add(a,b):\n return a+b",
            entry_point="add",
            cases=[dict(args=[2, 3])],
        )
    )
    assert result[0]["output"] == [5]
    result = env.execute(
        call(
            "python",
            code="def add(a,b):\n return a+b",
            entry_point="add",
            cases=[dict(args=[2, 3], expected=5)],
        )
    )
    assert "error" in result[0]


def test_tool_turns_mask_observations_and_keep_true_behavior_logprobs(
    tmp_path, tokenizer, monkeypatch
):
    torch.manual_seed(313)
    model = MiniQwen4ForCausalLM(tiny_config(vocab_size=512, max_position_embeddings=1024))
    configure_posttraining(model)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    row = dict(
        id="set-task",
        prompt="Set done to true then answer done.",
        domain="tools",
        mode="tool",
        effort="low",
        environment=dict(tools=["set_value"], state={"done": False}, writable_keys=["done"]),
        verifier=dict(
            kind="tool_state", state={"done": True}, answer=dict(kind="exact_text", answer="done")
        ),
    )
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(chat_template="control-v1")))
    dataset = TaskDataset(tmp_path, "train", tokenizer, 640)
    scripted = iter(
        [
            [18, *tokenizer.encode(call("set_value", key="done", value=True)).ids, 2],
            [17, *tokenizer.encode("done").ids, 2],
            [18, *tokenizer.encode(call("set_value", key="done", value=False)).ids, 2],
            [17, *tokenizer.encode("done").ids, 2],
        ]
    )

    # The scripted actions make state success/failure deterministic; probabilities
    # are scored by the actual frozen model on every complete prefix, including tools.
    def sample(policy, prompt, **kwargs):
        actions = next(scripted)
        assert len(actions) <= kwargs["max_new_tokens"]
        sequence = torch.cat([prompt, prompt.new_tensor([actions])], dim=1)
        labels = sequence.clone()
        labels[:, : prompt.shape[1]] = -100
        logp, mask = token_log_probs(
            policy(sequence).logits,
            labels,
            actions=True,
            forbidden_ids=forbidden_actions(policy),
            vocab_size=tokenizer.get_vocab_size(),
        )
        return sequence, logp[:, prompt.shape[1] - 1 :], mask[:, prompt.shape[1] - 1 :]

    monkeypatch.setattr("minifrontier.training.tool_rollouts.generate_ids", sample)
    objective = RolloutObjective(
        model, reference, tokenizer, dataset, group_size=2, max_new_tokens=256
    )
    x, y = dataset[0]
    prepared = objective.prepare(x[None], y[None])
    assert model.training and prepared["rewards"] == [1.0, 0.0]
    assert prepared["terminations"] == ["final", "final"]
    for index in range(2):
        observation = (prepared["inputs"][index] == 19).nonzero().item()
        end = observation + prepared["inputs"][index, observation:].tolist().index(2)
        assert prepared["labels"][index, observation - 1 : end + 3].eq(-100).all()
        assert prepared["old_logp"][index, observation - 2 : end + 2].eq(0).all()
        assert prepared["labels"][index].eq(2).sum() == 2  # Both actual assistant EOS actions.
    loss, _, _ = objective(model, x[None], y[None], prepared=prepared)
    assert objective.last_ratio_error < 2e-5
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    snapshot = state_hash(model.state_dict())
    record = persist(
        tmp_path / "traces",
        prepared,
        dataset,
        policy_hash=snapshot,
        version=dict(split="train", step=0, rank=0),
        run_spec={},
    )
    rows = [json.loads(line) for line in Path(record["path"]).read_text().splitlines()]
    assert rows[0]["policy_sha256"] == snapshot
    assert rows[0]["trace"]["turns"][0]["observations"][0]["output"] == {"changed": True}
    assert rows[0]["reward_components"]["verified_task"] == 1
    assert (
        persist(
            tmp_path / "traces",
            prepared,
            dataset,
            policy_hash=snapshot,
            version=dict(split="train", step=0, rank=0),
            run_spec={},
        )
        == record
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    assert state_hash(model.state_dict()) != snapshot


def test_quantity_and_location_require_structured_units_and_coordinates():
    spec = dict(
        kind="quantity",
        value=2.5,
        unit="m",
        unit_factors={"m": 1, "cm": 0.01},
        absolute_tolerance=0.001,
    )
    assert reward('{"value":250,"unit":"cm"}', spec) == 1
    assert reward('{"value":250,"unit":"kg"}', spec) == 0
    assert reward("first 250, answer 2.5 m", spec) == 0
    assert reward('{"value":NaN,"unit":"m"}', spec) == 0
    box = dict(kind="box", expected=[0.1, 0.2, 0.3, 0.4], max_coordinate_error=0.02)
    assert reward("[.1,.2,.3,.4]", box) == 0  # Invalid JSON is not repaired by regex.
    assert reward("[0.11,0.2,0.3,0.4]", box) == 1
    assert reward("[0.11,0.2,0.7,0.4]", box) == 0


def test_behavior_distribution_guard_fails_before_update(tmp_path, tokenizer, monkeypatch):
    from minifrontier.training import rollouts

    model = MiniQwen4ForCausalLM(tiny_config(vocab_size=512))
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    (tmp_path / "train.jsonl").write_text(
        json.dumps(dict(prompt="Calculate 2+3", domain="arithmetic", effort="low", answer="5"))
        + "\n"
    )
    dataset = TaskDataset(tmp_path, "train", tokenizer, 100)
    objective = RolloutObjective(
        model, reference, tokenizer, dataset, group_size=2, max_new_tokens=2
    )
    original = rollouts.generate_ids

    def wrong_behavior(*args, **kwargs):
        sequence, logp, mask = original(*args, **kwargs)
        return sequence, logp + 0.1, mask

    monkeypatch.setattr(rollouts, "generate_ids", wrong_behavior)
    x, y = dataset[0]
    with pytest.raises(FloatingPointError, match="frozen rollout ratio"):
        objective.prepare(x[None], y[None])
    assert model.training and all(p.grad is None for p in model.parameters())
