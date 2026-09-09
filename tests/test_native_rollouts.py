import copy
import json

import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config
from test_native_data import tokenizer as tokenizer
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.data.media_hash import decoded_hashes
from minifrontier.models.factory import configure_posttraining
from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.training.rollouts import RolloutObjective, TaskDataset


class ResidentTestTeachers(dict):
    def unload(self):
        pass  # Small CPU-only fixtures; the actual registry loads/unloads one GPU teacher.


@pytest.mark.parametrize(
    "device,amp",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.cuda),
        pytest.param("cuda", True, marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "family,method",
    [
        ("kimi", "grpo"),
        ("qwen", "grpo"),
        ("deepseek", "grpo"),
        ("kimi", "mopd"),
        ("deepseek", "opd"),
    ],
)
def test_media_survives_rollout_teacher_subset_and_policy_recomputation(
    tmp_path, tokenizer, monkeypatch, family, method, device, amp, record_property
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda":
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(617)
    common = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32)
    if family == "kimi":
        model = MiniKimiK3ForCausalLM(
            tiny_kimi(vocab_size=512, vision_config=KimiVisionConfig(qkv_hidden_size=48, **common))
        )
    elif family == "qwen":
        model = MiniQwen4ForCausalLM(
            tiny_config(
                vocab_size=512,
                hidden_size=32,
                max_position_embeddings=256,
                vision_config=QwenVisionConfig(**common),
            )
        )
    else:
        model = MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(vocab_size=512, vision_config=DeepSeekVisionConfig(**common))
        )
    configure_posttraining(model)
    model.to(device)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    rows = []
    for index, color in enumerate(((220, 10, 15), (15, 20, 230))):
        path = tmp_path / f"{index}.png"
        image = Image.new("RGB", (56, 56), color)
        image.save(path)
        rows.append(
            dict(
                prompt="Describe <|image|>",
                domain="vision",
                effort="low" if index == 0 else "high",
                verifier=dict(kind="exact_text", answer="red"),
                media=[dict(path=path.name, min_pixels=1024, **decoded_hashes(image))],
            )
        )
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    names = {"kimi": "minikimik3", "qwen": "miniqwen4", "deepseek": "minideepseekv4"}
    dataset = TaskDataset(
        tmp_path,
        "train",
        tokenizer,
        192,
        family=names[family],
        model_vocab_size=512,
        max_features=4,
    )
    objective = RolloutObjective(
        model, reference, tokenizer, dataset, group_size=2, max_new_tokens=4, method=method
    )
    rewards = iter([0.0, 1.0, 0.0, 1.0])
    # Reward correctness has independent verifier tests; this forces a nonzero
    # group advantage while checking real sampling probabilities and image gradients.
    monkeypatch.setattr("minifrontier.training.verifiers.reward", lambda *args: next(rewards))
    if method != "grpo":
        teachers = ResidentTestTeachers()
        for index, key in enumerate(("vision:low", "vision:high")):
            teacher = copy.deepcopy(reference)
            head = teacher.head if family == "deepseek" else teacher.lm_head
            with torch.no_grad():
                head.weight.mul_(1.2 + index * 0.1)
            teachers[key] = teacher
        objective.teachers = teachers
    inputs = torch.stack([dataset[i][0] for i in range(2)]).to(device)
    indices = torch.tensor([[0], [1]], device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        prepared = objective.prepare(inputs, indices)
    assert model.training
    assert [span["batch_index"] for span in prepared["media"]] == [0, 1, 2, 3]
    assert prepared["media_counts"][0] == 4 and len(prepared["terminations"]) == 4
    for span in prepared["media"]:
        start = span["start"]
        assert (
            prepared["labels"][span["batch_index"], start : start + span["feature_count"]]
            .eq(-100)
            .all()
        )
    if method == "opd":
        assert [entry["indices"].tolist() for entry in prepared["opd_targets"]] == [[2, 3], [0, 1]]
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        loss, _, _ = objective(model, inputs, indices, prepared=prepared)
    assert torch.isfinite(loss)
    tolerance = 0.02 if amp else 0.001 if family == "kimi" and device == "cuda" else 2e-5
    assert objective.last_ratio_error < tolerance
    record_property("frozen_ratio_max_error", objective.last_ratio_error)
    record_property("precision", "bf16" if amp else "fp32")
    record_property("tolerance", tolerance)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision.parameters())
    assert all(p.grad is None for p in reference.parameters())
