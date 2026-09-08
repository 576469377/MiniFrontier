"""Native image prefill through each draft objective and rejection replay."""

from unittest.mock import patch

import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config
from test_native_data import tokenizer as tokenizer
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.multimodal import prepare_record
from minifrontier.speculative import SpeculativeSession, probability
from minifrontier.training.drafts import DraftObjective, build_draft


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4", "minideepseekv4"])
def test_native_image_draft_gradients_and_rejected_cache(family, tokenizer, tmp_path):
    torch.manual_seed(792)
    Image.new("RGB", (56, 56), (210, 25, 19)).save(tmp_path / "red.png")
    example = prepare_record(
        dict(
            stage="sft",
            turns=[
                dict(role="user", content="Describe <|image|>"),
                dict(role="assistant", content="red"),
            ],
            media=[dict(path="red.png", min_pixels=1024)],
        ),
        tokenizer,
        family,
        root=tmp_path,
        max_features=4,
        model_vocab_size=512,
    )
    common = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32)
    if family == "minikimik3":
        target = MiniKimiK3ForCausalLM(
            tiny_kimi(
                num_hidden_layers=12,
                vocab_size=512,
                mtp_enabled=True,
                vision_config=KimiVisionConfig(qkv_hidden_size=48, **common),
            )
        )
    elif family == "miniqwen4":
        target = MiniQwen4ForCausalLM(
            tiny_config(
                vocab_size=512,
                hidden_size=32,
                mtp_enabled=True,
                max_position_embeddings=256,
                vision_config=QwenVisionConfig(**common),
            )
        )
    else:
        target = MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(
                vocab_size=512, mtp_enabled=True, vision_config=DeepSeekVisionConfig(**common)
            )
        )
    draft = build_draft(family, target)
    length = int(example.labels[0].ne(-100).nonzero()[0])
    prefix = example.input_ids[:, :length]
    media = example.extras["media"]
    # Target reads the real image, while the draft learns only from known taps.
    trajectory = torch.cat((prefix, torch.tensor([[21, 23, 25, 29, 31, 2]])), 1)
    with patch(
        "torch.multinomial",
        side_effect=lambda p, n: torch.full((p.shape[0], n), 23, device=p.device),
    ):
        loss, count, normalizer = DraftObjective(family, draft)(trajectory, length, media)
    assert count > 0 and normalizer > 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in draft.parameters())
    assert all(p.grad is None for p in target.parameters())
    draft.eval()
    with torch.no_grad():
        session = SpeculativeSession(target, prefix, media=media)
        with patch("minifrontier.speculative.accepts", side_effect=[True, False]):
            emitted = session.advance(lambda context, maximum: draft.propose(context, steps=3), 5)
        assert emitted.shape[1] == 2 and session.cache.length == length + 2
        expected = target(session.context.ids, media=media).logits[:, -1]
        torch.testing.assert_close(
            session.context.next_p, probability(expected, target), rtol=4e-5, atol=4e-6
        )
