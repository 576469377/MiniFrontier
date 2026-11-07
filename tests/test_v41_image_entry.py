from dataclasses import asdict

import pytest
import torch
from PIL import Image
from test_deepseek_v41 import tiny
from test_native_data import tokenizer as tokenizer

from minifrontier.data import sha256
from minifrontier.inference.runtime import load_checkpoint, respond
from minifrontier.models.minideepseekv41 import MiniDeepSeekV41ForCausalLM
from minifrontier.models.minideepseekv41.processing import process_image
from minifrontier.models.minideepseekv41.vision import DeepSeekVisionConfig
from minifrontier.training.transitions import load_previous


@pytest.mark.parametrize(("width", "height"), [(126, 84), (84, 126), (126, 42)])
def test_row_major_grid_and_feature_budget(width, height):
    im = Image.new("RGB", (width, height))
    span = process_image(im, start=1, min_pixels=1, max_features=64)
    h, w = height // 42, width // 42
    assert span["types"].tolist() == [0] + ([2] * w + [3]) * h + [4]
    assert span["feature_count"] == w * h
    assert process_image(im, start=8, min_pixels=1, max_features=1)["feature_count"] == 1


def test_migrate_old_engram_and_generate_with_png(tmp_path, tokenizer):
    torch.set_num_threads(1)
    vision = DeepSeekVisionConfig(
        depth=1, hidden_size=16, num_heads=2, intermediate_size=32, output_size=32
    )
    old = MiniDeepSeekV41ForCausalLM(
        tiny(vocab_size=tokenizer.get_vocab_size(), engram_pad_id=2)
    ).eval()
    new = MiniDeepSeekV41ForCausalLM(
        tiny(vocab_size=tokenizer.get_vocab_size(), vision_config=vision)
    ).eval()
    saved = dict(config=asdict(old.config), model=old.state_dict(), model_name="minideepseekv41")
    report = load_previous(new, saved, transition="text-to-vision")
    assert new.config.engram_pad_id == report["inherited_engram_pad_id"] == 2
    ids = torch.tensor([[1, 12, 22, 2]])
    with torch.no_grad():
        torch.testing.assert_close(new(ids).logits, old(ids).logits, rtol=0, atol=0)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    ckpt = tmp_path / "model.pt"
    torch.save(
        dict(
            saved,
            config=asdict(new.config),
            model=new.state_dict(),
            stage="sft",
            step=1,
            phase="sparse_pretrain",
            tokenizer_sha256=sha256(tmp_path / "tokenizer.json"),
        ),
        ckpt,
    )
    f = tmp_path / "test.png"
    Image.new("RGB", (84, 42), "blue").save(f)
    with torch.inference_mode():
        m, t, _ = load_checkpoint(ckpt)
        assert isinstance(
            respond(m, t, "Color?", images=[str(f)], temperature=0, max_new_tokens=2), str
        )
    with pytest.raises(ValueError, match="vision tower"):
        respond(old, t, "Color?", images=[str(f)], max_new_tokens=1)
