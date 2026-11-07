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
from minifrontier.multimodal import collate, prepare_record
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


def test_retaining_v41_activations_preserves_image_sft_gradients(tmp_path, tokenizer):
    vision = DeepSeekVisionConfig(
        depth=1, hidden_size=16, num_heads=2, intermediate_size=32, output_size=32
    )
    config = tiny(
        vocab_size=tokenizer.get_vocab_size(), vision_config=vision, gradient_checkpointing=True
    )
    model = MiniDeepSeekV41ForCausalLM(config).train()
    image = tmp_path / "shape.png"
    Image.new("RGB", (42, 42), "red").save(image)
    record = dict(
        stage="sft",
        turns=[
            dict(role="user", content="<|image|>\nColor?"),
            dict(role="assistant", content="red"),
        ],
        media=[dict(path=str(image))],
    )
    batch = collate(
        [
            prepare_record(
                record,
                tokenizer,
                "minideepseekv41",
                model_vocab_size=config.vocab_size,
                min_pixels=1,
            )
        ],
        "cpu",
    )

    def gradients():
        model.zero_grad(set_to_none=True)
        loss = model(batch.input_ids, labels=batch.labels, **batch.extras).loss
        loss.backward()
        return loss.detach(), {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
        }

    loss, expected = gradients()
    for marker in ("image_start", "image_end", "image_newline"):
        assert 0 < expected[marker].norm().item() < 100
    for marker in (model.image_start, model.image_end, model.image_newline, model.image_pad):
        assert marker.std().item() > 0
    model.retain_activations = True
    model.vision.retain_activations = True
    actual_loss, actual = gradients()
    torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
    assert expected.keys() == actual.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(actual[name], grad, rtol=0, atol=0)
