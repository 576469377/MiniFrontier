from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from minifrontier.models.minideepseekv41.vision import DeepSeekVision, DeepSeekVisionConfig
from minifrontier.training.v41_vision import (
    CaptionContrast,
    install_tower,
    positives,
    set_main_tower_trainable,
    tower_payload,
)
from scripts.v41_vision_curriculum import batches, read


def config(output=24):
    return dict(
        depth=2,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        patch_size=2,
        output_size=output,
        gradient_checkpointing=False,
    )


def test_batched_sigmoid_uses_native_vit_and_supervises_both_towers():
    torch.manual_seed(7)
    net = CaptionContrast(config(), 32, width=8)
    pixels = torch.rand(3, 3, 8, 8)
    ids = torch.tensor([[2, 3, 0], [7, 5, 0], [2, 3, 0]])
    patches = F.unfold(pixels, kernel_size=2, stride=2).transpose(1, 2).reshape(3, 16, 3, 2, 2)
    refs = F.normalize(
        net.image_head(torch.stack([net.vit(p, 4, 4).mean(0) for p in patches])), dim=-1
    )
    torch.testing.assert_close(net.images(pixels), refs)
    assert positives(ids).sum() == 5
    loss = net(pixels, ids)
    loss.backward()
    assert (
        loss.isfinite()
        and net.vit.patch_embed.proj.weight.grad.norm() > 0
        and net.words.weight.grad.norm() > 0
    )


def test_only_vit_is_inherited_and_final_norm_remains_trainable():
    src = DeepSeekVision(DeepSeekVisionConfig(**config()))
    dst = SimpleNamespace(vision=DeepSeekVision(DeepSeekVisionConfig(**config(32))))
    fresh = dst.vision.aligner.w1.weight.detach().clone()
    art = tower_payload(src.vit, config(), phase="contrastive", evidence={})
    with pytest.raises(ValueError, match="phase"):
        install_tower(dst, art, "main")
    install_tower(dst, art, "temporary")
    assert torch.equal(fresh, dst.vision.aligner.w1.weight)
    for before, after in zip(src.vit.parameters(), dst.vision.vit.parameters(), strict=True):
        assert torch.equal(before, after)
    set_main_tower_trainable(dst, False)
    assert dst.vision.vit.norm.weight.requires_grad and dst.vision.aligner.w1.weight.requires_grad
    assert not dst.vision.vit.blocks[0].attn.wqkv.weight.requires_grad


def test_order_unique_resume_and_true_main_config():
    full = list(batches(17, 4, 0, 20, 19))
    assert all(len(set(batch)) == 4 for batch in full)
    assert full[3:] == list(batches(17, 4, 3, 20, 19))
    main = read(Path("configs/strategies/minideepseekv41-vision-main.json"))
    temp = read(Path("configs/strategies/minideepseekv41-vision-temporary.json"))
    assert main["hierarchical_indexing"] and main["dim"] > temp["dim"]
    assert main["vision_config"]["hidden_size"] == temp["vision_config"]["hidden_size"]


def test_midrun_thaw_preserves_text_moments_and_defers_tower_updates():
    from minifrontier.training.transitions import set_visual_rates

    class Skeleton(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = DeepSeekVision(DeepSeekVisionConfig(**config()))
            self.embed = torch.nn.Linear(3, 3)

    model = Skeleton()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    set_visual_rates(opt, model, vision_lr=3e-5, projector_lr=3e-4)
    rgb = model.vision.vit.patch_embed.proj.weight
    fresh = rgb.clone()
    for active in (False, False, True):
        set_main_tower_trainable(model, active)
        opt.zero_grad(set_to_none=True)
        sum(p.square().mean() for p in model.parameters() if p.requires_grad).backward()
        opt.step()
        if not active:
            assert torch.equal(fresh, rgb) and rgb not in opt.state
    assert opt.state[model.embed.weight]["step"] == 3 and opt.state[rgb]["step"] == 1
    assert not torch.equal(fresh, rgb)
