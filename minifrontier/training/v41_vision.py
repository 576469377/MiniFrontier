"""Scratch vision-language contrastive training and ViT-only stage handoffs."""

from dataclasses import asdict
from math import isqrt, log
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from minifrontier.models.minideepseekv41.upstream_vision import (
    Block,
    apply_rotary,
    get_vision_cos_sin,
)
from minifrontier.models.minideepseekv41.vision import DeepSeekVision, DeepSeekVisionConfig


class CaptionContrast(nn.Module):
    def __init__(self, config, vocab_size, width=128):
        super().__init__()
        self.vit = DeepSeekVision(DeepSeekVisionConfig(**config)).vit
        self.image_head = nn.Linear(config["hidden_size"], width, bias=False)
        self.words = nn.Embedding(vocab_size, width, padding_idx=0)
        self.text_head = nn.Linear(width, width, bias=False)
        self.scale = nn.Parameter(torch.tensor(log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def images(self, pixels):
        # Stage A has a fixed letterboxed canvas, so all images share one 2-D RoPE grid.
        b, _, side, _ = pixels.shape
        p = self.vit.patch_embed.proj.in_features // 3
        patch = isqrt(p)
        grid = side // patch
        patches = F.unfold(pixels, kernel_size=patch, stride=patch).transpose(1, 2)
        x = self.vit.patch_embed.proj(patches)
        cos, sin = get_vision_cos_sin(grid, grid, self.vit.rope_dim, self.vit.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        for raw_block in self.vit.blocks:
            block = cast(Block, raw_block)
            attn = block.attn
            norm = block.norm1(x)
            q, k, v = (
                a.reshape(b, grid * grid, attn.n_heads, attn.head_dim)
                for a in attn.wqkv(norm).chunk(3, dim=-1)
            )
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            values = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            x = x + attn.wo(values.transpose(1, 2).reshape_as(x))
            x = x + block.mlp(block.norm2(x))
        return F.normalize(self.image_head(self.vit.norm(x).mean(1)).float(), dim=-1)

    def texts(self, ids):
        mask = ids.ne(0)
        pooled = (self.words(ids) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return F.normalize(self.text_head(pooled).float(), dim=-1)

    def forward(self, pixels, ids):
        similarities = self.images(pixels) @ self.texts(ids).T
        signed = positives(ids).float() * 2 - 1
        logits = self.scale.clamp(max=log(100)).exp() * similarities + self.bias
        return -F.logsigmoid(signed * logits).sum() / len(ids)


def positives(ids):
    # Equal model-visible captions, including after the length limit, are all positives.
    return ids[:, None].eq(ids[None, :]).all(-1)


def tower_payload(vit, config, *, phase, evidence):
    topology = asdict(DeepSeekVisionConfig(**config))
    for name in ("output_size", "gradient_checkpointing", "downsample_ratio"):
        topology.pop(name)
    return dict(
        schema="minifrontier-v41-vit-v1",
        phase=phase,
        topology=topology,
        vit=vit.state_dict(),
        evidence=evidence,
    )


def install_tower(model, artifact, role):
    required = {"temporary": "contrastive", "main": "generative"}[role]
    topology = tower_payload(model.vision.vit, asdict(model.vision.config), phase="", evidence={})
    if (
        artifact.get("schema") != topology["schema"]
        or artifact.get("phase") != required
        or artifact.get("topology") != topology["topology"]
    ):
        raise ValueError("incompatible ViT topology or curriculum phase")
    model.vision.vit.load_state_dict(artifact["vit"], strict=True)


def set_main_tower_trainable(model, trainable):
    if getattr(model, "vision", None) is None:
        raise ValueError("V4.1 tower freeze requires a vision configuration")
    model.vision.vit.requires_grad_(trainable)
    model.vision.vit.norm.requires_grad_(True)
