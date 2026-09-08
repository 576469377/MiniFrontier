"""Capacity-scaled native MoonViT-V2 and its output-RMSNorm PatchMergerMLPV2."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.strict_types import validate_dataclass_payload

from .upstream_vision import (
    MoonVision3dPatchEmbed,
    MoonViT3dEncoder,
    PatchMergerMLPV2,
    tpool_patch_merger,
)


@dataclass
class KimiVisionConfig:
    depth: int = 12
    hidden_size: int = 384
    qkv_hidden_size: int = 576
    num_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 14
    temporal_merge: int = 4
    spatial_merge: int = 2
    position_grid: int = 64
    output_size: int = 512
    gradient_checkpointing: bool = True


class KimiVision(nn.Module):
    def __init__(self, config: KimiVisionConfig):
        super().__init__()
        validate_dataclass_payload(KimiVisionConfig, asdict(config))
        if (
            config.qkv_hidden_size % config.num_heads
            or config.spatial_merge != 2
            or config.temporal_merge != 4
        ):
            raise ValueError("invalid native MoonViT head/merge relationship")
        self.config = config
        self.patch_embed = MoonVision3dPatchEmbed(
            config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.position_grid,
            pos_emb_width=config.position_grid,
            pos_emb_time=4,
            patch_embed_proj_bias=False,
            pos_emb_interpolation_mode="bilinear",
        )
        self.encoder = MoonViT3dEncoder(
            config.hidden_size,
            config.depth,
            dict(
                num_heads=config.num_heads,
                hidden_dim=config.hidden_size,
                qkv_hidden_size=config.qkv_hidden_size,
                mlp_dim=config.intermediate_size,
                norm_type="rmsnorm",
                mlp_type="mlp2",
                activation=nn.GELU(approximate="tanh"),
                attn_bias=False,
                linear_bias=False,
                attn_implementation="sdpa",
            ),
        )
        self.merger = PatchMergerMLPV2(
            SimpleNamespace(
                projector_ln_eps=1e-5,
                mm_hidden_size=config.hidden_size,
                hidden_size=config.output_size,
                merge_kernel_size=(2, 2),
            )
        )

    def forward(self, patches, grid_thw):
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or not grid_thw.numel():
            raise ValueError("MoonViT needs nonempty [segments,3] grids")
        if (grid_thw <= 0).any() or (grid_thw[:, 0] > 4).any() or (grid_thw[:, 1:] % 2).any():
            raise ValueError("MoonViT groups contain <=4 frames and complete 2x2 spatial merges")
        if patches.shape != (
            int(grid_thw.prod(-1).sum()),
            3,
            self.config.patch_size,
            self.config.patch_size,
        ):
            raise ValueError("patch pixels and MoonViT grid disagree")
        h = self.patch_embed(patches, grid_thw)
        freqs = self.encoder.rope_2d.get_freqs_cis(grid_thw, h.device)
        lengths = grid_thw.prod(-1)
        cu = torch.cat((lengths.new_zeros(1), lengths)).cumsum(0, dtype=torch.int32)
        for block in self.encoder.blocks:
            if self.training and self.config.gradient_checkpointing:
                h = checkpoint(block, h, cu, int(lengths.max()), freqs, use_reentrant=False)
            else:
                h = block(h, cu, int(lengths.max()), freqs)
        h = self.encoder.final_layernorm(h)
        return self.merger(tpool_patch_merger(h, grid_thw, (2, 2)))
