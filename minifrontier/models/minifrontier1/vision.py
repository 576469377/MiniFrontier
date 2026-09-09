# ViT blocks and patch embedding derive from Transformers 4177486 (Apache-2.0).
"""Random native Conv3D ViT; axial spatial positions and a 1536→512→512 merger."""

from dataclasses import asdict
from types import SimpleNamespace

from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.miniqwen4.upstream_vision import (
    Qwen4ExpVisionBlock,
    Qwen4ExpVisionPatchEmbed,
    Qwen4ExpVisionRotaryEmbedding,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)


class NativeVision(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        args = SimpleNamespace(
            **asdict(c),
            in_channels=3,
            hidden_act="gelu_pytorch_tanh",
            num_attention_heads=c.num_heads,
            _attn_implementation="sdpa",
            rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        )
        self.patch_embed = Qwen4ExpVisionPatchEmbed(args)
        self.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(args)
        self.blocks = nn.ModuleList(Qwen4ExpVisionBlock(args) for _ in range(c.depth))
        self.norm = nn.LayerNorm(c.hidden_size)
        self.merger = nn.Sequential(
            nn.Linear(c.hidden_size * 4, c.output_size),
            nn.GELU(),
            nn.Linear(c.output_size, c.output_size),
        )

    def forward(self, patches, grid_thw):
        c = self.config
        if (
            grid_thw.ndim != 2
            or grid_thw.shape[-1] != 3
            or not grid_thw.numel()
            or (grid_thw <= 0).any()
            or (grid_thw[:, 1:] % 2).any()
        ):
            raise ValueError("native vision requires complete temporal/spatial grids")
        if patches.numel() != int(grid_thw.prod(-1).sum()) * 3 * 2 * c.patch_size**2:
            raise ValueError("patch count/shape disagrees with Conv3D grid")
        h = self.patch_embed(patches)
        positions = get_vision_position_ids(grid_thw, 2)
        cu = get_vision_cu_seqlens(grid_thw)
        rotary = self.rotary_pos_emb(h, positions)
        for block in self.blocks:
            h = (
                checkpoint(block, h, cu, rotary, use_reentrant=False)
                if self.training and c.gradient_checkpointing
                else block(h, cu, rotary)
            )
        merged = self.merger(self.norm(h).reshape(-1, c.hidden_size * 4))
        return merged.split((grid_thw.prod(-1) // 4).tolist())
