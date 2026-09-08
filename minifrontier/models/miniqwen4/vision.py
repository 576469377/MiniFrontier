"""Native Conv3D Qwen4Exp vision tower, spatial interpolation and pre-LN merger."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.strict_types import validate_dataclass_payload

from .upstream_vision import (
    Qwen4ExpVisionBlock,
    Qwen4ExpVisionPatchEmbed,
    Qwen4ExpVisionPatchMerger,
    Qwen4ExpVisionRotaryEmbedding,
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


@dataclass
class QwenVisionConfig:
    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    output_size: int = 512
    gradient_checkpointing: bool = True


class QwenVision(nn.Module):
    def __init__(self, config: QwenVisionConfig):
        super().__init__()
        validate_dataclass_payload(QwenVisionConfig, asdict(config))
        if (
            config.hidden_size % (4 * config.num_heads)
            or int(config.num_position_embeddings**0.5) ** 2 != config.num_position_embeddings
        ):
            raise ValueError("invalid vision head or learned position grid")
        self.config = config
        args = SimpleNamespace(
            **asdict(config),
            out_hidden_size=config.output_size,
            in_channels=3,
            hidden_act="gelu_pytorch_tanh",
            num_attention_heads=config.num_heads,
            _attn_implementation="sdpa",
            rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        )
        self.patch_embed = Qwen4ExpVisionPatchEmbed(args)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(args)
        self.blocks = nn.ModuleList(Qwen4ExpVisionBlock(args) for _ in range(config.depth))
        self.merger = Qwen4ExpVisionPatchMerger(args)

    def forward(self, patches, grid_thw):
        c = self.config
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or not grid_thw.numel():
            raise ValueError("Qwen vision needs nonempty [media,3] grids")
        if (grid_thw <= 0).any() or (grid_thw[:, 1:] % c.spatial_merge_size).any():
            raise ValueError("Qwen vision needs complete spatial merges")
        expected = int(grid_thw.prod(-1).sum())
        if patches.numel() != expected * 3 * c.temporal_patch_size * c.patch_size**2:
            raise ValueError("Conv3D patch count does not match grid")
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            int(c.num_position_embeddings**0.5),
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=c.spatial_merge_size,
        )
        positions = get_vision_position_ids(grid_thw, c.spatial_merge_size)
        cu = get_vision_cu_seqlens(grid_thw)
        h = self.patch_embed(patches)
        h = h + (self.pos_embed(indices) * weights[..., None]).sum(1).to(h.dtype)
        rotary = self.rotary_pos_emb(h, positions)
        for block in self.blocks:
            if self.training and c.gradient_checkpointing:
                h = checkpoint(block, h, cu, rotary, use_reentrant=False)
            else:
                h = block(h, cu, rotary)
        merged = self.merger(h)
        return merged.split((grid_thw.prod(-1) // c.spatial_merge_size**2).tolist())
