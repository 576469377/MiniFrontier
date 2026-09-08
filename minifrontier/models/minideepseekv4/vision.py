"""Flash-Vision-Exp native 2D-RoPE/RMSNorm/SwiGLU ViT and 3x3 aligner."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.strict_types import validate_dataclass_payload

from .upstream_vision import Aligner, ViT, get_vision_cos_sin


@dataclass
class DeepSeekVisionConfig:
    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    intermediate_size: int = 1056
    patch_size: int = 14
    downsample_ratio: int = 3
    output_size: int = 512
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = True


class DeepSeekVision(nn.Module):
    def __init__(self, config: DeepSeekVisionConfig):
        super().__init__()
        validate_dataclass_payload(DeepSeekVisionConfig, asdict(config))
        if config.hidden_size % (4 * config.num_heads) or config.downsample_ratio != 3:
            raise ValueError("invalid native Vision-Exp head/aligner configuration")
        self.config = config
        args = SimpleNamespace(
            vision_n_layers=config.depth,
            vision_dim=config.hidden_size,
            vision_n_heads=config.num_heads,
            vision_inter_dim=config.intermediate_size,
            vision_patch_size=config.patch_size,
            vision_downsample_ratio=3,
            dim=config.output_size,
            vision_rope_theta=config.rope_theta,
        )
        self.vit, self.aligner = ViT(args), Aligner(args)

    def forward(self, patches, n_h, n_w):
        if min(n_h, n_w) < 1 or patches.shape[0] != n_h * n_w:
            raise ValueError("Vision-Exp patch/grid mismatch")
        x = self.vit.patch_embed(patches.to(self.vit.patch_embed.proj.weight.dtype))
        cos, sin = get_vision_cos_sin(n_h, n_w, self.vit.rope_dim, self.vit.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        for block in self.vit.blocks:
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.aligner(self.vit.norm(x), n_h, n_w)
