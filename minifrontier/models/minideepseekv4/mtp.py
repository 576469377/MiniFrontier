"""Text MTP, separate from DSpark, using V3 fusion and V4's genuine HC streams.

M:[h,e] 2H->H is exactly concatenation of the two V4 e_proj/h_proj matrices,
applied per stream. It preserves four distinct streams, rather than collapsing
and repeating the target hidden. Embedding and output vocabulary weight are
provided by the main model and are not copied into this module's state dict.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .expert import TrainingGate
from .upstream_layers import Block, Linear, RMSNorm


class DeepSeekMTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        args = config.upstream_config()
        args.compress_ratios = (*args.compress_ratios, 0)  # local one-block MTP uses SWA
        self.block = Block(config.n_layers, args)
        self.block.ffn.gate = TrainingGate(config.n_layers, args)
        h, hc, eps = config.dim, config.hc_mult, config.norm_eps
        self.enorm, self.hnorm, self.norm = RMSNorm(h, eps), RMSNorm(h, eps), RMSNorm(h, eps)
        self.fusion = Linear(h * 2, h)
        self.hc_head_fn = nn.Parameter(torch.empty(hc, hc * h))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 0.01))
        self.hc_head_base = nn.Parameter(torch.zeros(hc))

    def forward(self, hidden, embedding, shifted_ids, valid, head):
        e = self.enorm(embedding).unsqueeze(2).expand_as(hidden)
        h = self.fusion(torch.cat((self.hnorm(hidden), e), -1))
        self.block.ffn.gate.valid_mask = valid
        self.block.ffn.gate.sequence_balance_enabled = self.config.sequence_balance_coef > 0
        self.block.attn.query_valid = valid
        self.block.attn.image_visible = None
        if self.training and self.config.gradient_checkpointing:
            h, seq = checkpoint(self._block, h, shifted_ids, use_reentrant=False)
        else:
            h, seq = self._block(h, shifted_ids)
        sample = self.norm(head.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base))
        return sample, h, seq

    def _block(self, h, ids):
        h = self.block(h, 0, ids)
        return h, self.block.ffn.gate.sequence_loss
