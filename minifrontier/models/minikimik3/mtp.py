"""K3 mini MTP: local shifted fusion with the pinned MLA/LatentMoE/AttnRes block.

The released K3 config sets num_nextn_predict_layers=0. This module is explicitly
our training recipe, not an extracted official MTP training implementation.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attnres import _apply_attn_res
from .upstream_layers import KimiDecoderLayer, KimiRMSNorm


class KimiMTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        args = config.upstream_config()
        args.is_kda_layer = lambda i: False
        h, eps = config.hidden_size, config.rms_norm_eps
        self.enorm, self.hnorm = KimiRMSNorm(h, eps), KimiRMSNorm(h, eps)
        self.fusion = nn.Linear(h * 2, h, bias=False)
        self.block = KimiDecoderLayer(args, config.num_hidden_layers)
        self.output_attn_res_norm = KimiRMSNorm(h, eps)
        self.output_attn_res_proj = nn.Linear(h, 1, bias=False)
        self.norm = KimiRMSNorm(h, eps)

    def forward(self, hidden, embedding, blocks, valid):
        h = self.fusion(torch.cat((self.hnorm(hidden), self.enorm(embedding)), -1))
        batch, length, width = h.shape
        # Retain causal keys with a diagonal fallback for fully masked queries.
        legal = (
            torch.ones((length, length), device=h.device, dtype=torch.bool).tril()[None]
            & valid[:, None]
        )
        legal |= torch.eye(length, device=h.device, dtype=torch.bool)[None]
        mask = torch.zeros_like(legal, dtype=h.dtype).masked_fill(~legal, float("-inf"))[:, None]
        self.block.block_sparse_moe.gate.valid_mask = valid
        self.block.self_attn.valid_mask = valid
        if self.training and self.config.gradient_checkpointing:
            h, blocks = checkpoint(
                self.block, h, attention_mask=mask, block_residual=blocks, use_reentrant=False
            )
        else:
            h, blocks = self.block(h, attention_mask=mask, block_residual=blocks)
        h = _apply_attn_res(
            h.reshape(-1, width), blocks, self.output_attn_res_proj, self.output_attn_res_norm
        )
        return self.norm(h.view(batch, length, width))
