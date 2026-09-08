"""96eccb8 vLLM MTP residual-linear-shared fusion adapted to training primitives.

Parameter layout: genuine 4H hidden norm; shared H->H projections; no PLE;
independent shared_head.head (untied release mapping), GR+global/QSA decoder.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .upstream_core import Qwen4ExpTextGatedResidual
from .upstream_decoder import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding
from .upstream_ple import Qwen4ExpTextRMSNorm


class IndependentMTPHead(nn.Module):
    def __init__(self, h, vocab):
        super().__init__()
        self.head = nn.Linear(h, vocab, bias=False)


class QwenMTP(nn.Module):
    def __init__(self, config, initialize):
        super().__init__()
        self.config = config
        args = config.upstream_config()
        args.layer_types = ["full_attention"]
        args.ple_layer_ids = []
        h, hc = config.hidden_size, config.hc_count
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(h * hc, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(h, eps=config.rms_norm_eps)
        self.fc_hidden, self.fc_embedding = nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False)
        self.block = Qwen4ExpTextDecoderLayer(args, 0)
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(args)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(args, use_combine=False)
        self.shared_head = IndependentMTPHead(h, config.vocab_size)
        self.apply(initialize)
        self.configure_phase("dense_pretrain")

    def configure_phase(self, phase):
        from .modeling import MiniQwen4StageIndexer

        indexer = self.block.self_attn.indexer
        indexer.__class__ = MiniQwen4StageIndexer
        indexer.sparse_enabled = phase == "sparse_cpt"
        indexer.requires_grad_(phase != "dense_pretrain")
        self.phase = phase

    def forward(self, multi, embedding, valid, positions):
        h = self.pre_fc_norm_hidden(multi).unflatten(
            -1, (self.config.hc_count, self.config.hidden_size)
        )
        h = self.fc_hidden(h) + self.fc_embedding(self.pre_fc_norm_embedding(embedding)).unsqueeze(
            -2
        )
        h = h.flatten(-2)
        length = h.shape[1]
        legal = (
            torch.ones((length, length), device=h.device, dtype=torch.bool).tril()[None]
            & valid[:, None]
        )
        legal |= torch.eye(length, device=h.device, dtype=torch.bool)[None]
        mask = torch.zeros_like(legal, dtype=h.dtype).masked_fill(~legal, float("-inf"))[:, None]
        kwargs = dict(
            position_embeddings=self.rotary_emb(h, positions),
            attention_mask=mask,
            conv_mask=valid,
            valid=valid,
            indexer_phase=self.phase if self.phase != "dense_pretrain" else None,
        )
        if self.training and self.config.gradient_checkpointing:
            h, router, kl = checkpoint(self._block, h, use_reentrant=False, **kwargs)
        else:
            h, router, kl = self._block(h, **kwargs)
        return self.hyper_connection_mixer(h), h, router, kl

    def _block(self, h, *, valid, **kwargs):
        from .modeling import _decoder_with_router

        self.block.mlp.gate.valid_mask = valid
        return _decoder_with_router(self.block, h, **kwargs)
