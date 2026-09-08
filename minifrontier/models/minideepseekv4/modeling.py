"""MiniDeepSeek-V4 unquantized text training adapter, based on Flash 60d8d70.

Initialization is a local from-scratch recipe: normal(.02) projections, zero
router correction, small dynamic mHC scales and identity-biased static mixing.
The released checkpoint loader does not supply random-initialization values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.cache_utils import cache_transaction
from minifrontier.models.common import CausalLMOutput, validate_batch
from minifrontier.strict_types import validate_dataclass_payload
from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce
from minifrontier.training.mtp import shifted_batch

from .expert import TrainingGate
from .mtp import DeepSeekMTP
from .upstream_layers import Block, ParallelHead, RMSNorm
from .upstream_visibility import get_image_visible
from .vision import DeepSeekVision, DeepSeekVisionConfig


@dataclass
class MiniDeepSeekV4Config:
    vocab_size: int = 65536
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    head_dim: int = 64
    rope_head_dim: int = 16
    q_lora_rank: int = 128
    o_groups: int = 2
    o_lora_rank: int = 128
    moe_inter_dim: int = 256
    n_routed_experts: int = 32
    n_activated_experts: int = 2
    n_shared_experts: int = 1
    n_hash_layers: int = 2
    window_size: int = 64
    compress_ratios: tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128)
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    route_scale: float = 1.0  # Historical checkpoint default; Text-v2 explicitly uses 1.5.
    sequence_balance_coef: float = 0.0  # New Text-v2 config opts into 1e-4.
    swiglu_limit: float = 10.0
    initializer_range: float = 0.02
    qat_scheme: str = "bf16"
    expert_execution: str = "loop"
    pad_token_id: int = 0
    eos_token_id: int = 2
    gradient_checkpointing: bool = True
    forbidden_action_ids: tuple[int, ...] = (0, 1)
    mtp_enabled: bool = False
    mtp_loss_coef: float = 0.1
    vision_config: DeepSeekVisionConfig | None = None

    def upstream_config(self):
        validate_dataclass_payload(type(self), asdict(self))
        for name, value in asdict(self).items():
            if type(value) is int and name not in {"pad_token_id", "n_hash_layers"} and value < 1:
                raise ValueError(f"{name} must be positive")
        if len(self.compress_ratios) != self.n_layers or any(
            r not in (0, 4, 128) for r in self.compress_ratios
        ):
            raise ValueError("one compression ratio (0/4/128) is required per layer")
        if (
            self.n_heads % self.o_groups
            or self.rope_head_dim % 2
            or self.rope_head_dim > min(self.head_dim, self.index_head_dim)
        ):
            raise ValueError("invalid attention dimensions")
        if self.n_activated_experts > self.n_routed_experts or self.n_shared_experts != 1:
            raise ValueError("invalid V4 expert configuration")
        if not 0 <= self.n_hash_layers <= self.n_layers:
            raise ValueError("invalid number of hash layers")
        if (
            not 0 <= self.pad_token_id < self.vocab_size
            or not 0 <= self.eos_token_id < self.vocab_size
            or self.pad_token_id == self.eos_token_id
        ):
            raise ValueError("invalid special token IDs")
        return SimpleNamespace(
            **asdict(self),
            score_func="sqrtsoftplus",
            expert_dtype=None,
            max_batch_size=1,
        )


class MiniDeepSeekV4ForCausalLM(nn.Module):
    source_revision = "60d8d70770c6776ff598c94bb586a859a38244f1"
    training_ready = True

    def __init__(self, config: MiniDeepSeekV4Config, *, training_phase="dense_pretrain"):
        super().__init__()
        if isinstance(config.vision_config, dict):
            config.vision_config = DeepSeekVisionConfig(**config.vision_config)
        self.config = config
        self.indexer_loss_enabled = True
        args = config.upstream_config()
        self.embed = nn.Embedding(config.vocab_size, config.dim, config.pad_token_id)
        self.layers = nn.ModuleList(Block(i, args) for i in range(config.n_layers))
        for i, raw_layer in enumerate(self.layers):
            layer = cast(Block, raw_layer)
            layer.ffn.gate = TrainingGate(i, args)
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = ParallelHead(config.vocab_size, config.dim, config.norm_eps, config.hc_eps)
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, config.hc_mult * config.dim))
        self.hc_head_base = nn.Parameter(torch.zeros(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.full((1,), 0.01))
        self.vision = DeepSeekVision(config.vision_config) if config.vision_config else None
        if self.vision is not None:
            assert config.vision_config is not None
            if config.vision_config.output_size != config.dim:
                raise ValueError("visual aligner output and text width must match")
            self.image_start = nn.Parameter(torch.empty(config.dim))
            self.image_end = nn.Parameter(torch.empty(config.dim))
            self.image_newline = nn.Parameter(torch.empty(config.dim))
            self.image_pad = nn.Parameter(torch.empty(config.dim))
        self.mtp = None
        self._initialize()
        self.mtp = DeepSeekMTP(config) if config.mtp_enabled else None
        if self.mtp is not None:
            self._initialize(self.mtp)
        self.configure_training_phase(training_phase)
        if config.qat_scheme not in {"bf16", "mxfp4-indexer-v1"}:
            raise ValueError("unsupported DeepSeek QAT recipe")
        if config.qat_scheme != "bf16":
            from minifrontier.training.deepseek_qat import configure

            self.qat_recipe = configure(self)
        if config.expert_execution != "loop":
            from minifrontier.models.grouped_experts import configure as configure_experts

            configure_experts(self, config.expert_execution)

    @torch.no_grad()
    def _initialize(self, module=None):
        for name, p in (self if module is None else module).named_parameters():
            if p.is_floating_point():
                if p.ndim >= 2 or name in {
                    "image_start",
                    "image_end",
                    "image_newline",
                    "image_pad",
                }:
                    nn.init.normal_(p, std=self.config.initializer_range)
                elif name.endswith("weight"):
                    p.fill_(1)
                elif name.endswith("scale"):
                    p.fill_(0.01)
                else:
                    p.zero_()
        self.embed.weight[self.config.pad_token_id].zero_()
        for raw_layer in self.layers if module is None else [module.block]:
            layer = cast(Block, raw_layer)
            hc = self.config.hc_mult
            for base in (layer.hc_attn_base, layer.hc_ffn_base):
                base[2 * hc :].view(hc, hc).copy_(torch.eye(hc) * 4)
            gate = layer.ffn.gate
            if gate.hash:
                # Deterministic per-token hash routing with distinct selected experts.
                ids = torch.arange(self.config.vocab_size)[:, None]
                slots = torch.arange(self.config.n_activated_experts)[None, :]
                gate.tid2eid.copy_(
                    ((ids * 2654435761 + slots) % self.config.n_routed_experts).int()
                )

    def configure_training_phase(self, phase):
        if phase not in {"dense_pretrain", "dense_distill", "sparse_cpt"}:
            raise ValueError("unsupported DeepSeek training phase")
        self.training_phase = phase
        for parameter in self.parameters():
            if parameter.is_floating_point():
                parameter.requires_grad_(phase != "dense_distill")
        for raw_layer in self.layers:
            layer = cast(Block, raw_layer)
            layer.attn.training_phase = phase
            if layer.attn.indexer is not None:
                layer.attn.indexer.requires_grad_(phase != "dense_pretrain")
            if layer.ffn.gate.bias_vl is not None:
                layer.ffn.gate.bias_vl.requires_grad_(False)
            if layer.ffn.gate.bias is not None:
                layer.ffn.gate.bias.requires_grad_(False)
        if self.mtp is not None:
            self.mtp.requires_grad_(phase != "dense_distill" and self.config.mtp_loss_coef > 0)
            for name, p in self.mtp.named_parameters():
                if name.endswith(("gate.bias", "gate.bias_vl")):
                    p.requires_grad_(False)
        self.zero_grad(set_to_none=True)

    @staticmethod
    def _block(layer, h, input_ids, valid_mask, sequence_balance_enabled, image_visible):
        layer.ffn.gate.valid_mask = valid_mask
        layer.ffn.gate.sequence_balance_enabled = sequence_balance_enabled
        layer.attn.query_valid = valid_mask
        layer.attn.image_visible = image_visible
        h = layer(h, 0, input_ids)
        return h, layer.attn.indexer_loss, layer.ffn.gate.sequence_loss

    @cache_transaction
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        *,
        inputs_embeds=None,
        cache=None,
        media=None,
        return_hidden=False,
        return_logits=True,
        opd_targets=None,
        opd_mask=None,
        opd_vocab_size=None,
        return_taps=False,
    ):
        if cache is not None:
            if self.training or torch.is_grad_enabled() or labels is not None:
                raise ValueError("cached inference requires eval, no_grad and no labels")
            if attention_mask is not None and not attention_mask.bool().all():
                raise ValueError("cached batches must be unpadded")
            if cache.length + input_ids.shape[1] > self.config.max_seq_len:
                raise ValueError("cached sequence exceeds context")
            if cache.length and media:
                raise ValueError("complete images must be in the initial prefill")
            cache.prepare(self, input_ids)
        for index, layer in enumerate(self.layers):
            layer.attn.decode_state = None if cache is None else cache.layers[index]  # type: ignore[union-attr]
        image_mask = input_ids >= self.config.vocab_size
        if image_mask.any() and (
            self.vision is None or (input_ids >= self.config.vocab_size + 5).any()
        ):
            raise ValueError("invalid Vision-Exp logical image IDs")
        safe_ids = input_ids.masked_fill(image_mask, self.config.pad_token_id)
        labels = validate_batch(safe_ids, self.config, attention_mask, labels)
        if labels is not None and labels[image_mask].ne(-100).any():
            raise ValueError("image sentinel spans are not softmax targets")
        embeddings = self.embed(safe_ids) if inputs_embeds is None else inputs_embeds
        embeddings = embeddings.clone()
        coverage = torch.zeros_like(image_mask)
        visible = None
        if media:
            if self.vision is None:
                raise ValueError("checkpoint has no native visual tower")
            params = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            )
            for span in media:
                batch, start = span["batch_index"], span["start"]
                types, perm = span["types"].to(input_ids.device), span["perm"].to(input_ids.device)
                end = start + types.numel()
                if (
                    not 0 <= batch < input_ids.shape[0]
                    or not 0 <= start < end <= input_ids.shape[1]
                ):
                    raise ValueError("image span outside language sequence")
                if coverage[batch, start:end].any() or not torch.equal(
                    input_ids[batch, start:end], self.config.vocab_size + types
                ):
                    raise ValueError("overlap or sentinel/type mismatch")
                if types.eq(0).sum() != 1 or types.eq(4).sum() != 1 or types[-1] != 4:
                    raise ValueError("image must contain complete start/end boundaries")
                features = self.vision(
                    span["patches"].to(input_ids.device), span["n_vit_h"], span["n_vit_w"]
                )
                if features.shape[0] != types.eq(2).sum() or not torch.equal(
                    perm.sort().values, torch.arange(features.shape[0], device=perm.device)
                ):
                    raise ValueError("N-layout feature permutation mismatch")
                block = params[types].clone()
                block[types == 2] = features[perm].to(block.dtype)
                embeddings[batch, start:end] = block.to(embeddings.dtype)
                coverage[batch, start:end] = True
            # The actual processor span length includes start/end/newline/pad.
            visible = get_image_visible(
                input_ids, self.config.vocab_size, max(s["types"].numel() for s in media)
            )
        if not torch.equal(coverage, image_mask):
            raise ValueError("image sentinel has missing media features")
        if embeddings.shape != (*input_ids.shape, self.config.dim):
            raise ValueError("DeepSeek input embeddings must match language width")
        h = embeddings.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
        losses = []
        sequence_losses = []
        valid_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        seq_enabled = (
            self.config.sequence_balance_coef > 0 and self.training_phase != "dense_distill"
        )
        taps = []
        for index, raw_layer in enumerate(self.layers):
            layer = cast(Block, raw_layer)
            layer.attn.indexer_loss_enabled = self.indexer_loss_enabled
            if self.training and self.config.gradient_checkpointing:
                h, kl, seq = checkpoint(
                    self._block,
                    layer,
                    h,
                    input_ids,
                    valid_mask,
                    seq_enabled,
                    visible,
                    use_reentrant=False,
                )
            else:
                h, kl, seq = self._block(layer, h, input_ids, valid_mask, seq_enabled, visible)
            if not layer.ffn.gate.hash:
                sequence_losses.append(seq)
            if layer.attn.indexer is not None:
                losses.append(kl)
            if return_taps and index >= self.config.n_layers - 3:
                taps.append(h.mean(dim=2))
        features = self.norm(
            self.head.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        )
        logits = self.head.get_logits(features) if return_logits else None
        lm = None
        if labels is not None:
            lm = (
                causal_lm_loss(logits, labels)
                if logits is not None
                else chunked_linear_ce(features, self.head.weight, labels)
            )
        kl = torch.stack(losses).mean() if losses else features.new_zeros(())
        loss = lm
        seq = torch.stack(sequence_losses).mean() if sequence_losses else features.new_zeros(())
        if loss is not None and seq_enabled:
            loss = loss + self.config.sequence_balance_coef * seq
        if self.training_phase == "dense_distill":
            loss = kl
        elif self.training_phase == "sparse_cpt" and lm is not None and self.indexer_loss_enabled:
            assert loss is not None
            loss = loss + kl
        if cache is not None:
            cache.commit(input_ids.shape[1])
        mtp_loss, mtp_tokens = None, 0
        mtp_aux, mtp_aux_count = None, 0
        if (
            self.mtp is not None
            and labels is not None
            and self.config.mtp_loss_coef > 0
            and self.training_phase != "dense_distill"
        ):
            shifted, targets, valid = shifted_batch(
                input_ids, labels, vocab_size=self.config.vocab_size, attention_mask=attention_mask
            )
            predicted, _, mtp_seq = self.mtp(h, self.embed(shifted), shifted, valid, self.head)
            mtp_loss = chunked_linear_ce(predicted, self.head.weight, targets, shift=False)
            mtp_tokens = int(targets.ne(-100).sum())
            mtp_aux = self.config.sequence_balance_coef * mtp_seq
            mtp_aux_count = int(valid.any(-1).sum())
            loss = loss + mtp_aux
            # MTP sequence balance is normalized independently with its own active samples.
            loss = loss + self.config.mtp_loss_coef * mtp_loss
        if opd_targets is not None:
            if labels is not None or opd_mask is None:
                raise ValueError("OPD uses generated response masks, not CE targets")
            from minifrontier.training.deepseek_opd import trajectory_loss
            from minifrontier.training.distributions import forbidden_actions

            loss = trajectory_loss(
                features,
                self.head.weight,
                opd_targets,
                opd_mask,
                vocab_size=opd_vocab_size or self.config.vocab_size,
                forbidden_ids=forbidden_actions(self),
            )
        return CausalLMOutput(
            logits,
            loss,
            lm,
            aux_loss=seq,
            indexer_loss=kl,
            hidden_states=features if return_hidden else None,
            multistream_hidden=h if return_hidden else None,
            mtp_loss=mtp_loss,
            mtp_tokens=mtp_tokens,
            mtp_aux_loss=mtp_aux,
            mtp_aux_count=mtp_aux_count,
            tapped_hidden_states=tuple(taps) if return_taps else None,
        )
