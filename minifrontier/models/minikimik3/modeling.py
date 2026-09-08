# Source-derived Kimi integration; Kimi K3 License retained in upstream_layers.py.
"""MiniKimi-K3 text LM built from the pinned K3 decoder, KDA, MLA and LatentMoE.

Initialization: source normal(0, .02) projections, K3 report A_log=0,
Mamba-style log-uniform dt in [.001,.1] transformed through inverse softplus;
zero routing correction and zero AttnRes query are explicit local choices.
The released inference code leaves dt_bias/correction_bias uninitialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.cache_utils import cache_transaction
from minifrontier.models.common import CausalLMOutput, insert_media_embeddings, validate_batch
from minifrontier.strict_types import validate_dataclass_payload
from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce
from minifrontier.training.mtp import shifted_batch

from .attnres import _apply_attn_res
from .mtp import KimiMTP
from .upstream_layers import KimiDecoderLayer, KimiDeltaAttention, KimiMoEGate, KimiRMSNorm
from .vision import KimiVision, KimiVisionConfig


@dataclass
class MiniKimiK3Config:
    vocab_size: int = 65536
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 2048
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    kda_head_dim: int = 64
    num_experts: int = 32
    num_experts_per_token: int = 2
    num_shared_experts: int = 2
    moe_intermediate_size: int = 256
    routed_expert_hidden_size: int = 256
    attn_res_block_size: int = 4
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    eos_token_id: int = 2
    gradient_checkpointing: bool = True
    forbidden_action_ids: tuple[int, ...] = (0, 1)
    router_fp32: bool = False  # Historical runs retain their original AMP routing.
    qat_scheme: str = "bf16"
    mtp_enabled: bool = False
    mtp_loss_coef: float = 0.1
    vision_config: KimiVisionConfig | None = None
    image_token_id: int = 7

    def upstream_config(self):
        validate_dataclass_payload(type(self), asdict(self))
        for name, value in asdict(self).items():
            if type(value) is int and name != "pad_token_id" and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError("source MLA requires equal Q and KV head counts")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("top-k exceeds expert count")
        if (
            not 0 <= self.pad_token_id < self.vocab_size
            or not 0 <= self.eos_token_id < self.vocab_size
            or self.pad_token_id == self.eos_token_id
        ):
            raise ValueError("invalid special token IDs")
        config = SimpleNamespace(
            **asdict(self),
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            mla_use_nope=True,
            mla_use_output_gate=True,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            latent_moe_use_norm=True,
            routed_scaling_factor=1.0,
            moe_router_activation_func="sigmoid",
            moe_renormalize=True,
            num_expert_group=1,
            topk_group=1,
            is_mla=True,
            _attn_implementation="eager",
            linear_attn_config=dict(
                short_conv_kernel_size=4,
                head_dim=self.kda_head_dim,
                num_heads=self.num_attention_heads,
                use_full_rank_gate=True,
                gate_lower_bound=-5.0,
            ),
        )
        config.is_kda_layer = lambda i: (i + 1) % 4 != 0 and i != self.num_hidden_layers - 1
        return config


class MiniKimiK3ForCausalLM(nn.Module):
    source_revision = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
    training_ready = True

    def __init__(self, config: MiniKimiK3Config):
        super().__init__()
        self.config = config
        if isinstance(config.vision_config, dict):
            config.vision_config = KimiVisionConfig(**config.vision_config)
        official = config.upstream_config()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            KimiDecoderLayer(official, i) for i in range(config.num_hidden_layers)
        )
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # No preceding residual block exists at the first attention sublayer.
        self.layers[0].self_attention_res_norm.requires_grad_(False)
        self.layers[0].self_attention_res_proj.requires_grad_(False)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if "res_proj.weight" in name:
                    p.zero_()
        self.vision = KimiVision(config.vision_config) if config.vision_config is not None else None
        if self.vision is not None and self.vision.config.output_size != config.hidden_size:
            raise ValueError("MoonViT merger output must match language hidden width")

        self.mtp = KimiMTP(config) if config.mtp_enabled else None
        if self.mtp is not None:
            self.mtp.apply(self._initialize)
            with torch.no_grad():
                for name, p in self.mtp.named_parameters():
                    if "res_proj.weight" in name:
                        p.zero_()
            if config.mtp_loss_coef == 0:
                self.mtp.requires_grad_(False)

        if config.router_fp32:
            from .routing import FP32KimiGate

            for module in self.modules():
                if isinstance(module, KimiMoEGate):
                    module.__class__ = FP32KimiGate
        if config.qat_scheme not in {"bf16", "mxfp4-mxfp8-v1"}:
            raise ValueError("unsupported Kimi QAT recipe")
        if config.qat_scheme != "bf16":
            from minifrontier.training.kimi_qat import configure

            self.qat_recipe = configure(self)

    @torch.no_grad()
    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding):
                module.weight[self.config.pad_token_id].zero_()
        if isinstance(module, KimiDeltaAttention):
            module.A_log.zero_()
            dt = torch.empty_like(module.dt_bias).uniform_(-6.907755, -2.302585).exp()
            module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if isinstance(module, KimiMoEGate):
            module.e_score_correction_bias.zero_()
            module.e_score_correction_bias.requires_grad_(False)

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
    ):
        labels = validate_batch(input_ids, self.config, attention_mask, labels)
        b, length = input_ids.shape
        past = 0 if cache is None else cache.length
        if cache is not None:
            if self.training or torch.is_grad_enabled() or labels is not None:
                raise ValueError("cached inference requires eval, no_grad and no labels")
            if attention_mask is not None and not attention_mask.bool().all():
                raise ValueError("cached batches must be unpadded")
            if past + length > self.config.max_position_embeddings:
                raise ValueError("cached sequence exceeds context")
            if past and media:
                raise ValueError("complete image spans must be in the initial prefill")
            cache.prepare(self, input_ids)
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if h.shape != (*input_ids.shape, self.config.hidden_size):
            raise ValueError("inputs_embeds shape differs from language inputs")
        if media is not None and self.vision is None:
            raise ValueError("this checkpoint has no native vision tower")
        if self.vision is not None:
            h, image_mask = insert_media_embeddings(
                input_ids, h, media, self.vision, self.config.image_token_id
            )
            if labels is not None and labels[image_mask].ne(-100).any():
                raise ValueError("image features must not be CE targets")
        blocks = h.new_zeros(b * length, 0, self.config.hidden_size)
        # Right padding is future-only and cannot influence any supervised token,
        # including the recurrent KDA state. Never flatten independent sequences.
        allowed = (
            torch.arange(past + length, device=h.device)[None]
            <= torch.arange(past, past + length, device=h.device)[:, None]
        )
        causal = torch.zeros_like(allowed, dtype=h.dtype).masked_fill(~allowed, float("-inf"))[
            None, None
        ]
        for layer in self.layers:
            mask = None if layer.is_linear_attn else causal
            if self.training and self.config.gradient_checkpointing:
                h, blocks = checkpoint(
                    layer, h, attention_mask=mask, block_residual=blocks, use_reentrant=False
                )
            else:
                h, blocks = layer(
                    h, attention_mask=mask, block_residual=blocks, past_key_values=cache
                )
        h = _apply_attn_res(
            h.reshape(-1, self.config.hidden_size),
            blocks,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(b, length, -1)
        features = self.norm(h)
        logits = self.lm_head(features) if return_logits else None
        loss = None
        if labels is not None:
            loss = (
                causal_lm_loss(logits, labels)
                if logits is not None
                else chunked_linear_ce(features, self.lm_head.weight, labels)
            )
        if cache is not None:
            cache.commit(length)
        lm_loss = loss
        mtp_loss = None
        mtp_tokens = 0
        if self.mtp is not None and labels is not None and self.config.mtp_loss_coef > 0:
            shifted, targets, valid = shifted_batch(
                input_ids,
                labels,
                vocab_size=self.config.vocab_size,
                image_token_id=self.config.image_token_id,
                attention_mask=attention_mask,
            )
            predicted = self.mtp(features, self.embed_tokens(shifted), blocks, valid)
            mtp_loss = chunked_linear_ce(predicted, self.lm_head.weight, targets, shift=False)
            mtp_tokens = int(targets.ne(-100).sum())
            loss = loss + self.config.mtp_loss_coef * mtp_loss
        return CausalLMOutput(
            logits,
            loss,
            lm_loss,
            hidden_states=features if return_hidden else None,
            mtp_loss=mtp_loss,
            mtp_tokens=mtp_tokens,
        )
