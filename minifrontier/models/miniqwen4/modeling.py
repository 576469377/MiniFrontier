"""New source-only Qwen text backbone, separate from the retired experiment.

The text stack retains the published sparse computation; the LM adapter adds
the report's full-attention stage, untied head and source router auxiliary loss.
MTP and native multimodal training remain pending. QSA objectives implement
indexer-only dense distillation and joint sparse CPT. The cache adapter
supports unpadded no-grad text inference, not the entire HF generation API.
No historical model is used as a fallback for missing features.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.cache_utils import cache_transaction
from minifrontier.models.common import insert_media_embeddings
from minifrontier.strict_types import validate_dataclass_payload
from minifrontier.training.mtp import shifted_batch
from minifrontier.training.qwen_balance import normalized_router_loss

from .cache import MiniQwen4Cache
from .mtp import QwenMTP
from .qsa import indexer_kl_loss
from .upstream_core import Qwen4ExpTextGatedDeltaNet, Qwen4ExpTextGatedResidual
from .upstream_decoder import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextExperts,
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextSparseMoeBlock,
)
from .upstream_ple import Qwen4ExpTextPLELayer, Qwen4ExpTextRMSNorm
from .vision import QwenVision, QwenVisionConfig


@dataclass
class MiniQwen4Config:
    """Capacity fields retain upstream names; integration choices are explicit."""

    vocab_size: int = 65536
    hidden_size: int = 512
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 4096
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 12
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    num_experts: int = 64
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 192
    shared_expert_intermediate_size: int = 192
    hc_count: int = 4
    hc_lowrank: int = 128
    ple_layer_ids: tuple[int, ...] = (2,)
    ple_embed_dim: int = 512
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 2
    ngram_vocab_size_base: int = 32768
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 32
    indexer_budget: int = 512
    indexer_compress_ratio: int = 4
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    output_gate_type: str = "sigmoid"
    pad_token_id: int = 0
    eos_token_id: int = 2
    gradient_checkpointing: bool = True
    forbidden_action_ids: tuple[int, ...] = (0, 1)
    mtp_enabled: bool = False
    mtp_loss_coef: float = 0.1
    vision_config: QwenVisionConfig | None = None
    image_token_id: int = 7

    def upstream_config(self) -> SimpleNamespace:
        validate_dataclass_payload(type(self), asdict(self))
        values = asdict(self)
        non_dimensions = {"seed", "pad_token_id", "eos_token_id"}
        for name, value in values.items():
            if type(value) is int and name not in non_dimensions and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("invalid seed or padding token")
        if not 0 <= self.eos_token_id < self.vocab_size or self.eos_token_id == self.pad_token_id:
            raise ValueError("EOS must be in vocabulary and distinct from padding")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Q heads must divide into complete KV groups")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("GDN value heads must divide into complete key-head groups")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("expert top-k exceeds expert count")
        if self.ngram_size != 3 or self.indexer_kv_heads != 1:
            raise ValueError("the pinned source requires ngram_size=3 and one indexer KV head")
        if self.ple_embed_dim % (2 * self.heads_per_ngram):
            raise ValueError("PLE embedding dimension must divide across all n-gram heads")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer budget must contain complete compressed blocks")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if not 0 < self.partial_rotary_factor <= 1 or rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("rotary dimension must be positive and even")
        if rotary_dim > self.indexer_head_dim:
            raise ValueError("rotary dimension exceeds indexer head dimension")
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.rms_norm_eps, self.initializer_range, self.rope_theta)
        ):
            raise ValueError("normalization, initialization and RoPE constants must be positive")
        if self.output_gate_type not in {"silu", "sigmoid"}:
            raise ValueError("unsupported upstream output gate")
        layer_types = [
            "qwen_sparse_attention"
            if (i + 1) % self.full_attention_interval == 0
            else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]
        if tuple(sorted(set(self.ple_layer_ids))) != self.ple_layer_ids:
            raise ValueError("PLE layer IDs must be sorted and unique (upstream one-based IDs)")
        for index in self.ple_layer_ids:
            if (
                not 1 <= index <= self.num_hidden_layers
                or layer_types[index - 1] != "linear_attention"
            ):
                raise ValueError("PLE must refer to a GDN layer")
        values.update(
            layer_types=layer_types,
            hidden_act="silu",
            intermediate_size=self.moe_intermediate_size,
            attention_bias=False,
            attention_dropout=0.0,
            norm_topk_prob=True,
            _attn_implementation="eager",
            rope_parameters={
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                # Same interleaved frequency allocation, scaled to the actual
                # rotary dimension: 16 dims -> [3,3,2] instead of [11,11,10].
                "mrope_section": [
                    (rotary_dim // 2 + 2) // 3,
                    (rotary_dim // 2 + 1) // 3,
                    rotary_dim // 2 // 3,
                ],
            },
        )
        return SimpleNamespace(**values)


class MiniQwen4TextModel(nn.Module):
    """Source-derived text stack used by the educational causal LM adapter."""

    source_revision = "4177486a9f199bd7be520eff14431071d5d41ec5"
    training_ready = True

    def __init__(self, config: MiniQwen4Config, *, attention_mode: str = "sparse") -> None:
        super().__init__()
        self.config = config
        official = config.upstream_config()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen4ExpTextDecoderLayer(official, i) for i in range(config.num_hidden_layers)]
        )
        if attention_mode not in {"full", "sparse"}:
            raise ValueError("attention_mode must be full or sparse")
        self.attention_mode = attention_mode
        if attention_mode == "full":
            for index, raw_layer in enumerate(self.layers):
                layer = cast(Qwen4ExpTextDecoderLayer, raw_layer)
                if hasattr(layer, "self_attn"):
                    layer.self_attn.indexer = MiniQwen4StageIndexer(official, index)
                    layer.self_attn.indexer.requires_grad_(False)
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(official)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(official, use_combine=False)
        self.apply(self._initialize)

    @torch.no_grad()
    def _initialize(self, module: nn.Module) -> None:
        # Same initialization distributions as the pinned HF model; identical
        # seeded RNG consumption across different wrappers is not claimed.
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        if isinstance(module, Qwen4ExpTextGatedDeltaNet):
            module.dt_bias.fill_(1)
            module.A_log.uniform_(0.01, 16).log_()
        elif isinstance(module, Qwen4ExpTextRMSNorm):
            module.weight.zero_()
        elif isinstance(module, Qwen4ExpTextExperts):
            nn.init.normal_(module.gate_up_proj, std=std)
            nn.init.normal_(module.down_proj, std=std)
        elif isinstance(module, Qwen4ExpTextSparseMoeBlock):
            nn.init.normal_(module.gate.weight, std=std)
        if isinstance(module, Qwen4ExpTextPLELayer):
            module.conv1d.weight.zero_()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_router_logits: bool = False,
        cache: MiniQwen4Cache | None = None,
        indexer_phase: str | None = None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        ple_input_ids: Tensor | None = None,
        return_multistream: bool = False,
    ) -> (
        Tensor
        | tuple[Tensor, tuple[Tensor, ...]]
        | tuple[Tensor, tuple[Tensor, ...], Tensor]
        | tuple[Tensor, tuple[Tensor, ...], Tensor, Tensor]
    ):
        if indexer_phase is not None:
            if indexer_phase not in {"dense_distill", "sparse_cpt"} or not output_router_logits:
                raise ValueError("indexer phase requires router outputs and a supported CPT phase")
            if cache is not None:
                raise ValueError("indexer training does not support caches")
            expected_mode = "full" if indexer_phase == "dense_distill" else "sparse"
            if self.attention_mode != expected_mode:
                raise ValueError("indexer objective does not match model attention mode")
        if (
            input_ids.ndim != 2
            or input_ids.numel() == 0
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("input_ids must be a nonempty integer [batch, sequence] tensor")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input_ids must be on the model device")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("token IDs outside vocabulary")
        batch, length = input_ids.shape
        past_length = 0 if cache is None else cache.length
        if length + past_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds configured maximum")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (
            attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device
            or attention_mask.is_complex()
        ):
            raise ValueError("attention mask must match input shape and device")
        if bool(((attention_mask != 0) & (attention_mask != 1)).any()):
            raise ValueError("attention mask must be binary")
        valid = attention_mask.bool()
        if not bool(valid.any(-1).all()):
            raise ValueError("each sample must contain valid tokens")
        # Initial public adapter supports unpadded/right-padded text only.
        if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("source text adapter currently requires right padding")
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("cache requires eval mode and no_grad/inference_mode")
            if not bool(valid.all()):
                raise ValueError("cached text inference currently requires unpadded batches")
            cache.prepare(
                self,
                len(self.layers),
                batch,
                input_ids.device,
                (
                    self.attention_mode,
                    self.embed_tokens.weight.dtype,
                    torch.get_autocast_dtype(input_ids.device.type)
                    if torch.is_autocast_enabled(input_ids.device.type)
                    else None,
                ),
            )
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if h.shape != (*input_ids.shape, self.config.hidden_size):
            raise ValueError("inputs_embeds must match Qwen input token dimensions")
        total_length = past_length + length
        positions = position_ids
        if cache is not None and past_length and cache.position_ids is not None:
            if positions is not None:
                raise ValueError("cached continuation positions follow the stored mRoPE prefix")
            next_position = cache.position_ids.amax(dim=(0, 2)) + 1
            appended = (
                next_position[None, :, None] + torch.arange(length, device=h.device)[None, None, :]
            )
            positions = torch.cat((cache.position_ids, appended.expand(3, -1, -1)), -1)
        if positions is None:
            positions = (
                torch.arange(total_length, device=h.device).view(1, 1, -1).expand(3, batch, -1)
            )
        elif (
            positions.shape != (3, batch, total_length)
            or positions.dtype != torch.long
            or (positions < 0).any()
        ):
            raise ValueError("mRoPE positions must be nonnegative int64 [3,B,total_length]")
        rotary = self.rotary_emb(h, positions)
        if cache is not None:
            cache.position_ids = positions.clone()
        key_positions = torch.arange(total_length, device=h.device)
        query_positions = key_positions[-length:]
        visible = key_positions[None, :] <= query_positions[:, None]
        key_valid = torch.cat((valid.new_ones(batch, past_length), valid), dim=1)
        visible = visible[None, None] & key_valid[:, None, None, :]
        mask = torch.zeros(batch, 1, length, total_length, device=h.device, dtype=h.dtype)
        mask.masked_fill_(~visible, torch.finfo(h.dtype).min)
        ple_ids = input_ids if ple_input_ids is None else ple_input_ids
        if ple_ids.shape != input_ids.shape or not torch.equal(ple_ids, input_ids):
            raise ValueError("PLE must receive the original input IDs before visual replacement")
        ple_ids = ple_ids.masked_fill(~valid, self.config.eos_token_id)
        h = h.repeat(1, 1, self.config.hc_count)
        routers = []
        indexer_losses = []
        for layer in self.layers:
            kwargs = dict(
                position_embeddings=rotary,
                attention_mask=mask,
                conv_mask=valid,
                past_key_values=cache,
                ple_input_ids=ple_ids,
            )
            if output_router_logits:
                if self.training and self.config.gradient_checkpointing:
                    h, logits, kl = checkpoint(
                        _decoder_with_router,
                        layer,
                        h,
                        use_reentrant=False,
                        indexer_phase=indexer_phase,
                        **kwargs,
                    )
                else:
                    h, logits, kl = _decoder_with_router(
                        layer, h, indexer_phase=indexer_phase, **kwargs
                    )
                routers.append(logits)
                if indexer_phase is not None and hasattr(layer, "self_attn"):
                    indexer_losses.append(kl)
            elif self.training and self.config.gradient_checkpointing:
                h = checkpoint(layer, h, use_reentrant=False, **kwargs)
            else:
                h = layer(h, **kwargs)
        multistream = h
        h = self.hyper_connection_mixer(h)
        if cache is not None:
            cache.commit(length)
        if return_multistream:
            return (
                h,
                tuple(routers),
                torch.stack(indexer_losses).mean() if indexer_losses else h.new_zeros(()),
                multistream,
            )
        if indexer_phase is not None:
            if not indexer_losses:
                raise ValueError("indexer training requires at least one attention layer")
            # Cross-layer averaging is an explicit local integration choice;
            # each layer's token reduction follows report equations 18/20.
            return h, tuple(routers), torch.stack(indexer_losses).mean()
        return (h, tuple(routers)) if output_router_logits else h


class MiniQwen4StageIndexer(Qwen4ExpTextQSAIndexer):
    """Report base-pretraining policy: retain full causal attention.

    No Q/K/V, gating, normalization or attention computation is replaced.
    Indexer parameters retain their upstream layout but are dormant/frozen in
    this stage. The sparse inference stack continues using the original class.
    """

    sparse_enabled = False

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        if self.sparse_enabled:
            return super().forward(
                hidden_states, position_embeddings, attention_mask, past_key_values
            )
        return (
            torch.zeros_like(attention_mask)
            if attention_mask.is_floating_point()
            else torch.ones_like(attention_mask)
        )


def _decoder_with_router(layer, hidden_states, *, indexer_phase=None, **kwargs):
    """Capture the already-computed router output; no duplicate routing pass.

    A temporary hook is local to each invocation, including checkpoint replay.
    Return the captured tensor as a checkpoint output so its gradient is retained.
    """
    captured = []
    captured_kl = []
    handle = layer.mlp.gate.register_forward_hook(
        lambda module, args, output: captured.append(output[0])
    )
    attention_handle = None
    if indexer_phase is not None and hasattr(layer, "self_attn"):

        def collect_kl(module, args, call_kwargs, output):
            captured_kl.append(
                indexer_kl_loss(
                    module.indexer,
                    args[0],
                    args[1],
                    call_kwargs["attention_mask"],
                    output[1],
                    kwargs["conv_mask"],
                    selected_only=indexer_phase == "sparse_cpt",
                )
            )

        attention_handle = layer.self_attn.register_forward_hook(collect_kl, with_kwargs=True)
    try:
        result = layer(hidden_states, **kwargs)
    finally:
        handle.remove()
        if attention_handle is not None:
            attention_handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one router call per decoder invocation")
    if attention_handle is not None and len(captured_kl) != 1:
        raise RuntimeError("expected exactly one QSA loss per attention layer invocation")
    kl = captured_kl[0] if captured_kl else hidden_states.new_zeros((), dtype=torch.float32)
    return result, captured[0], kl


@dataclass
class MiniQwen4LMOutput:
    logits: Tensor | None
    loss: Tensor | None
    lm_loss: Tensor | None
    aux_loss: Tensor
    router_logits: tuple[Tensor, ...]
    indexer_loss: Tensor | None = None
    hidden_states: Tensor | None = None
    multistream_hidden: Tensor | None = None
    mtp_loss: Tensor | None = None
    mtp_tokens: int = 0
    mtp_aux_loss: Tensor | None = None
    mtp_aux_count: int = 0


class MiniQwen4ForCausalLM(nn.Module):
    """Source text model + the published untied LM head and router auxiliary loss.

    Text pretraining, indexer phase transitions and downstream stages are
    executable through the shared trainer. MTP and native multimodal training
    remain outside the implemented text lifecycle.
    """

    training_ready = True

    def __init__(
        self,
        config: MiniQwen4Config,
        *,
        router_aux_loss_coef: float = 0.001,
        training_phase: str = "dense_pretrain",
        indexer_kl_coef: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 <= router_aux_loss_coef < float("inf"):
            raise ValueError("router auxiliary coefficient must be finite and nonnegative")
        if not 0 < indexer_kl_coef < float("inf"):
            raise ValueError("indexer KL coefficient must be finite and positive")
        self.config = config
        self.model = MiniQwen4TextModel(config, attention_mode="full")
        if isinstance(config.vision_config, dict):
            config.vision_config = QwenVisionConfig(**config.vision_config)
        self.vision = QwenVision(config.vision_config) if config.vision_config is not None else None
        if self.vision is not None and self.vision.config.output_size != config.hidden_size:
            raise ValueError("Qwen vision merger output must match language width")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        self.router_aux_loss_coef = router_aux_loss_coef
        self.indexer_kl_coef = indexer_kl_coef
        self.indexer_loss_enabled = True
        self.mtp = QwenMTP(config, self.model._initialize) if config.mtp_enabled else None
        self._configure_training_phase(training_phase)

    def _configure_training_phase(self, phase: str) -> None:
        if phase not in {"dense_pretrain", "dense_distill", "sparse_cpt"}:
            raise ValueError("unsupported MiniQwen4 training phase")
        if phase != "dense_pretrain" and not any(
            hasattr(layer, "self_attn") for layer in self.model.layers
        ):
            raise ValueError("QSA stages require at least one attention layer")
        self.training_phase = phase
        self.model.attention_mode = "sparse" if phase == "sparse_cpt" else "full"
        self.requires_grad_(phase != "dense_distill")
        for raw_layer in self.model.layers:
            layer = cast(Qwen4ExpTextDecoderLayer, raw_layer)
            if hasattr(layer, "self_attn"):
                indexer = cast(MiniQwen4StageIndexer, layer.self_attn.indexer)
                indexer.sparse_enabled = phase == "sparse_cpt"
                indexer.requires_grad_(phase != "dense_pretrain")
        if self.mtp is not None:
            self.mtp.configure_phase(phase)
        # A phase boundary must not reuse gradients from the previous objective.
        if self.mtp is not None and self.config.mtp_loss_coef == 0:
            self.mtp.requires_grad_(False)
        self.zero_grad(set_to_none=True)

    def transition_training_phase(self, phase: str) -> None:
        """Monotonic stages; caller must rebuild DDP and optimizer afterwards.

        Parameter objects/weights are preserved. A resume constructs the model
        with its saved phase rather than replaying transitions.
        """
        following = {"dense_pretrain": "dense_distill", "dense_distill": "sparse_cpt"}
        if following.get(self.training_phase) != phase:
            raise ValueError("expected dense_pretrain -> dense_distill -> sparse_cpt")
        self._configure_training_phase(phase)

    @cache_transaction
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        cache: MiniQwen4Cache | None = None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        ple_input_ids: Tensor | None = None,
        media=None,
        return_hidden=False,
        return_logits=True,
    ) -> MiniQwen4LMOutput:
        from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce

        if cache is not None and labels is not None:
            raise ValueError("cached LM inference does not accept training labels")
        if media and cache is not None and cache.length:
            raise ValueError("complete images must be in the initial prefill")
        if media is not None and self.vision is None:
            raise ValueError("this checkpoint has no native vision tower")
        if self.vision is not None:
            embeddings = (
                self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            )
            inputs_embeds, image_mask = insert_media_embeddings(
                input_ids, embeddings, media, self.vision, self.config.image_token_id
            )
            if labels is not None and labels[image_mask].ne(-100).any():
                raise ValueError("Qwen image placeholders must not be LM targets")
        if media and position_ids is None:
            from .processing import position_ids as media_positions

            position_ids = media_positions(input_ids, media)
        phase = (
            self.training_phase
            if self.training_phase != "dense_pretrain"
            and cache is None
            and self.indexer_loss_enabled
            else None
        )
        result = self.model(
            input_ids,
            attention_mask,
            output_router_logits=True,
            cache=cache,
            indexer_phase=phase,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ple_input_ids=ple_input_ids,
            return_multistream=return_hidden or (self.mtp is not None and labels is not None),
        )
        hidden, routers = result[:2]
        indexer_loss = result[2] if phase is not None else None
        logits = self.lm_head(hidden) if return_logits else None
        aux = normalized_router_loss(
            routers,
            self.config.num_experts,
            self.config.num_experts_per_tok,
            attention_mask,
            getattr(self, "router_window_frequency", None),
        )
        if not isinstance(aux, Tensor):
            raise RuntimeError("upstream router auxiliary loss did not return a tensor")
        lm_loss = None
        loss = None
        if labels is not None:
            if (
                labels.dtype != torch.int64
                or labels.shape != input_ids.shape
                or labels.device != input_ids.device
            ):
                raise ValueError("labels must be int64 and match input shape and device")
            if bool(((labels != -100) & ((labels < 0) | (labels >= self.config.vocab_size))).any()):
                raise ValueError("labels must be vocabulary IDs or -100")
            targets = labels
            if attention_mask is not None:
                targets = labels.masked_fill(~attention_mask.bool(), -100)
            lm_loss = (
                causal_lm_loss(logits, targets)
                if logits is not None
                else chunked_linear_ce(hidden, self.lm_head.weight, targets)
            )
            loss = lm_loss + self.router_aux_loss_coef * aux
        if indexer_loss is not None:
            if self.training_phase == "dense_distill":
                loss = self.indexer_kl_coef * indexer_loss
            elif loss is not None:
                loss = loss + self.indexer_kl_coef * indexer_loss
        mtp_loss, mtp_tokens = None, 0
        mtp_aux, mtp_aux_count = None, 0
        if self.mtp is not None and labels is not None and self.config.mtp_loss_coef > 0:
            shifted, targets, valid = shifted_batch(
                input_ids,
                labels,
                vocab_size=self.config.vocab_size,
                image_token_id=self.config.image_token_id,
                attention_mask=attention_mask,
            )
            positions = (
                position_ids
                if position_ids is not None
                else torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            predicted, _, mtp_router, mtp_kl = self.mtp(
                result[3], self.model.embed_tokens(shifted), valid, positions + 1
            )
            mtp_aux_count = int(valid.sum())
            mtp_aux = self.router_aux_loss_coef * normalized_router_loss(
                (mtp_router,),
                self.config.num_experts,
                self.config.num_experts_per_tok,
                valid,
                getattr(self.mtp, "router_window_frequency", None),
            )
            if self.training_phase == "dense_distill":
                mtp_aux = self.indexer_kl_coef * mtp_kl
            else:
                mtp_loss = chunked_linear_ce(
                    predicted, self.mtp.shared_head.head.weight, targets, shift=False
                )
                mtp_tokens = int(targets.ne(-100).sum())
                loss = loss + self.config.mtp_loss_coef * mtp_loss
                if self.training_phase == "sparse_cpt" and self.indexer_loss_enabled:
                    mtp_aux = mtp_aux + self.indexer_kl_coef * mtp_kl
            loss = loss + mtp_aux
        return MiniQwen4LMOutput(
            logits,
            loss,
            lm_loss,
            aux,
            routers,
            indexer_loss,
            hidden_states=hidden if return_hidden else None,
            multistream_hidden=result[3] if return_hidden else None,
            mtp_loss=mtp_loss,
            mtp_tokens=mtp_tokens,
            mtp_aux_loss=mtp_aux,
            mtp_aux_count=mtp_aux_count,
        )
