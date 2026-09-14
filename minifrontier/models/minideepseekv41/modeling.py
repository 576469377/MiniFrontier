"""MiniDeepSeek-V4.1: CED, CSA2, Single-Pass mHC and conditional Engram.

Architecture: official DeepSeek-V4.1-Flash dba1be0, MIT. Mini dimensions and
random initialization are local choices. The sparse indexer KL is a training
adaptation, not a released upstream trainer. Backbone pretraining has no MTP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.common import CausalLMOutput, validate_batch
from minifrontier.models.deepseek_v41_layers import CSA2Attention, CSA2State, SinglePassHC
from minifrontier.models.minideepseekv4.expert import TrainingGate
from minifrontier.models.minideepseekv4.upstream_layers import MoE, RMSNorm
from minifrontier.models.minideepseekv4.vision import DeepSeekVision, DeepSeekVisionConfig
from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce

from .engram import Engram, compressed_token_map, table_layout


@dataclass
class MiniDeepSeekV41Config:
    vocab_size: int = 65536
    dim: int = 512
    n_layers: int = 12
    n_encoder_layers: int = 6
    n_heads: int = 8
    head_dim: int = 64
    rope_head_dim: int = 16
    q_lora_rank: int = 128
    o_groups: int = 2
    o_lora_rank: int = 128
    moe_inter_dim: int = 256
    n_routed_experts: int = 24
    n_activated_experts: int = 2
    n_shared_experts: int = 1
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1)
    kv_source_layers: tuple[int, ...] = (2, 4, 6)
    index_source_layers: tuple[int, ...] = (2, 4, 6, 8, 10)
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 16
    index_candidate_source_layer: int = 6
    index_candidate_topk_blocks: int = 16
    index_candidate_block_size: int = 8
    hierarchical_indexing: bool = False
    attention_chunk_size: int = 64
    indexer_loss_coef: float = 1.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-20
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    route_scale: float = 1.5
    sequence_balance_coef: float = 1e-4
    swiglu_limit: float = 10.0
    initializer_range: float = 0.02
    expert_execution: str = "batched"
    engram_layer_ids: tuple[int, ...] = (1, 4)
    engram_vocab_size: int = 4096
    engram_n_heads: int = 2
    engram_head_dim: int = 32
    engram_max_ngram_size: int = 4
    engram_pad_id: int = 2
    pad_token_id: int = 0
    eos_token_id: int = 2
    gradient_checkpointing: bool = True
    forbidden_action_ids: tuple[int, ...] = (0, 1)
    mtp_enabled: bool = False
    mtp_loss_coef: float = 0.0
    vision_config: DeepSeekVisionConfig | None = None

    def upstream_config(self):
        self.validate()
        return SimpleNamespace(**asdict(self))

    def validate(self) -> None:
        if self.mtp_enabled:
            raise ValueError("V4.1 backbone pretraining omits MTP; DSpark is trained separately")
        if not 0 < self.n_encoder_layers < self.n_layers:
            raise ValueError("CED requires nonempty encoder and decoder")
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError("one compression ratio is required per layer")
        if any(r not in (0, 2) for r in self.compress_ratios[: self.n_encoder_layers]) or any(
            r != 1 for r in self.compress_ratios[self.n_encoder_layers :]
        ):
            raise ValueError("CED uses SWA/2:1 in encoder and 1:1 in decoder")
        if self.n_encoder_layers not in self.kv_source_layers or not set(
            self.kv_source_layers
        ) <= set(self.index_source_layers):
            raise ValueError("decoder begins in Full mode; every Full source owns an indexer")
        if any(
            i < 0 or i >= self.n_layers
            for i in (*self.kv_source_layers, *self.index_source_layers, *self.engram_layer_ids)
        ):
            raise ValueError("source or Engram layer outside backbone")
        if any(self.compress_ratios[i] == 0 for i in self.index_source_layers):
            raise ValueError("pure SWA layers do not own global indexers")
        owner_ratio = 0
        for i, ratio in enumerate(self.compress_ratios):
            if i in self.kv_source_layers:
                owner_ratio = ratio
            if ratio and ratio != owner_ratio:
                raise ValueError("CSA2 consumer must follow a compatible Full source")
        if (
            self.hierarchical_indexing
            and self.index_candidate_source_layer != self.n_encoder_layers
        ):
            raise ValueError("hierarchical candidate owner must be first decoder Full layer")
        if (
            self.n_heads % self.o_groups
            or self.rope_head_dim % 2
            or self.rope_head_dim > min(self.head_dim, self.index_head_dim)
        ):
            raise ValueError("invalid attention head dimensions")
        if not 1 <= self.n_activated_experts <= self.n_routed_experts or self.n_shared_experts != 1:
            raise ValueError("invalid MoE expert counts")
        if self.engram_max_ngram_size < 2 or self.engram_vocab_size < 2:
            raise ValueError("Engram requires at least bigrams and two buckets")
        if (
            min(
                self.dim,
                self.attention_chunk_size,
                self.index_topk,
                self.window_size,
                self.hc_mult,
                self.hc_sinkhorn_iters,
            )
            < 1
        ):
            raise ValueError("model dimensions and attention capacities must be positive")
        if not 0 <= self.engram_pad_id < self.vocab_size:
            raise ValueError("Engram pad ID outside tokenizer vocabulary")


class V41Gate(TrainingGate):
    """V4.1 has content routing on every layer, with separate image correction."""

    def forward(self, x, input_ids=None):
        with torch.autocast(x.device.type, enabled=False):
            weights, ids = super().forward(x.float(), input_ids)
            if self.topk == 1:
                # Official norm_topk_prob normalization applies only for top-k > 1.
                scores = torch.nn.functional.softplus(
                    torch.nn.functional.linear(x.float(), self.weight.float())
                ).sqrt()
                weights = scores.gather(-1, ids.long()) * self.route_scale
        return weights, ids


class Block(nn.Module):
    def __init__(self, config: MiniDeepSeekV41Config, layer_id: int, layout: dict[int, list[int]]):
        super().__init__()
        self.attn = CSA2Attention(config, layer_id)
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.hc_attn, self.hc_ffn = SinglePassHC(config), SinglePassHC(config)
        args = SimpleNamespace(
            **asdict(config), n_hash_layers=0, score_func="sqrtsoftplus", expert_dtype=None
        )
        self.ffn = MoE(layer_id, args)
        self.ffn.gate = V41Gate(layer_id, args)
        self.engram = Engram(config, layer_id, layout[layer_id]) if layer_id in layout else None
        self.sequence_balance_enabled = config.sequence_balance_coef > 0

    def forward(self, h, incoming_pre, state, input_ids, valid, encoder_hidden):
        if self.engram is not None:
            text_mask = valid & input_ids.lt(self.engram.config.vocab_size)
            safe_ids = input_ids.masked_fill(~text_mask, self.engram.config.pad_token_id)
            h = self.engram(h, safe_ids, text_mask)
        pre, post, comb = self.hc_attn.coefficients(h)
        attention, state, kl = self.attn(
            self.attn_norm(self.hc_attn.mix(h, incoming_pre)), state, valid, encoder_hidden
        )
        h = self.hc_attn.combine(attention, h, post, comb)
        ffn_pre, post, comb = self.hc_ffn.coefficients(h)
        gate = cast(V41Gate, self.ffn.gate)
        gate.valid_mask = valid
        gate.sequence_balance_enabled = self.sequence_balance_enabled
        output = self.ffn(self.ffn_norm(self.hc_ffn.mix(h, pre)), input_ids)
        h = self.hc_ffn.combine(output, h, post, comb)
        return h, ffn_pre, state, kl, self.ffn.gate.sequence_loss


class MiniDeepSeekV41ForCausalLM(nn.Module):
    source_revision = "dba1be0a40aa45a94ad051997016db3960a90277"
    training_ready = True
    cache_implementation = "exact-prefix-recomputation"

    def __init__(self, config: MiniDeepSeekV41Config, *, training_phase="sparse_pretrain"):
        super().__init__()
        if isinstance(config.vision_config, dict):
            config.vision_config = DeepSeekVisionConfig(**config.vision_config)
        config.validate()
        self.config, self.indexer_loss_enabled = config, True
        self.embed = nn.Embedding(config.vocab_size, config.dim, config.pad_token_id)
        layout = table_layout(config)
        self.layers = nn.ModuleList(Block(config, i, layout) for i in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # This shared tower has identical operations to dba1be0 inference/vision.py.
        self.vision = DeepSeekVision(config.vision_config) if config.vision_config else None
        if self.vision is not None:
            assert config.vision_config is not None
            if config.vision_config.output_size != config.dim:
                raise ValueError("vision projector width must match CED backbone")
            self.image_start = nn.Parameter(torch.zeros(config.dim))
            self.image_end = nn.Parameter(torch.zeros(config.dim))
            self.image_newline = nn.Parameter(torch.zeros(config.dim))
            self.image_pad = nn.Parameter(torch.zeros(config.dim))
        self._initialize()
        if config.expert_execution != "loop":
            from minifrontier.models.grouped_experts import configure

            configure(self, config.expert_execution)
        self.configure_training_phase(training_phase)

    @torch.no_grad()
    def _initialize(self):
        for name, parameter in self.named_parameters():
            if name.endswith((".q_weight", ".k_weight")):
                parameter.fill_(1)
            elif parameter.ndim >= 2:
                nn.init.normal_(parameter, std=self.config.initializer_range)
            elif name.endswith("weight"):
                parameter.fill_(1)
            elif name.endswith("scale"):
                parameter.fill_(0.01)
            elif not name.endswith(".base"):
                parameter.zero_()
        self.embed.weight[self.config.pad_token_id].zero_()

    def configure_training_phase(self, phase):
        if phase not in {"sparse_pretrain", "sparse_cpt", "posttrain"}:
            raise ValueError("V4.1 trains sparse attention from scratch without dense warmup")
        self.training_phase = phase

    def bind_tokenizer(self, tokenizer):
        mapping = compressed_token_map(tokenizer, self.config.vocab_size)
        for raw_layer in self.layers:
            layer = cast(Block, raw_layer)
            if layer.engram is not None:
                layer.engram.bind_token_map(mapping)

    def optimizer_metadata(self) -> dict[str, dict[str, Any]]:
        roles = {}
        for name, parameter in self.named_parameters():
            if name in {"embed.weight", "head.weight"} or name.endswith(".engram.embed.weight"):
                roles[name] = {"kind": "sinkhorn", "lr_scale": 5.0 if ".engram." in name else 1.0}
            elif parameter.ndim >= 2 and not name.endswith((".q_weight", ".k_weight")):
                role: dict[str, Any] = {"kind": "muon"}
                if name.endswith(".indexer.wq_b.weight"):
                    role["heads"] = self.config.index_n_heads
                elif name.endswith(".attn.wq_b.weight"):
                    role["heads"] = self.config.n_heads
                elif name.endswith(".indexer.wk.weight"):
                    role["heads"] = 1
                elif name.startswith("vision.") and name.endswith(".attn.wqkv.weight"):
                    assert self.config.vision_config is not None
                    width = self.config.vision_config.hidden_size
                    head_dim = width // self.config.vision_config.num_heads
                    role["blocks"] = [(i, i + head_dim) for i in range(0, 2 * width, head_dim)] + [
                        (2 * width, 3 * width)
                    ]
                roles[name] = role
            else:
                roles[name] = {"kind": "adamw"}
                if name.endswith((".q_weight", ".k_weight")):
                    roles[name]["weight_decay_norm"] = True
        return roles

    def _embeddings(self, input_ids, media, inputs_embeds):
        image_mask = input_ids >= self.config.vocab_size
        if image_mask.any() and (
            self.vision is None or (input_ids >= self.config.vocab_size + 5).any()
        ):
            raise ValueError("invalid V4.1 image sentinel IDs")
        safe_ids = input_ids.masked_fill(image_mask, self.config.pad_token_id)
        embeddings = self.embed(safe_ids) if inputs_embeds is None else inputs_embeds
        coverage = torch.zeros_like(image_mask)
        if media:
            if self.vision is None:
                raise ValueError("this configuration has no vision encoder")
            embeddings = embeddings.clone()
            params = torch.stack(
                (
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                )
            )
            for span in media:
                batch, start = span["batch_index"], span["start"]
                types = span["types"].to(input_ids.device)
                end = start + types.numel()
                if (
                    not 0 <= batch < input_ids.shape[0]
                    or not 0 <= start < end <= input_ids.shape[1]
                ):
                    raise ValueError("image span outside language sequence")
                if coverage[batch, start:end].any() or not torch.equal(
                    input_ids[batch, start:end], self.config.vocab_size + types
                ):
                    raise ValueError("image span sentinel mismatch or overlap")
                if (
                    types[0] != 0
                    or types[-1] != 4
                    or types.eq(0).sum() != 1
                    or types.eq(4).sum() != 1
                ):
                    raise ValueError("image span needs complete unique boundaries")
                features = self.vision(
                    span["patches"].to(input_ids.device), span["n_vit_h"], span["n_vit_w"]
                )
                perm = span.get(
                    "perm", torch.arange(features.shape[0], device=input_ids.device)
                ).to(input_ids.device)
                if features.shape[0] != types.eq(2).sum() or not torch.equal(
                    perm.sort().values, torch.arange(features.shape[0], device=perm.device)
                ):
                    raise ValueError("image feature count/permutation mismatch")
                block = params[types].clone()
                block[types == 2] = features[perm].to(block.dtype)
                embeddings[batch, start:end] = block.to(embeddings.dtype)
                coverage[batch, start:end] = True
        if not torch.equal(coverage, image_mask):
            raise ValueError("image sentinel missing corresponding pixels")
        return safe_ids, embeddings, image_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        *,
        inputs_embeds=None,
        media=None,
        cache=None,
        return_hidden=False,
        return_logits=True,
        return_taps=False,
    ):
        if cache is not None:
            return cache.forward(
                self,
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                media=media,
                return_hidden=return_hidden,
                return_logits=return_logits,
                return_taps=return_taps,
            )
        safe_ids, embeddings, image_mask = self._embeddings(input_ids, media, inputs_embeds)
        labels = validate_batch(safe_ids, self.config, attention_mask, labels)
        if labels is not None and labels[image_mask].ne(-100).any():
            raise ValueError("image spans cannot be language CE targets")
        valid = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        h = embeddings.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        pre = torch.zeros_like(h[..., 0], dtype=torch.float32)
        pre[..., 0] = 1
        state, encoder_hidden = CSA2State(), None
        index_losses, sequence_losses, taps = [], [], []
        for i, raw_layer in enumerate(self.layers):
            layer = cast(Block, raw_layer)
            if i == self.config.n_encoder_layers:
                encoder_hidden = layer.attn_norm(SinglePassHC.mix(h, pre))
            layer.attn.indexer_loss_enabled = self.indexer_loss_enabled
            if return_taps and i >= self.config.n_layers - 3:
                taps.append(h.mean(-2))
            if self.training and self.config.gradient_checkpointing:
                h, pre, state, kl, seq = checkpoint(
                    layer, h, pre, state, input_ids, valid, encoder_hidden, use_reentrant=False
                )
            else:
                h, pre, state, kl, seq = layer(h, pre, state, input_ids, valid, encoder_hidden)
            if layer.attn.indexer is not None:
                index_losses.append(kl)
            sequence_losses.append(seq)
        features = self.norm(SinglePassHC.mix(h, pre))
        logits = self.head(features) if return_logits else None
        lm = (
            None
            if labels is None
            else causal_lm_loss(logits, labels)
            if logits is not None
            else chunked_linear_ce(features, self.head.weight, labels)
        )
        kl = torch.stack(index_losses).mean() if index_losses else features.new_zeros(())
        seq = torch.stack(sequence_losses).mean()
        loss = (
            None
            if lm is None
            else lm + self.config.sequence_balance_coef * seq + self.config.indexer_loss_coef * kl
        )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm,
            aux_loss=seq,
            indexer_loss=kl,
            hidden_states=features if return_hidden else None,
            multistream_hidden=h if return_hidden else None,
            tapped_hidden_states=tuple(taps) if return_taps else None,
            index_query_tokens=int(valid.sum()) * len(index_losses),
        )
