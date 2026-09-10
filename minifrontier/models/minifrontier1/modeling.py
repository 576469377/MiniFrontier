# Fusion reuses Kimi K3 and Apache-2.0 GR/vision primitives; see LICENSES and source-map.json.
"""Independent native four-stream MiniFrontier1 decoder and common causal LM interface."""

from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.cache_utils import cache_transaction
from minifrontier.models.common import CausalLMOutput, insert_media_embeddings, validate_batch
from minifrontier.training.losses import causal_lm_loss, chunked_linear_ce

from .configuration import MiniFrontier1Config
from .csa import CSA, Compressor
from .indexer import prefill_directory
from .kda import KDA, prefill_groups
from .lookup import NgramLookup
from .moe import LatentMoE, Router
from .mtp import MF1MTPBlock, mtp_targets
from .processing import token_metadata
from .qsa_mla import QSAMLA
from .residual import GatedResidual
from .vision import NativeVision


class MF1DecoderLayer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.attention_gr = GatedResidual(c)
        self.attention = {"kda": KDA, "csa4": CSA, "qsa_mla": QSAMLA}[kind](c)
        self.moe_gr = GatedResidual(c)
        self.moe = LatentMoE(c)

    def forward(self, residual, metadata, state=None, *, cache_output=True):
        h, weights = self.attention_gr.read(residual)
        if self.kind == "kda":
            update, state = self.attention(
                h,
                metadata["segment_ids"],
                state,
                cache_output=cache_output,
                layout=metadata.get("kda_prefill"),
            )
            loss, count = h.sum() * 0, 0
        else:
            update, state, loss, count = self.attention(
                h, metadata, state, cache_output=cache_output
            )
        residual = self.attention_gr.inject(residual, update, weights)
        h, weights = self.moe_gr.read(residual)
        update = self.moe(h, metadata.get("valid_token_indices"))
        return self.moe_gr.inject(residual, update, weights), state, loss, count


class MiniFrontier1ForCausalLM(nn.Module):
    source_revision = "minifrontier1-native-fusion-v1"
    training_ready = True  # executable reference; capacity/quality gates are separate

    def __init__(self, config: MiniFrontier1Config, training_phase="dense_pretrain"):
        super().__init__()
        self.config = config
        c = config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size, c.pad_token_id)
        self.layers = nn.ModuleList(MF1DecoderLayer(c, kind) for kind in c.attention_schedule)
        self.final_gr = GatedResidual(c, read_only=True)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lookup = NgramLookup(c) if c.lookup_enabled else None
        self.vision = NativeVision(c.vision_config)
        self.mtp = MF1MTPBlock(c) if c.mtp_enabled else None
        self.indexer_loss_enabled = True
        self.apply(self._initialize)
        # Residual-output scaling is applied once; GR's read/inject scaling is unchanged.
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith(
                    (
                        "attention.out.weight",
                        "core.o_proj.weight",
                        "moe.up.weight",
                        "shared.down.weight",
                    )
                ):
                    parameter.mul_((2 * c.num_hidden_layers) ** -0.5)
        self.set_phase(training_phase)

    @torch.no_grad()
    def _initialize(self, module):
        c = self.config
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv3d)):
            nn.init.normal_(module.weight, std=c.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(cast(torch.Tensor, module.bias))
        if isinstance(module, Router):
            nn.init.normal_(module.weight, std=c.initializer_range)
        if isinstance(module, KDA):
            module.core.A_log.zero_()
            # Choose retention directly under alpha=exp(-5*sigmoid(z)), not inverse-softplus.
            retention = torch.empty_like(module.core.dt_bias).uniform_(0.8, 0.99)
            module.core.dt_bias.copy_(torch.logit(-retention.log() / 5))
        if isinstance(module, Compressor):
            module.gate.weight.zero_()

    def set_phase(self, phase):
        if phase not in {"dense_pretrain", "dense_distill", "sparse_cpt"}:
            raise ValueError("invalid MF1 attention phase")
        self.training_phase = phase
        for name, p in self.named_parameters():
            p.requires_grad_(phase != "dense_distill" or ".indexer." in name)
        for layer in self.layers:
            if layer.kind != "kda":
                cast(Any, layer).attention.training_phase = phase
        if self.mtp is not None and self.config.mtp_loss_coef == 0:
            self.mtp.requires_grad_(False)

    @cache_transaction
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        *,
        inputs_embeds=None,
        media=None,
        segment_ids=None,
        position_ids=None,
        cache=None,
        return_hidden=False,
        return_logits=True,
        return_taps=False,
    ):
        labels = validate_batch(input_ids, self.config, attention_mask, labels)
        c = self.config
        past = 0 if cache is None else cache.length
        if cache is not None:
            if self.training or torch.is_grad_enabled() or labels is not None:
                raise ValueError("cached MF1 inference requires eval/no_grad/no labels")
            if (
                input_ids.eq(c.pad_token_id).any()
                or past + input_ids.shape[1] > c.max_position_embeddings
            ):
                raise ValueError("cached inference requires unpadded inputs within context")
            if past and (media or segment_ids is not None):
                raise ValueError("media and packed samples must be complete in initial prefill")
            if segment_ids is not None and any(row.unique().numel() > 1 for row in segment_ids):
                raise ValueError("cached generation requires one independent sample per batch row")
            cache.prepare(self, input_ids)
        metadata = token_metadata(
            input_ids,
            c,
            media,
            segment_ids=segment_ids,
            position_ids=position_ids,
            offset=past,
            position_base=cache.position_base if cache is not None else None,
        )
        valid_indices = metadata["segment_ids"].flatten().ge(0).nonzero().flatten()
        metadata["valid_token_indices"] = (
            valid_indices if len(valid_indices) != input_ids.numel() else None
        )
        if cache is None:
            metadata["kda_prefill"] = prefill_groups(metadata["segment_ids"])
            metadata["unpacked_prefill"] = metadata["kda_prefill"]["direct"]
            if self.training_phase == "dense_pretrain":
                metadata["prefill_directory"] = prefill_directory(metadata)
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if h.shape != (*input_ids.shape, c.hidden_size):
            raise ValueError("input embedding shape differs from decoder")
        if media:
            h, image_mask = insert_media_embeddings(
                input_ids, h, media, self.vision, c.image_token_id
            )
            if labels is not None and labels[image_mask].ne(-100).any():
                raise ValueError("visual embeddings cannot be CE targets")
        if labels is not None:
            labels = labels.clone().masked_fill(metadata["segment_ids"].lt(0), -100)
            labels[:, 1:] = labels[:, 1:].masked_fill(
                metadata["segment_ids"][:, 1:].ne(metadata["segment_ids"][:, :-1]), -100
            )
        residual = h.unsqueeze(-2).expand(*h.shape[:-1], c.hc_count, c.hidden_size)
        losses, counts, taps = [], [], []
        for i, layer in enumerate(self.layers):
            if self.lookup is not None and i == c.lookup_layer - 1:
                read = cast(MF1DecoderLayer, layer).attention_gr(residual)
                update, state = self.lookup(
                    read,
                    input_ids,
                    metadata["segment_ids"],
                    metadata["modality"],
                    cache.lookup if cache is not None else None,
                )
                residual = residual + update.unsqueeze(-2)
                if cache is not None:
                    cache.lookup = state
            if layer.kind != "kda":
                cast(Any, layer).attention.indexer_loss_enabled = self.indexer_loss_enabled
            if self.training and c.gradient_checkpointing:

                def run(r, layer=layer):
                    value, _, loss, count = layer(r, metadata, cache_output=False)
                    return value, loss, count

                residual, loss, count = checkpoint(run, residual, use_reentrant=False)
            else:
                residual, state, loss, count = layer(
                    residual,
                    metadata,
                    cache.layers[i] if cache is not None else None,
                    cache_output=cache is not None,
                )
                if cache is not None:
                    cache.layers[i] = state
            losses.append(loss)
            counts.append(count)
            if return_taps:
                taps.append(cast(MF1DecoderLayer, layer).attention_gr(residual))
        features = self.final_gr(residual)
        logits = self.lm_head(features) if return_logits else None
        lm_loss = (
            None
            if labels is None
            else causal_lm_loss(logits, labels)
            if logits is not None
            else chunked_linear_ce(features, self.lm_head.weight, labels)
        )
        indexer_loss = sum(losses) / max(1, sum(counts))
        mtp_loss, mtp_tokens = None, 0
        loss = lm_loss
        if self.training_phase == "dense_distill":
            loss = indexer_loss
        elif loss is not None:
            loss = loss + c.indexer_loss_coef * indexer_loss
            if self.mtp is not None and c.mtp_loss_coef > 0:
                shifted, targets = mtp_targets(input_ids, labels, metadata, c)
                predicted = self.mtp(residual, self.embed_tokens(shifted), metadata)
                mtp_loss = chunked_linear_ce(predicted, self.lm_head.weight, targets, shift=False)
                mtp_tokens = int(targets.ne(-100).sum())
                loss = loss + c.mtp_loss_coef * mtp_loss
        if cache is not None:
            cache.commit(input_ids.shape[1], metadata["position_ids"])
        result = CausalLMOutput(
            logits,
            loss,
            lm_loss,
            indexer_loss=indexer_loss,
            hidden_states=features if return_hidden else None,
            multistream_hidden=residual if return_hidden else None,
            mtp_loss=mtp_loss,
            mtp_tokens=mtp_tokens,
            tapped_hidden_states=tuple(taps) if return_taps else None,
        )
        result.index_query_tokens = sum(counts)
        return result
