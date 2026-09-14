# Copyright 2026 The Qwen Team and MiniFrontier contributors.
# SPDX-License-Identifier: Apache-2.0
"""Optional dense SDPA execution for the pinned Qwen attention projections.

Adapted from Qwen Transformers revision 4177486a9f199bd7be520eff14431071d5d41ec5.
The Q/K/V, normalization, rotary and output-gate operations retain their source
order. Only the dense attention kernel changes; QSA teacher probabilities and
cached inference continue through the unchanged eager implementation.
"""

import torch
from torch.nn import functional as F

from .upstream_decoder import Qwen4ExpTextAttention, apply_rotary_pos_emb, repeat_kv


class DenseSDPAQwenAttention(Qwen4ExpTextAttention):
    """An execution adapter with the same parameters and state-dict keys."""

    dense_sdpa_enabled = False

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        if (
            not self.dense_sdpa_enabled
            or past_key_values is not None
            or self.attention_dropout != 0
            or getattr(self.indexer, "sparse_enabled", True)
            or kwargs.get("output_attentions", False)
        ):
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        selected_token_mask = self.indexer(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask
        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            repeat_kv(key_states, self.num_key_value_groups),
            repeat_kv(value_states, self.num_key_value_groups),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        ).transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None


def configure_dense_attention(model, backend):
    if backend not in {"eager", "sdpa"}:
        raise ValueError("dense attention backend must be eager or sdpa")
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextAttention):
            module.__class__ = DenseSDPAQwenAttention
            module.dense_sdpa_enabled = backend == "sdpa"
