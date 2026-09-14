# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional FLA chunk execution for Qwen Transformers 4177486 Gated DeltaNet.

The uncached path preserves the source projections, convolution, padding mask,
gate activations, head repetition and output normalization. FLA is an optional
runtime dependency; CPU, cached inference and unsupported inputs retain the
unchanged torch reference. Its fused BF16 kernel has different rounding order.
"""

from functools import lru_cache

import torch
from torch.nn import functional as F

from .upstream_core import Qwen4ExpTextGatedDeltaNet, causal_conv1d_fn
from .upstream_ple import apply_mask_to_padding_states


@lru_cache(maxsize=1)
def _load_fla_chunk():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("the fla GDN backend requires the optional fla-core package") from exc
    return chunk_gated_delta_rule


def fla_chunk(query, key, value, *, g, beta, initial_state=None, output_final_state=False):
    """Use the source's log-space decay, Q/K norm and [B, HV, K, V] FP32 state."""
    return _load_fla_chunk()(
        q=query,
        k=key,
        v=value,
        g=g.float(),
        beta=beta,
        scale=key.shape[-1] ** -0.5,
        initial_state=None if initial_state is None else initial_state.float(),
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        cu_seqlens=None,
        chunk_size=64,
    )


class FLAQwenGatedDeltaNet(Qwen4ExpTextGatedDeltaNet):
    """Same parameters and state keys; only uncached CUDA BF16 may use FLA."""

    fla_enabled = False

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if (
            not self.fla_enabled
            or hidden_states.device.type != "cuda"
            or cache_params is not None
            or kwargs
            or self.head_k_dim > 256
            or not (
                hidden_states.dtype == torch.bfloat16
                or (
                    torch.is_autocast_enabled("cuda")
                    and torch.get_autocast_dtype("cuda") == torch.bfloat16
                )
            )
        ):
            return super().forward(
                hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
            )
        return self._forward_chunk(hidden_states, attention_mask, fla_chunk)

    def _forward_chunk(self, hidden_states, attention_mask, chunk_kernel):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core_attn_out, _ = chunk_kernel(
            query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False
        )
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def configure_gdn(model, backend):
    if backend not in {"torch", "fla"}:
        raise ValueError("GDN backend must be torch or fla")
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextGatedDeltaNet):
            module.__class__ = FLAQwenGatedDeltaNet
            module.fla_enabled = backend == "fla"
