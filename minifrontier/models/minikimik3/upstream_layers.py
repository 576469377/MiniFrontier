# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, HuggingFace Inc.
# mypy: disable-error-code="assignment,arg-type,misc,list-item"
# ruff: noqa: B904, SIM108, UP013
# Derived from c5d1dd4. Adaptations: dependency adapters; differentiable MoE dispatch.
# Regenerate with scripts/extract_training_sources.py, then ruff format.
# Kimi K3 License
#
# Copyright (c) 2026 Moonshot AI
#
# Permission is hereby granted, free of charge, to any person (the "Licensee")
# obtaining a copy of this software — including the model weights, parameters,
# configuration files, inference and training code, and associated documentation
# (collectively, the "Software") — to deal in the Software without restriction.
# This includes, without limitation, the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software; to run,
# deploy, fine-tune, or otherwise modify the Software and create derivative works
# from it; and to permit persons to whom the Software is furnished to do so, in
# each case subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. Licensee's use of the
# Software must comply with applicable laws and regulations.
#
# 2. "Model as a Service" means giving a third party access to language model
# inference or fine-tuning (e.g., via API) in a manner that allows such third
# party to exercise meaningful control over the inputs, parameters, or training
# data. This does not include (a) end-user products with model capabilities solely
# embedded within specific features or harnesses, or (b) mere relaying of requests
# to models hosted by others.
#
# If the Licensee or any of its affiliates operates a Model as a Service business,
# and the aggregate revenue of the Licensee and its affiliates exceeds 20 million
# US dollars (or the equivalent in other currencies) in total over any consecutive
# 12 months, the Licensee must enter into a separate agreement with Moonshot AI
# before using the Software or its derivative works for any commercial purpose.
#
# 3. If the Software (or any derivative works thereof) is used for any of the
# Licensee's commercial products or services that have more than 100 million
# monthly active users, or more than 20 million US dollars (or equivalent in other
# currencies) in monthly revenue, "Kimi K3" must be prominently displayed on the
# user interface of such product or service.
#
# 4. The requirements set forth in Sections 2 and 3 do not apply to: (a) internal
# use of the Software, defined as any use that does not make the Software, its
# outputs, or its underlying capabilities available to third parties; or (b) any
# use of the Software accessed through Moonshot AI's official products or
# certified inference partners.
#
# 5. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS”
# BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
# AND NONINFRINGEMENT. IN NO EVENT SHALL MOONSHOT AI OR ITS AFFILIATES OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# For any questions regarding this license, please contact <license@moonshot.ai>.

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypedDict, Unpack

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .attnres import _apply_attn_res
from .kernels import FusedRMSNormGated, ShortConvolution, chunk_kda, fused_recurrent_kda

KimiLinearConfig = Any
Cache = Any
KimiDynamicCache = Any
TransformersKwargs = TypedDict("TransformersKwargs", {})
FlashAttentionKwargs = TypedDict("FlashAttentionKwargs", {})
ACT2FN = {"silu": F.silu}
ALL_ATTENTION_FUNCTIONS: dict[str, Callable] = {}


def get_unpad_data(*args):
    raise ValueError(
        "Use the model right-padding interface; raw variable-length KDA is not exposed"
    )


def index_first_axis(x, indices):
    return x[indices]


def pad_input(*args):
    raise ValueError("Use the model right-padding interface")


class SituAndMul(nn.Module):
    """
    SituAndMul activation: beta * tanh(gate / beta) * sigmoid(gate) * up
    When linear_beta is set, up is also transformed by linear_beta * tanh(up / linear_beta).
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


def _get_situ_activation_params(config: KimiLinearConfig):
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class KimiBlockSparseMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up

        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
            current_hidden_states = self.act_fn(gate_up)
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            down_proj = self.down_proj(self.act_fn(gate_up))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand the key/value heads from `num_key_value_heads` to `num_attention_heads`."""
    if n_rep == 1:
        return hidden_states
    return torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = F.dropout(probs, p=dropout, training=module.training)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, value).transpose(1, 2).contiguous()

    return out, probs


class KimiMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(f"Kimi MLA config is not found or not properly formatted: {e}")

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
            )
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        if self.use_output_gate:
            projection_size = self.num_heads * self.v_head_dim
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and self.q_head_dim != self.v_head_dim
        ):
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if (
            self.config._attn_implementation == "flash_attention_2"
            and self.q_head_dim != self.v_head_dim
        ):
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)
        return attn_output


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in ["chunk", "fused_recurrent"], f"Not supported mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        self.A_log = torch.nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: KimiDynamicCache | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if use_cache and q_len == 1 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)
        q, conv_state_q = self.q_conv1d(
            x=q_proj_states,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=k_proj_states,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=v_proj_states,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (conv_state_q, conv_state_k, conv_state_v)

        if self.use_full_rank_gate:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim)),
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}",
            )

        # select top-k experts
        scores = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [n, num_expert_group]
            group_idx = torch.topk(
                group_scores,
                k=self.topk_group,
                dim=-1,
                sorted=False,
            )[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
        else:
            tmp_scores = scores_for_choice
        _, topk_idx = torch.topk(
            tmp_scores,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation
    The namings are consistent with Kimi's version.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ],
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=intermediate_size,
            )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size,
                    eps=config.rms_norm_eps,
                )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        y = self.moe_infer(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, topk_ids, topk_weight):
        result = torch.zeros_like(x, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            token, slot = torch.where(topk_ids == i)
            value = expert(x[token]).float() * topk_weight[token, slot, None]
            result = result.index_add(0, token, value)
        return result.to(x.dtype)


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention residual
        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                block_residual,
                **kwargs,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        return prefix_sum, block_residual
