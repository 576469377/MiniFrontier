"""Padded batched GEMM with source-specific activations, weighting and sum order.

An explicit execution option preserves the original parameters and state keys.
It does not change routing, capacities, precision policy or the optimizer. The
loop implementation remains the default until numerical/performance comparison.
Source-derived Kimi computation retains the Kimi K3 license in its upstream_layers;
DeepSeek retains MIT and Qwen retains Apache-2.0, as listed in THIRD_PARTY_NOTICES.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F

from minifrontier.training.mx_quant import MXLinear, fake_mx

from .minideepseekv4.upstream_layers import MoE
from .minikimik3.upstream_layers import KimiSparseMoeBlock
from .miniqwen4.upstream_decoder import Qwen4ExpTextExperts


def pack(x, ids, experts):
    order = ids.flatten().argsort(stable=True)
    expert_ids = ids.flatten()[order]
    counts = torch.bincount(expert_ids, minlength=experts)
    capacity = int(counts.max())
    # Never allocate E times the full token count for a collapsed router.
    # Fallback keeps exact source routing; no token is dropped or capacity-clipped.
    if experts * capacity > 4 * order.numel():
        return None, None
    offsets = counts.cumsum(0) - counts
    slots = torch.arange(order.numel(), device=x.device) - offsets[expert_ids]
    tokens = order // ids.shape[1]
    values = x.new_zeros(experts, capacity, x.shape[-1])
    values[expert_ids, slots] = x[tokens]
    return values, (order, expert_ids, slots)


def sum_routes(packed, routing, ids, dtype, weights=None):
    order, expert_ids, slots = routing
    values = packed[expert_ids, slots]
    if weights is not None:
        values = values.float() * weights.flatten()[order, None]
    unsorted = values.new_zeros(order.numel(), values.shape[-1]).index_copy(0, order, values)
    by_token = unsorted.view(*ids.shape, -1)
    # Match the original increasing-expert traversal, including BF16 rounding.
    sorted_values = by_token.gather(1, ids.argsort(-1)[..., None].expand_as(by_token)).to(dtype)
    result = torch.zeros_like(sorted_values[:, 0])
    for slot in range(ids.shape[1]):
        result = result + sorted_values[:, slot]
    return result


def projection(x, linears):
    weight = torch.stack([module.weight for module in linears])
    if any(module.bias is not None for module in linears):
        raise ValueError("native routed projections must be bias-free")
    if isinstance(linears[0], MXLinear):
        if linears[0].activation_bits is not None:
            x = fake_mx(x, bits=linears[0].activation_bits)
        weight = fake_mx(weight)
    return torch.bmm(x, weight.transpose(1, 2))


class BatchedKimiMoE(KimiSparseMoeBlock):
    def moe_infer(self, x, topk_ids, topk_weight):
        packed, routing = pack(x, topk_ids, len(self.experts))
        if packed is None:
            return super().moe_infer(x, topk_ids, topk_weight)
        gate = projection(packed, [expert.w1 for expert in self.experts])
        up = projection(packed, [expert.w3 for expert in self.experts])
        activation = (
            cast(Any, self.experts[0]).act_fn(torch.cat((gate, up), -1))
            if self.config.hidden_act == "situ"
            else cast(Any, self.experts[0]).act_fn(gate) * up
        )
        values = projection(activation, [expert.w2 for expert in self.experts])
        return sum_routes(values, routing, topk_ids, torch.float32, topk_weight).to(x.dtype)


class BatchedDeepSeekMoE(MoE):
    def forward(self, x, input_ids):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, ids = self.gate(x, input_ids.flatten())
        packed, routing = pack(x, ids, len(self.experts))
        if packed is None:
            # Reuse the already-computed route to avoid collecting router stats twice.
            result = torch.zeros_like(x, dtype=torch.float32)
            for index, expert in enumerate(self.experts):
                token, slot = torch.where(ids == index)
                value = expert(x[token], weights[token, slot, None])
                result = result.index_add(0, token, value.float())
            return (result + self.shared_experts(x)).to(x.dtype).view(shape)
        gate = projection(packed, [expert.w1 for expert in self.experts]).float()
        up = projection(packed, [expert.w3 for expert in self.experts]).float()
        limit = cast(Any, self.experts[0]).swiglu_limit
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        order, expert_ids, slots = routing
        routed_weights = weights.new_zeros(packed.shape[:2])
        routed_weights[expert_ids, slots] = weights.flatten()[order]
        # V4 applies routing weight before the down projection and BF16 boundary.
        activation = (F.silu(gate) * up * routed_weights[..., None]).to(x.dtype)
        values = projection(activation, [expert.w2 for expert in self.experts])
        result = sum_routes(values, routing, ids, torch.float32)
        return (result + self.shared_experts(x)).to(x.dtype).view(shape)


class BatchedQwenExperts(Qwen4ExpTextExperts):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        packed, routing = pack(hidden_states, top_k_index, self.num_experts)
        if packed is None:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        gate, up = torch.bmm(packed, self.gate_up_proj.transpose(1, 2)).chunk(2, -1)
        values = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
        return sum_routes(values, routing, top_k_index, hidden_states.dtype, top_k_weights)


def configure(model, mode):
    if mode not in {"loop", "batched"}:
        raise ValueError("expert execution must be loop or batched")
    kinds = [
        (KimiSparseMoeBlock, BatchedKimiMoE),
        (MoE, BatchedDeepSeekMoE),
        (Qwen4ExpTextExperts, BatchedQwenExperts),
    ]
    for module in model.modules():
        for original, batched in kinds:
            if isinstance(module, original):
                module.__class__ = batched if mode == "batched" else original
