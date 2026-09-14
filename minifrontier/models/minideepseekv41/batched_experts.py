# SPDX-License-Identifier: MIT
"""Model-local batched expert execution; preserves the upstream routing and sum order."""

from typing import Any, cast

import torch
import torch.nn.functional as F

from minifrontier.models.grouped_experts import configure, pack, projection, sum_routes

from .layers import MoE


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
        # DeepSeek applies routing weight before the down projection and BF16 boundary.
        activation = (F.silu(gate) * up * routed_weights[..., None]).to(x.dtype)
        values = projection(activation, [expert.w2 for expert in self.experts])
        result = sum_routes(values, routing, ids, torch.float32)
        return (result + self.shared_experts(x)).to(x.dtype).view(shape)


def configure_experts(model, mode):
    configure(model, mode, implementations=((MoE, BatchedDeepSeekMoE),))
