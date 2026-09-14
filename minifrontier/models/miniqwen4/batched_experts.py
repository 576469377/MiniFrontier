# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
"""Qwen 4177486 expert execution adapters; routing, parameters and sum order are unchanged."""

import torch
import torch.nn.functional as F

from minifrontier.models.grouped_experts import pack, sum_routes

from .upstream_decoder import Qwen4ExpTextExperts


class LoopQwenExperts(Qwen4ExpTextExperts):
    """Move the small expert ID list once instead of testing CUDA scalars per expert."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            # nonzero is ordered by expert ID and can never yield num_experts.
            expert_hit = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero().flatten().tolist()
        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


class BatchedQwenExperts(LoopQwenExperts):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        packed, routing = pack(hidden_states, top_k_index, self.num_experts)
        if packed is None:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        gate, up = torch.bmm(packed, self.gate_up_proj.transpose(1, 2)).chunk(2, -1)
        values = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
        return sum_routes(values, routing, top_k_index, hidden_states.dtype, top_k_weights)


def configure_experts(model, mode):
    if mode not in {"loop", "batched"}:
        raise ValueError("expert execution must be loop or batched")
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextExperts):
            module.__class__ = BatchedQwenExperts if mode == "batched" else LoopQwenExperts
