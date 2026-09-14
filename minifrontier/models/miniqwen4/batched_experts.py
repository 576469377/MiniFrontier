# SPDX-License-Identifier: Apache-2.0
"""Model-local batched expert execution; preserves the upstream routing and sum order."""

import torch

from minifrontier.models.grouped_experts import configure, pack, sum_routes

from .upstream_decoder import Qwen4ExpTextExperts


class BatchedQwenExperts(Qwen4ExpTextExperts):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        packed, routing = pack(hidden_states, top_k_index, self.num_experts)
        if packed is None:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        gate, up = torch.bmm(packed, self.gate_up_proj.transpose(1, 2)).chunk(2, -1)
        values = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
        return sum_routes(values, routing, top_k_index, hidden_states.dtype, top_k_weights)


def configure_experts(model, mode):
    configure(model, mode, implementations=((Qwen4ExpTextExperts, BatchedQwenExperts),))
