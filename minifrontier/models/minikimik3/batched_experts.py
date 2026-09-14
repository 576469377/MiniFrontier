# SPDX-License-Identifier: LicenseRef-Kimi-K3
"""Model-local batched expert execution; preserves the upstream routing and sum order."""

from typing import Any, cast

import torch

from minifrontier.models.grouped_experts import configure, pack, projection, sum_routes

from .upstream_layers import KimiSparseMoeBlock


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


def configure_experts(model, mode):
    configure(model, mode, implementations=((KimiSparseMoeBlock, BatchedKimiMoE),))
