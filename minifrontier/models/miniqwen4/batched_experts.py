# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
"""Qwen 4177486 expert execution adapters; routing, parameters and sum order are unchanged."""

import torch
import torch.nn.functional as F

from .upstream_decoder import Qwen4ExpTextExperts


class LoopQwenExperts(Qwen4ExpTextExperts):
    """Group routes once, retaining upstream slot-major rows and expert sum order."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            # Upstream where([slot, token]) visits all tokens in slot 0 first.
            # A stable sort of token-major IDs would change each expert's row order.
            ids = top_k_index.transpose(0, 1).reshape(-1)
            order = ids.argsort(stable=True)
            counts = ids.new_zeros(self.num_experts)
            counts.scatter_add_(0, ids, torch.ones_like(ids))
            sizes = counts.tolist()  # One fixed-size host transfer, no per-expert nonzero.
        # One multi-output view joins the expert gradients once. Selecting each
        # weight separately creates a full [experts, ...] zero gradient per slice.
        # Keep these views outside no_grad so all active expert gradients survive.
        gate_up_weights = self.gate_up_proj.unbind(0)
        down_weights = self.down_proj.unbind(0)
        start = 0
        for expert_idx, size in enumerate(sizes):
            if not size:
                continue
            positions = order[start : start + size]
            start += size
            top_k_pos = positions // hidden_states.shape[0]
            token_idx = positions % hidden_states.shape[0]
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, gate_up_weights[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_weights[expert_idx])
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    def reference(self, hidden_states, top_k_index, top_k_weights):
        """Previous loop for same-weight numerical and execution comparisons."""
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
    """Count-bucketed GEMMs with Qwen's route order, dtype and reductions.

    Each nonempty expert gets a power-of-two capacity, with less than twice the
    routed rows allocated overall. Duplicate routes retain the exact loop.
    GEMM batching can change floating-point rounding; no token is discarded.
    """

    execution_stats_enabled = False

    def _record_execution(self, backend, routes, capacities=(), members=()):
        if self.execution_stats_enabled:
            self.last_execution = {
                "backend": backend,
                "routed_assignments": routes,
                "packed_assignments": sum(c * n for c, n in zip(capacities, members, strict=True)),
                "bucket_capacities": tuple(capacities),
                "bucket_expert_counts": tuple(members),
            }

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if top_k_index.numel() == 0:
            self._record_execution("empty_loop", 0)
            return super().forward(hidden_states, top_k_index, top_k_weights)
        tokens, top_k = top_k_index.shape
        with torch.no_grad():
            ids = top_k_index.transpose(0, 1).reshape(-1)
            order = ids.argsort(stable=True)
            counts = ids.new_zeros(self.num_experts)
            counts.scatter_add_(0, ids, torch.ones_like(ids))
            ascending = top_k_index.argsort(dim=-1, stable=True)
            sorted_ids = top_k_index.gather(1, ascending)
            duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()
            # One bounded host transfer determines buckets and duplicate fallback.
            totals = torch.cat((counts, duplicate.to(counts.dtype).reshape(1))).tolist()
        if totals[-1]:
            self._record_execution("duplicate_loop", ids.numel())
            return super().forward(hidden_states, top_k_index, top_k_weights)
        buckets: dict[int, list[tuple[int, int, int]]] = {}
        offset = 0
        for expert, count in enumerate(totals[:-1]):
            if count:
                capacity = 1 << (count - 1).bit_length()
                buckets.setdefault(capacity, []).append((expert, offset, count))
            offset += count
        route_values = None
        flat_weights = top_k_weights.transpose(0, 1).reshape(-1)
        for capacity, members in sorted(buckets.items()):
            expert_ids = ids.new_tensor([expert for expert, _, _ in members])
            sizes = ids.new_tensor([count for _, _, count in members])
            route_count = sum(count for _, _, count in members)
            positions = torch.cat([order[start : start + count] for _, start, count in members])
            local_experts = torch.repeat_interleave(
                torch.arange(len(members), device=ids.device), sizes, output_size=route_count
            )
            starts = torch.repeat_interleave(
                sizes.cumsum(0) - sizes, sizes, output_size=route_count
            )
            slots = torch.arange(route_count, device=ids.device) - starts
            packed = hidden_states.new_zeros(len(members), capacity, self.hidden_dim)
            packed[local_experts, slots] = hidden_states[positions % tokens]
            gate_up = self.gate_up_proj.index_select(0, expert_ids)
            down = self.down_proj.index_select(0, expert_ids)
            gate, up = torch.bmm(packed, gate_up.transpose(1, 2)).chunk(2, -1)
            values = torch.bmm(self.act_fn(gate) * up, down.transpose(1, 2))
            # Preserve BF16 * BF16 rounding before accumulator conversion.
            values = values[local_experts, slots] * flat_weights[positions, None]
            if route_values is None:
                route_values = values.new_zeros(ids.numel(), self.hidden_dim)
            # Buckets own disjoint route rows; no earlier value is read here.
            route_values.index_copy_(0, positions, values)
        assert route_values is not None
        self._record_execution(
            "bucketed", ids.numel(), sorted(buckets), [len(buckets[c]) for c in sorted(buckets)]
        )
        route_values = route_values.view(top_k, tokens, self.hidden_dim).transpose(0, 1)
        ordered = route_values.gather(1, ascending[..., None].expand(-1, -1, self.hidden_dim)).to(
            hidden_states.dtype
        )
        result = torch.zeros_like(hidden_states)
        for slot in range(top_k):
            result = result + ordered[:, slot]
        return result


def configure_experts(model, mode):
    if mode not in {"loop", "batched"}:
        raise ValueError("expert execution must be loop or batched")
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextExperts):
            module.__class__ = BatchedQwenExperts if mode == "batched" else LoopQwenExperts
