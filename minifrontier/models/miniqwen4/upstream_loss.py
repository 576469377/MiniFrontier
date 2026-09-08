# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md.
# Unchanged function from transformers 4177486a9f199bd7be520eff14431071d5d41ec5.
# Upstream optional-num_experts and float-to-Tensor annotations are retained;
# the source LM adapter always supplies a validated positive expert count.
# mypy: disable-error-code="arg-type,assignment,operator"
from __future__ import annotations

import torch


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Accumulate assignment counts and probability sums layer by layer, normalizing at the end,
    # so peak memory stays O(seq_len * num_experts) regardless of the number of layers.
    compute_device = gate_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    router_prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    total_rows = 0.0

    if attention_mask is not None:
        # The same flat mask applies to every layer's [batch_size * sequence_length] rows.
        flat_mask = attention_mask.reshape(-1).to(device=compute_device, dtype=torch.float32)

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate.to(compute_device), dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if attention_mask is None:
            # Count of top-k assignments per expert
            tokens_per_expert_sum = (
                tokens_per_expert_sum
                + torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
            )
            # Sum of routing probabilities per expert
            router_prob_sum = router_prob_sum + routing_weights.float().sum(dim=0)
            total_rows = total_rows + routing_weights.shape[0]
        else:
            # Same reductions, weighted by the attention mask to exclude padding tokens
            tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
                num_experts, dtype=torch.float32, device=compute_device
            ).scatter_add_(0, selected_experts.reshape(-1), flat_mask.repeat_interleave(top_k))
            router_prob_sum = router_prob_sum + (
                routing_weights.float() * flat_mask.unsqueeze(-1)
            ).sum(dim=0)
            total_rows = total_rows + flat_mask.sum()

    tokens_per_expert = tokens_per_expert_sum / total_rows
    router_prob_per_expert = router_prob_sum / total_rows

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
