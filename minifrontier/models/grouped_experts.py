"""Model-agnostic packed GEMM helpers with stable routing and sum order.

Each model owns its expert adapter and explicitly supplies its implementation
classes. This module never imports a model package.
"""

import torch

from minifrontier.training.mx_quant import MXLinear, fake_mx


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


def configure(model, mode, *, implementations):
    if mode not in {"loop", "batched"}:
        raise ValueError("expert execution must be loop or batched")
    for module in model.modules():
        for original, batched in implementations:
            if isinstance(module, original):
                module.__class__ = batched if mode == "batched" else original
