"""Exact source router loss over an accumulation/DDP window.

Assignments are nondifferentiable. A no-grad replay first fixes their global
frequency; the gradient pass sums router probabilities using that frequency.
This trades an extra forward for bounded activation memory and exact gradients.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist


def normalized_router_loss(routers, experts, top_k, valid_mask=None, frequency=None):
    if frequency is None and (valid_mask is None or valid_mask.any()):
        from minifrontier.models.miniqwen4.upstream_loss import load_balancing_loss_func

        return load_balancing_loss_func(routers, experts, top_k, valid_mask)
    counts = torch.zeros(experts, device=routers[0].device, dtype=torch.float32)
    probabilities = torch.zeros_like(counts)
    denominator = counts.new_zeros(())
    for logits in routers:
        p = logits.softmax(-1)
        valid = (
            torch.ones(p.shape[0], device=p.device, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.flatten().bool()
        )
        selected = p.detach().topk(top_k, dim=-1).indices[valid]
        counts += torch.bincount(selected.flatten(), minlength=experts)
        probabilities = probabilities + (p.float() * valid[:, None]).sum(0)
        denominator += valid.sum()
    frequency = counts / denominator.clamp_min(1) if frequency is None else frequency
    mean_probability = probabilities / denominator.clamp_min(1)
    return (frequency * mean_probability.unsqueeze(0)).sum() * experts


class QwenWindowBalance:
    def __init__(self, model):
        self.model = model
        self.gates = [("main", layer.mlp.gate) for layer in model.model.layers]
        if model.mtp is not None:
            self.gates.append(("mtp", model.mtp.block.mlp.gate))
        self.totals = {
            key: torch.zeros(model.config.num_experts + 1, device=model.lm_head.weight.device)
            for key, _ in self.gates
        }

    @contextmanager
    def capture(self, valid_mask):
        handles = []
        for key, gate in self.gates:

            @torch.no_grad()
            def collect(module, args, output, key=key):
                valid = module.valid_mask if key == "mtp" else valid_mask
                probabilities = output[0].softmax(-1)
                selected = probabilities.topk(
                    self.model.config.num_experts_per_tok, dim=-1
                ).indices[valid.flatten().bool()]
                self.totals[key][:-1] += torch.bincount(
                    selected.flatten(), minlength=self.model.config.num_experts
                )
                self.totals[key][-1] += valid.sum()

            handles.append(gate.register_forward_hook(collect))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def finalize(self):
        for key, total in self.totals.items():
            if dist.is_initialized():
                dist.all_reduce(total)
            frequency = total[:-1] / total[-1].clamp_min(1)
            target = self.model if key == "main" else self.model.mtp
            target.router_window_frequency = frequency

    def clear(self):
        self.model.router_window_frequency = None
        if self.model.mtp is not None:
            self.model.mtp.router_window_frequency = None
