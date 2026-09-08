"""Independent per-matrix optimizer used to check fused semantic blocks."""

import math

import torch

from minifrontier.training.polar_express import polar_express_orthogonalize


class ReferenceMuon(torch.optim.Optimizer):
    def __init__(self, params, *, lr, weight_decay):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                state["momentum"].lerp_(parameter.grad, 0.05)
                direction = parameter.grad.lerp(state["momentum"], 0.95)
                update = polar_express_orthogonalize(direction)
                update = update * (0.2 * math.sqrt(max(update.shape)))
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
