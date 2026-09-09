# LatentMoE/SiTU math adapted from Kimi c5d1dd4; Kimi K3 License applies.
"""FP32 no-drop latent expert aggregation and a single full-width shared expert."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def activation(gate, up, kind):
    g, u = gate.float(), up.float()
    return (
        (4 * (g / 4).tanh() * g.sigmoid()) * (25 * (u / 25).tanh())
        if kind == "situ"
        else F.silu(g) * u
    )


class Expert(nn.Module):
    def __init__(self, width, intermediate, kind):
        super().__init__()
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)
        self.kind = kind

    def forward(self, x):
        g, u = self.gate(x), self.up(x)
        return self.down(activation(g, u, self.kind).to(g.dtype))


class Router(nn.Module):
    correction_bias: torch.Tensor

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c.num_experts, c.hidden_size))
        self.register_buffer("correction_bias", torch.zeros(c.num_experts))
        self.top_k = c.num_experts_per_token

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(x.float(), self.weight.float()).sigmoid()
        selected = (scores + self.correction_bias).argsort(dim=-1, descending=True, stable=True)[
            ..., : self.top_k
        ]
        weights = scores.gather(-1, selected)
        return selected, weights / weights.sum(-1, keepdim=True).clamp_min(1e-12), scores


class LatentMoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, latent = c.hidden_size, c.routed_expert_hidden_size
        self.router = Router(c)
        self.down = nn.Linear(h, latent, bias=False)
        self.experts = nn.ModuleList(
            Expert(latent, c.moe_intermediate_size, c.activation) for _ in range(c.num_experts)
        )
        self.norm = RMSNorm(latent, c.rms_norm_eps)
        self.up = nn.Linear(latent, h, bias=False)
        self.shared = Expert(h, c.shared_intermediate_size, c.activation)
        self.scale = c.routed_scale

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        selected, weights, _ = self.router(flat)
        latent = self.down(flat)
        combined = torch.zeros_like(latent, dtype=torch.float32)
        for index, expert in enumerate(self.experts):
            token, slot = (selected == index).nonzero(as_tuple=True)
            if token.numel():
                update = expert(latent[token]).float() * weights[token, slot, None]
                combined = combined.index_add(0, token, update)
        routed = self.up(self.norm(combined).to(latent.dtype)) * self.scale
        return (routed + self.shared(flat)).reshape(shape)
