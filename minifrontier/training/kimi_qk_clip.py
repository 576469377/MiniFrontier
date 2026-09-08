"""K2-derived per-head QK-Clip adapted to K3's actual NoPE MLA projections.

Only nonshared head rows are scaled. Shared latent down-projections and shared
key rows are not changed. KDA state does not use this softmax-attention rule.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from minifrontier.models.minikimik3.upstream_layers import KimiMLAAttention


class KimiQKClip:
    def __init__(self, model, tau=100.0):
        if tau <= 0:
            raise ValueError("QK-Clip threshold must be positive")
        self.tau = tau
        self.layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, KimiMLAAttention)
        ]
        self.maxima = {
            name: torch.zeros(module.num_heads, device=module.o_proj.weight.device)
            for name, module in self.layers
        }
        self.updates = 0

    @contextmanager
    def capture(self, valid_mask):
        handles = []
        for name, module in self.layers:

            @torch.no_grad()
            def collect(module, args, call_kwargs, output, name=name):
                x = args[0] if args else call_kwargs["hidden_states"]
                b, length, _ = x.shape
                mask = getattr(module, "valid_mask", None)
                mask = valid_mask if mask is None else mask
                q = (
                    module.q_b_proj(module.q_a_layernorm(module.q_a_proj(x)))
                    if module.q_lora_rank is not None
                    else module.q_proj(x)
                )
                q = q.view(b, length, module.num_heads, module.q_head_dim).transpose(1, 2)
                latent, shared = module.kv_a_proj_with_mqa(x).split(
                    [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
                )
                kv = module.kv_b_proj(module.kv_a_layernorm(latent)).view(
                    b, length, module.num_heads, module.qk_nope_head_dim + module.v_head_dim
                )
                k = torch.cat(
                    (
                        kv[..., : module.qk_nope_head_dim],
                        shared[:, :, None].expand(-1, -1, module.num_heads, -1),
                    ),
                    dim=-1,
                ).transpose(1, 2)
                positions = torch.arange(length, device=x.device)
                for start in range(0, length, 64):
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        logits = q[:, :, start : start + 64].float() @ k.float().mT * module.scaling
                    allowed = positions[None, :] <= positions[start : start + 64, None]
                    allowed = (
                        allowed[None, None]
                        & mask[:, None, None, :].bool()
                        & mask[:, None, start : start + 64, None].bool()
                    )
                    observed = logits.masked_fill(~allowed, 0).amax((0, 2, 3))
                    self.maxima[name].copy_(torch.maximum(self.maxima[name], observed))

            handles.append(module.register_forward_hook(collect, with_kwargs=True))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def update(self):
        metrics = {}
        for name, module in self.layers:
            maxima = self.maxima[name]
            if dist.is_initialized():
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            gamma = (self.tau / maxima.clamp_min(self.tau)).clamp(max=1)
            scale = gamma.sqrt()
            q = module.q_b_proj if module.q_lora_rank is not None else module.q_proj
            q.weight.view(module.num_heads, module.q_head_dim, -1).mul_(scale[:, None, None])
            kv = module.kv_b_proj.weight.view(
                module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1
            )
            kv[:, : module.qk_nope_head_dim].mul_(scale[:, None, None])
            metrics[name] = dict(
                max_logit=float(maxima.max()),
                clipped_heads=int((gamma < 1).sum()),
                min_gamma=float(gamma.min()),
                threshold=self.tau,
            )
            maxima.zero_()
        self.updates += 1
        return metrics

    def state_dict(self):
        return dict(tau=self.tau, maxima=self.maxima, updates=self.updates)

    def load_state_dict(self, state):
        if state["tau"] != self.tau or set(state["maxima"]) != set(self.maxima):
            raise ValueError("QK-Clip contract differs")
        for key, value in state["maxima"].items():
            self.maxima[key].copy_(value)
        self.updates = state["updates"]
