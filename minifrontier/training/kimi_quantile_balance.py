"""Kimi K3 Eq. 14: next-update Quantile Balancing, without an auxiliary loss.

Exact oracle uses linear interpolation at rank (m-1)*(1-k/n), including ties.
The additive histogram approximates each bracketing order statistic by its bin
midpoint. Its quantile error is bounded by one bin width. Bounded sigmoid scores
allow a guaranteed range derived from the OLD bias, shared by all ranks, rather
than silently clamping outliers. Histograms never retain token-sized tensors.
"""

from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
import torch.nn.functional as F


def margins(scores, old_bias, top_k):
    if scores.ndim != 2 or scores.shape[1] != old_bias.numel() or not 0 < top_k < scores.shape[1]:
        raise ValueError("QB needs [tokens, experts] and 0 < top-k < experts")
    cutoff = (scores + old_bias).topk(top_k + 1, dim=-1).values[:, -1]
    return scores - cutoff[:, None]


@torch.no_grad()
def exact_quantile_bias(scores, old_bias, top_k):
    if scores.shape[0] == 0:
        return old_bias.clone()
    proposed = -torch.quantile(
        margins(scores.float(), old_bias.float(), top_k),
        1 - top_k / scores.shape[1],
        dim=0,
        interpolation="linear",
    )
    return proposed - proposed.mean()


class QuantileHistogram:
    def __init__(self, bias, top_k, bins=2048):
        if bins < 16 or not 0 < top_k < bias.numel():
            raise ValueError("invalid QB histogram dimensions")
        self.bias, self.top_k, self.bins = bias, top_k, bins
        self.counts = torch.zeros((bias.numel(), bins), dtype=torch.int64, device=bias.device)
        self.load = torch.zeros(bias.numel(), dtype=torch.int64, device=bias.device)
        self.updates = 0
        self.overflows = 0
        self.previous_bias = bias.detach().clone()
        self._range()

    def _range(self):
        self.low = -1.0 - float(self.bias.max()) - 1e-5
        self.high = 1.0 - float(self.bias.min()) + 1e-5

    @torch.no_grad()
    def add(self, scores, selected):
        if scores.numel() == 0:
            return
        values = margins(scores.float(), self.bias.float(), self.top_k)
        bad = ~torch.isfinite(values) | (values < self.low) | (values > self.high)
        if bad.any():
            self.overflows += int(bad.sum())
            raise FloatingPointError("QB margin outside guaranteed sigmoid range; update aborted")
        indices = ((values - self.low) / (self.high - self.low) * self.bins).long()
        self.counts.scatter_add_(1, indices.T, torch.ones_like(indices.T))
        self.load.add_(torch.bincount(selected.flatten(), minlength=self.bias.numel()))

    @torch.no_grad()
    def update(self):
        if dist.is_initialized():
            dist.all_reduce(self.counts)
            dist.all_reduce(self.load)
        total = int(self.counts[0].sum())
        if total == 0:
            return None
        if not bool((self.counts.sum(-1) == total).all()):
            raise ValueError("QB expert histogram counts disagree")
        position = (total - 1) * (1 - self.top_k / self.bias.numel())
        lo, hi = int(position), min(int(position) + 1, total - 1)
        cumulative = self.counts.cumsum(-1)

        def order_stat(rank):
            index = (cumulative > rank).to(torch.int32).argmax(-1)
            return self.low + (index.float() + 0.5) * (self.high - self.low) / self.bins

        quantile = order_stat(lo).lerp(order_stat(hi), position - lo)
        self.previous_bias.copy_(self.bias)
        self.bias.copy_(-quantile + quantile.mean())
        self.updates += 1
        loads = self.load.float()
        metrics = dict(
            tokens=total,
            max_mean_load=float(loads.max() / loads.mean()),
            load_cv=float(loads.std(unbiased=False) / loads.mean()),
            idle_fraction=float((loads == 0).float().mean()),
            bias_min=float(self.bias.min()),
            bias_max=float(self.bias.max()),
            bin_width=(self.high - self.low) / self.bins,
            overflow=self.overflows,
            updates=self.updates,
        )
        self.counts.zero_()
        self.load.zero_()
        self._range()
        return metrics

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "counts",
                "load",
                "updates",
                "overflows",
                "low",
                "high",
                "previous_bias",
                "bins",
                "top_k",
            )
        }

    def load_state_dict(self, state):
        if state["bins"] != self.bins or state["top_k"] != self.top_k:
            raise ValueError("QB histogram configuration differs")
        for key, value in state.items():
            if isinstance(getattr(self, key), torch.Tensor):
                getattr(self, key).copy_(value)
            else:
                setattr(self, key, value)


class KimiQuantileBalance:
    def __init__(self, model, bins=2048):
        from minifrontier.models.minikimik3.upstream_layers import KimiMoEGate

        self.gates = [
            (name, module, QuantileHistogram(module.e_score_correction_bias, module.top_k, bins))
            for name, module in model.named_modules()
            if isinstance(module, KimiMoEGate)
        ]
        if any(
            g.moe_router_activation_func != "sigmoid" or g.num_expert_group != 1
            for _, g, _ in self.gates
        ):
            raise ValueError("QB production range requires ungrouped sigmoid Kimi routing")

    @contextlib.contextmanager
    def capture(self, valid_mask):
        handles = []
        for _, gate, histogram in self.gates:

            def collect(module, args, output, histogram=histogram):
                with torch.no_grad():
                    x = args[0].reshape(-1, module.gating_dim)
                    # Same autocast policy as the gate that just ran.
                    context = (
                        torch.autocast(device_type=x.device.type, enabled=False)
                        if getattr(module, "force_fp32", False)
                        else contextlib.nullcontext()
                    )
                    with context:
                        scores = F.linear(x.float(), module.weight.float()).sigmoid()
                    mask = getattr(module, "valid_mask", None)
                    valid = (valid_mask if mask is None else mask).flatten().bool()
                    if valid.numel() != x.shape[0]:
                        raise ValueError("QB valid-token mask does not match router inputs")
                    histogram.add(scores[valid], output[0][valid])

            handles.append(gate.register_forward_hook(collect))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def update(self, rate=None):
        return {name: histogram.update() for name, _, histogram in self.gates}

    def state_dict(self):
        return {name: histogram.state_dict() for name, _, histogram in self.gates}

    def load_state_dict(self, state):
        if set(state) != {name for name, _, _ in self.gates}:
            raise ValueError("QB checkpoint router topology differs")
        for name, _, histogram in self.gates:
            histogram.load_state_dict(state[name])
