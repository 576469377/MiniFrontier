# KDA projections/recurrence derive from Kimi c5d1dd4 under the Kimi K3 License.
"""Checked thin KDA adapter; reset both recurrence and convolution at packed boundaries."""

from itertools import pairwise
from types import SimpleNamespace

import torch
from torch import nn

from minifrontier.models.minikimik3.kernels import reference_kda
from minifrontier.models.minikimik3.upstream_layers import KimiDeltaAttention


class KDA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.core = KimiDeltaAttention(
            SimpleNamespace(
                hidden_size=c.hidden_size,
                rms_norm_eps=c.rms_norm_eps,
                linear_attn_config=dict(
                    short_conv_kernel_size=c.conv_size,
                    head_dim=c.kda_head_dim,
                    num_heads=c.num_attention_heads,
                    use_full_rank_gate=True,
                    gate_lower_bound=-5.0,
                ),
            ),
            0,
        )
        self.backend = c.kda_backend

    def forward(self, x, segments, state=None):
        # Each contiguous independent sample is passed separately to the source primitive.
        rows = []
        next_states = []
        for b in range(x.shape[0]):
            values = []
            previous = {} if state is None else state[b]
            starts = [
                0,
                *(segments[b, 1:] != segments[b, :-1]).nonzero().flatten().add(1).tolist(),
                x.shape[1],
            ]
            for left, right in pairwise(starts):
                segment = int(segments[b, left])
                if segment < 0:
                    values.append(x[b : b + 1, left:right] * 0)
                    previous = {}
                    continue
                carry = previous if previous.get("segment") == segment else {}
                cache = SimpleNamespace(
                    conv_states=[carry.get("conv")], recurrent_states=[carry.get("recurrent")]
                )
                z = x[b : b + 1, left:right]
                if self.backend == "reference":
                    core = self.core
                    convs = carry.get("conv") or (None, None, None)
                    projected, histories = [], []
                    for proj, conv, history in zip(
                        (core.q_proj, core.k_proj, core.v_proj),
                        (core.q_conv1d, core.k_conv1d, core.v_conv1d),
                        convs,
                        strict=True,
                    ):
                        v, hist = conv(proj(z), cache=history, output_final_state=True)
                        projected.append(v.unflatten(-1, (core.num_heads, core.head_dim)))
                        histories.append(hist)
                    q, k, v = projected
                    out, recurrent = reference_kda(
                        q,
                        k,
                        v,
                        core.f_b_proj(core.f_a_proj(z)).unflatten(
                            -1, (core.num_heads, core.head_dim)
                        ),
                        core.b_proj(z),
                        core.A_log,
                        core.dt_bias,
                        initial_state=carry.get("recurrent"),
                    )
                    out = core.o_proj(
                        core.o_norm(
                            out, core.g_proj(z).unflatten(-1, (core.num_heads, core.head_dim))
                        ).flatten(-2)
                    )
                    cache.conv_states[0], cache.recurrent_states[0] = tuple(histories), recurrent
                else:
                    # Source chooses a recurrent kernel for single-token *inference*.
                    training = self.core.training
                    if z.shape[1] == 1:
                        self.core.training = False
                    try:
                        out = self.core(z, cache_params=cache)
                    finally:
                        self.core.training = training
                values.append(out)
                previous = dict(
                    segment=segment, conv=cache.conv_states[0], recurrent=cache.recurrent_states[0]
                )
            rows.append(torch.cat(values, 1))
            next_states.append(previous)
        return torch.cat(rows), next_states
