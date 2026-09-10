# KDA projections/recurrence derive from Kimi c5d1dd4 under the Kimi K3 License.
"""Checked thin KDA adapter; reset both recurrence and convolution at packed boundaries."""

from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from minifrontier.models.minikimik3.kernels import chunk_kda, fused_recurrent_kda, reference_kda
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
        if state is None:
            return self.prefill(x, segments)
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
                out, conv, recurrent = self._run(z, carry)
                cache.conv_states[0], cache.recurrent_states[0] = conv, recurrent
                values.append(out)
                previous = dict(
                    segment=segment, conv=cache.conv_states[0], recurrent=cache.recurrent_states[0]
                )
            rows.append(torch.cat(values, 1))
            next_states.append(previous)
        return torch.cat(rows), next_states

    def _run(self, z, carry):
        core = self.core
        convs = carry.get("conv") or (None, None, None)
        projected, histories = [], []
        for proj, conv, history in zip(
            (core.q_proj, core.k_proj, core.v_proj),
            (core.q_conv1d, core.k_conv1d, core.v_conv1d),
            convs,
            strict=True,
        ):
            value, hist = conv(proj(z), cache=history, output_final_state=True)
            projected.append(value.unflatten(-1, (core.num_heads, core.head_dim)))
            histories.append(hist)
        # FLA's fused recurrent entry is forward-only. A one-token training
        # segment must still use its differentiable chunk kernel.
        kernel = (
            reference_kda
            if self.backend == "reference"
            else chunk_kda
            if torch.is_grad_enabled() or z.shape[1] > 1
            else fused_recurrent_kda
        )
        out, recurrent = kernel(
            q=projected[0],
            k=projected[1],
            v=projected[2],
            g=core.f_b_proj(core.f_a_proj(z)).unflatten(-1, (core.num_heads, core.head_dim)),
            beta=core.b_proj(z).float(),
            A_log=core.A_log,
            dt_bias=core.dt_bias,
            initial_state=carry.get("recurrent"),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            transpose_state_layout=True,
        )
        out = core.o_proj(
            core.o_norm(out, core.g_proj(z).unflatten(-1, (core.num_heads, core.head_dim))).flatten(
                -2
            )
        )
        return out, tuple(histories), recurrent

    def prefill(self, x, segments):
        """Batch independent runs of equal length; never carry across packed boundaries."""
        groups: dict[int, list[tuple[int, int, int, int]]] = {}
        final_segments = []
        has_padding = False
        for b, row in enumerate(segments.tolist()):
            has_padding |= any(segment < 0 for segment in row)
            starts = [0, *(i for i in range(1, len(row)) if row[i] != row[i - 1]), len(row)]
            for left, right in pairwise(starts):
                if row[left] >= 0:
                    groups.setdefault(right - left, []).append((b, left, right, row[left]))
            final_segments.append(row[-1])
        flat = x.flatten(0, 1)
        result = None
        states: list[dict[str, Any]] = [{} for _ in range(len(x))]
        for length, members in groups.items():
            indices = torch.tensor(
                [b * x.shape[1] + left for b, left, _, _ in members], device=x.device
            )[:, None] + torch.arange(length, device=x.device)
            output, conv, recurrent = self._run(flat[indices], {})
            if result is None:
                dtype = torch.promote_types(x.dtype, output.dtype) if has_padding else output.dtype
                result = flat.to(dtype) * 0
            result = result.index_copy(0, indices.flatten(), output.flatten(0, 1).to(result.dtype))
            for i, (b, _, right, segment) in enumerate(members):
                if right == x.shape[1] and final_segments[b] >= 0:
                    states[b] = dict(
                        segment=segment,
                        conv=tuple(v[i : i + 1] for v in conv),
                        recurrent=recurrent[i : i + 1],
                    )
        return (flat * 0 if result is None else result).reshape_as(x), states
