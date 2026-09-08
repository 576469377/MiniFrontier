"""Reproduce the reviewed, dependency-light Kimi/DeepSeek source extractions.

Original snapshots are immutable. Only inference restrictions and dependency
adapters listed below change; numerical training kernels live in separate files.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    return "\n\n".join(
        "".join(lines[n.lineno - 1 : n.end_lineno])
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    )


def main():
    base = ROOT / "third_party/upstream/kimi-k3-c5d1dd4"
    header = "\n".join("# " + line for line in (base / "LICENSE").read_text().splitlines())
    names = {
        "SituAndMul",
        "_get_situ_activation_params",
        "KimiRMSNorm",
        "KimiBlockSparseMLP",
        "KimiMLP",
        "repeat_kv",
        "eager_attention_forward",
        "KimiMLAAttention",
        "KimiDeltaAttention",
        "KimiMoEGate",
        "KimiSparseMoeBlock",
        "KimiDecoderLayer",
    }
    body = extract(base / "modeling_kimi_linear.py", names)
    body = body.replace("        assert not self.training\n", "")
    body = body.replace("    @torch.no_grad()\n    def moe_infer", "    def moe_infer")
    body = body.replace(
        """        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")""",
        "        y = self.moe_infer(hidden_states, topk_idx, topk_weight)",
    )
    # Never convert token counts through numpy or copy tensors to CPU during dispatch.
    start = body.index("        cnts = topk_ids.new_zeros")
    end = body.index("\n\nclass KimiDecoderLayer", start)
    body = (
        body[:start]
        + """        result = torch.zeros_like(x, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            token, slot = torch.where(topk_ids == i)
            value = expert(x[token]).float() * topk_weight[token, slot, None]
            result = result.index_add(0, token, value)
        return result.to(x.dtype)
"""
        + body[end:]
    )
    preamble = """
from __future__ import annotations
import math
from collections.abc import Callable
from typing import Any, TypedDict, Unpack
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from .attnres import _apply_attn_res
from .kernels import ShortConvolution, FusedRMSNormGated, chunk_kda, fused_recurrent_kda
KimiLinearConfig = Any
Cache = Any
KimiDynamicCache = Any
TransformersKwargs = TypedDict("TransformersKwargs", {})
FlashAttentionKwargs = TypedDict("FlashAttentionKwargs", {})
ACT2FN = {"silu": F.silu}
ALL_ATTENTION_FUNCTIONS: dict[str, Callable] = {}

def get_unpad_data(*args):
    raise ValueError("Use the model right-padding interface; raw variable-length KDA is not exposed")

def index_first_axis(x, indices):
    return x[indices]

def pad_input(*args):
    raise ValueError("Use the model right-padding interface")
"""
    (ROOT / "minifrontier/models/minikimik3/upstream_layers.py").write_text(
        "# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, HuggingFace Inc.\n"
        '# mypy: disable-error-code="assignment,arg-type,misc,list-item"\n'
        "# ruff: noqa: B904, SIM108, UP013\n"
        "# Derived from c5d1dd4. Adaptations: dependency adapters; differentiable MoE dispatch.\n"
        "# Regenerate with scripts/extract_training_sources.py, then ruff format.\n"
        + header
        + "\n"
        + preamble
        + "\n"
        + body
        + "\n"
    )
    base = ROOT / "third_party/upstream/deepseek-v4-60d8d70"
    header = "\n".join("# " + line for line in (base / "LICENSE").read_text().splitlines())
    names = {
        "set_dtype",
        "RMSNorm",
        "precompute_freqs_cis",
        "Gate",
        "Expert",
        "MoE",
        "Block",
        "ParallelHead",
        "MTPBlock",
    }
    body = extract(base / "model.py", names)
    # extract() excludes top-level decorators: restore these two only.
    body = body.replace("def set_dtype(", "@contextmanager\ndef set_dtype(")
    body = body.replace("def precompute_freqs_cis(", "@lru_cache(2)\ndef precompute_freqs_cis(")
    body = body.replace(
        "        self.attn = Attention(layer_id, args)",
        "        from .attention import Attention\n        self.attn = Attention(layer_id, args)",
    )
    body = body.replace("    @torch.inference_mode()\n", "")
    body = body.replace(
        "return F.linear(x[:, -1].float(), self.weight)", "return F.linear(x, self.weight)"
    )
    body = body.replace(
        "        weights /= weights.sum(dim=-1, keepdim=True)",
        "        weights = weights / weights.sum(dim=-1, keepdim=True)",
    )
    body = body.replace(
        "        weights *= self.route_scale", "        weights = weights * self.route_scale"
    )
    body = body.replace(
        "        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()\n",
        "",
    )
    body = body.replace("            if counts[i] == 0:\n                continue\n", "")
    body = body.replace(
        "            y[idx] += expert(x[idx], weights[idx, top, None])",
        "            y = y.index_add(0, idx, expert(x[idx], weights[idx, top, None]).float())",
    )
    body = body.replace(
        "        mixes = F.linear(x, hc_fn) * rsqrt",
        "        with torch.autocast(device_type=x.device.type, enabled=False):\n            mixes = F.linear(x, hc_fn) * rsqrt",
    )
    preamble = """
from __future__ import annotations
import math
from typing import Any, Optional, Tuple
from functools import lru_cache
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from .kernels import hc_split_sinkhorn
world_size = 1  # Replicated DDP; never infer tensor parallelism from the process group.
rank = 0
ModelArgs = Any
ParallelEmbedding = nn.Embedding

def linear(x, weight, bias=None):
    return F.linear(x, weight, bias)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
"""
    (ROOT / "minifrontier/models/minideepseekv4/upstream_layers.py").write_text(
        "# Derived from DeepSeek-V4-Flash 60d8d70. Unquantized differentiable DDP adaptation.\n"
        '# mypy: disable-error-code="assignment,misc,override"\n'
        "# ruff: noqa: UP035, UP045, UP006, SIM108\n"
        "# Changes: no inference_mode, full-sequence head, autograd-safe routing, local Attention.\n"
        + header
        + "\n"
        + preamble
        + "\n"
        + body
        + "\n"
    )


if __name__ == "__main__":
    main()
