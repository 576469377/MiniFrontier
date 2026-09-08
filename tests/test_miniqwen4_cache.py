"""Cache integration against pinned HF state updates and full-sequence output."""

import ast
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import pytest
import torch
from test_miniqwen4 import tiny_config

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM, MiniQwen4TextModel
from minifrontier.models.miniqwen4.cache import MiniQwen4Cache


def original_cache_types():
    path = (
        Path(__file__).resolve().parents[1]
        / "third_party/upstream/qwen4_exp-4177486/cache_utils.py"
    )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "4b284431cb3a881b6e6f8b8c6430df6f2efdcb3366a2484c7984ae88c612c61a"
    )
    names = {
        "CacheLayerMixin",
        "DynamicLayer",
        "DynamicIndexedLayer",
        "LinearAttentionCacheLayerMixin",
        "LinearAttentionLayer",
    }
    nodes = ast.parse("from __future__ import annotations").body
    nodes.extend(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names)
    namespace = dict(
        torch=torch,
        ABC=ABC,
        abstractmethod=abstractmethod,
        deprecate_kwarg=lambda *a, **k: lambda target: target,
        is_torchdynamo_compiling=lambda: False,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class OriginalStateCache(MiniQwen4Cache):
    """Only model-level lifecycle is local; state math calls original HF classes."""

    def prepare(self, owner, layers, batch, device, dtype):
        fresh = not self.layers
        super().prepare(owner, layers, batch, device, dtype)
        if fresh:
            types = original_cache_types()
            self.layers = [
                types["DynamicIndexedLayer"]()
                if hasattr(layer, "self_attn")
                else types["LinearAttentionLayer"](number_of_states=3)
                for layer in owner.layers
            ]

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.layers[layer_idx].has_previous_state[state_idx]

    def update_conv_state(self, states, layer_idx, state_idx=0, *, conv_kernel_size):
        return self.layers[layer_idx].update_conv_state(states, state_idx, conv_kernel_size)

    def update_recurrent_state(self, states, layer_idx):
        return self.layers[layer_idx].update_recurrent_state(states)

    def update_indexer(self, states, layer_idx):
        return self.layers[layer_idx].update_indexer(states)

    def update(self, keys, values, layer_idx):
        return self.layers[layer_idx].update(keys, values)


@pytest.mark.parametrize("mode", ["full", "sparse"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pieces", [(1, 1, 1, 1, 1, 1, 1, 1, 1), (3, 1, 2, 3)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@torch.no_grad()
def test_model_cache_matches_upstream_state_updates_and_full_forward(mode, pieces, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(603)
    model = (
        MiniQwen4TextModel(tiny_config(), attention_mode=mode).to(device=device, dtype=dtype).eval()
    )
    # Initialization zeros this convolution; activate it to test real PLE history.
    activated = 0
    for name, parameter in model.named_parameters():
        if ".ple.conv1d.weight" in name:
            parameter.normal_(std=0.03)
            activated += 1
    assert activated == 1
    ids = torch.tensor([[5, 6, 2, 2, 7, 8, 9, 2, 10], [2, 8, 7, 9, 2, 6, 5, 4, 3]], device=device)
    full = model(ids)
    actual_cache, oracle_cache = MiniQwen4Cache(), OriginalStateCache()
    actual_parts = []
    start = 0
    for size in pieces:
        part = ids[:, start : start + size]
        actual = model(part, cache=actual_cache)
        oracle = model(part, cache=oracle_cache)
        torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
        actual_parts.append(actual)
        start += size
        assert actual_cache.length == start
        for left, right in zip(actual_cache.layers, oracle_cache.layers, strict=True):
            if hasattr(right, "conv_states"):
                for index, state in left.conv_states.items():
                    torch.testing.assert_close(state, right.conv_states[index], atol=0, rtol=0)
                for index, state in left.recurrent_states.items():
                    torch.testing.assert_close(state, right.recurrent_states[index], atol=0, rtol=0)
            else:
                torch.testing.assert_close(left.keys, right.keys, atol=0, rtol=0)
                torch.testing.assert_close(left.values, right.values, atol=0, rtol=0)
                if mode == "sparse":
                    torch.testing.assert_close(
                        left.indexer_keys, right.indexer_keys, atol=0, rtol=0
                    )
    # Recurrent-vs-chunk kernels have different accumulation orders; exact
    # equality is required above for original-vs-adapter, not across kernels.
    tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(torch.cat(actual_parts, 1), full, **tolerance)


def test_cache_rejects_unsafe_reuse_and_training():
    model = MiniQwen4ForCausalLM(tiny_config())
    ids = torch.tensor([[3, 4, 5]])
    cache = MiniQwen4Cache()
    with pytest.raises(ValueError, match="eval mode"):
        model(ids, cache=cache)
    model.eval()
    with pytest.raises(ValueError, match="eval mode"):
        model(ids, cache=cache)
    with torch.no_grad():
        with pytest.raises(ValueError, match="unpadded"):
            model(ids, torch.tensor([[1, 1, 0]]), cache=cache)
        with pytest.raises(ValueError, match="labels"):
            model(ids, labels=ids, cache=cache)
        assert cache.failed
        cache.reset()
        full = model(ids).logits
        cached = torch.cat([model(ids[:, i : i + 1], cache=cache).logits for i in range(3)], 1)
        torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-5)
        with pytest.raises(ValueError, match="another model"):
            MiniQwen4ForCausalLM(tiny_config()).eval()(ids, cache=cache)
        cache.reset()
        model(ids, cache=cache)
        with pytest.raises(ValueError, match="another model"):
            model(ids.expand(2, -1), cache=cache)
        cache.reset()
        model(ids, cache=cache)
        with (
            torch.autocast("cpu", dtype=torch.bfloat16),
            pytest.raises(ValueError, match="another model"),
        ):
            model(ids, cache=cache)
        cache.failed = True
        with pytest.raises(ValueError, match="failed previously"):
            model(ids, cache=cache)
        cache.reset()
        assert cache.length == 0 and cache.layers == []
        torch.testing.assert_close(model(ids, cache=cache).logits, full, atol=0, rtol=0)


def test_partial_forward_invalidates_cache():
    model = MiniQwen4TextModel(tiny_config()).eval()
    cache = MiniQwen4Cache()

    def fail(module, args):
        raise RuntimeError("injected layer failure")

    handle = model.layers[-1].register_forward_pre_hook(fail)
    try:
        with torch.no_grad(), pytest.raises(RuntimeError, match="injected"):
            model(torch.tensor([[3, 4, 5]]), cache=cache)
    finally:
        handle.remove()
    assert cache.failed and cache.length == 0 and cache.layers[0].conv_states
    with torch.no_grad(), pytest.raises(ValueError, match="failed previously"):
        model(torch.tensor([[3]]), cache=cache)
