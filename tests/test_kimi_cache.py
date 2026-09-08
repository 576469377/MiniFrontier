"""Incremental KDA/conv/MLA and AttnRes, prefix chunks and rejected-draft rollback."""

from unittest.mock import patch

import pytest
import torch
from test_new_backbones import tiny_kimi

from minifrontier.models.minikimik3 import MiniKimiK3Cache, MiniKimiK3ForCausalLM


@pytest.mark.parametrize("chunk_sizes", [(11, 1, 1, 1, 1, 1), (3, 4, 2, 7)])
def test_kimi_incremental_full_prefix_and_rollback(chunk_sizes):
    torch.manual_seed(731)
    model = MiniKimiK3ForCausalLM(tiny_kimi()).eval()
    ids = torch.randint(10, 60, (2, 16))
    cache = MiniKimiK3Cache()
    with torch.no_grad():
        full = model(ids).logits
        parts = []
        start = 0
        for size in chunk_sizes:
            parts.append(model(ids[:, start : start + size], cache=cache).logits)
            start += size
        torch.testing.assert_close(torch.cat(parts, 1), full, rtol=2e-5, atol=2e-6)
        assert cache.length == 16
        snapshot = cache.snapshot()
        a = model(ids[:, :2], cache=cache).logits
        cache.restore(snapshot)
        b = model(ids[:, :2], cache=cache).logits
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        cache.restore(snapshot)
        other = MiniKimiK3ForCausalLM(tiny_kimi()).eval()
        with pytest.raises(ValueError, match="owner"):
            other(ids[:, :1], cache=cache)
        with pytest.raises(ValueError, match="invalidated"):
            model(ids[:, :1], cache=cache)
        cache.reset()
        torch.testing.assert_close(model(ids, cache=cache).logits, full, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
def test_kimi_bf16_cuda_incremental_matches_fla_chunk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(731)
    model = MiniKimiK3ForCausalLM(tiny_kimi(router_fp32=True)).cuda().eval()
    ids = torch.randint(10, 60, (2, 19), device="cuda")
    cache = MiniKimiK3Cache()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(ids).logits
        first = model(ids[:, :11], cache=cache).logits
        following = [model(ids[:, i : i + 1], cache=cache).logits for i in range(11, 19)]
    # Compare to the FP32 recurrence oracle: chunk and single-token kernels
    # accumulate differently in BF16. Near-zero logits make pointwise rtol misleading.
    from minifrontier.models.minikimik3.kernels import reference_kda

    with (
        torch.no_grad(),
        patch("minifrontier.models.minikimik3.upstream_layers.chunk_kda", reference_kda),
    ):
        reference = model(ids).logits.float()
    cached = torch.cat([first, *following], 1).float()
    cache_error = (cached - full.float()).square().mean().sqrt()
    amp_error = (full.float() - reference).square().mean().sqrt()
    assert cache_error <= amp_error + 1e-6
    assert cache_error / reference.square().mean().sqrt() < 0.015
    assert (cached.argmax(-1) == full.argmax(-1)).float().mean() >= 0.99
