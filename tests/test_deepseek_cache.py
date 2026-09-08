"""SWA/CSA/HCA boundary decode, bounded tails, source forward and cache rollback."""

import pytest
import torch
from test_new_backbones import tiny_deepseek

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache, MiniDeepSeekV4ForCausalLM


@pytest.mark.parametrize("phase", ["dense_pretrain", "sparse_cpt"])
@pytest.mark.parametrize("prefix", [3, 127, 128, 129])
def test_compressed_cache_matches_full_sequence_at_boundaries(phase, prefix):
    torch.manual_seed(44)
    model = MiniDeepSeekV4ForCausalLM(
        tiny_deepseek(window_size=128, max_seq_len=256), training_phase=phase
    ).eval()
    ids = torch.randint(10, 60, (2, prefix + 7))
    cache = MiniDeepSeekV4Cache()
    with torch.no_grad():
        expected = model(ids).logits
        first = model(ids[:, :prefix], cache=cache).logits
        following = model(ids[:, prefix:], cache=cache).logits
        torch.testing.assert_close(torch.cat((first, following), 1), expected, rtol=3e-5, atol=3e-6)
        for layer, state in zip(model.layers, cache.layers, strict=True):
            assert state["kv"].shape[1] <= 128
            if layer.attn.compress_ratio:
                assert state["tail"].shape[1] <= 2 * layer.attn.compress_ratio
                count = 0 if state["compressed"] is None else state["compressed"].shape[1]
                assert count == ids.shape[1] // layer.attn.compress_ratio
        snapshot = cache.snapshot()
        original = model(ids[:, :2], cache=cache).logits
        cache.restore(snapshot)
        torch.testing.assert_close(original, model(ids[:, :2], cache=cache).logits, rtol=0, atol=0)
        cache.restore(snapshot)
        model.embed.weight.add_(0.001)
        with pytest.raises(ValueError, match="weights"):
            model(ids[:, :1], cache=cache)
        with pytest.raises(ValueError, match="invalidated"):
            model(ids[:, :1], cache=cache)


@pytest.mark.cuda
def test_deepseek_bf16_incremental_crosses_hca_boundary():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(947)
    model = (
        MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(window_size=128, max_seq_len=256), training_phase="sparse_cpt"
        )
        .cuda()
        .eval()
    )
    ids = torch.randint(10, 60, (1, 132), device="cuda")
    cache = MiniDeepSeekV4Cache()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(ids).logits
        first = model(ids[:, :127], cache=cache).logits
        following = model(ids[:, 127:], cache=cache).logits
    with torch.no_grad():
        reference = model(ids).logits.float()
    cached = torch.cat((first, following), 1).float()
    cache_error = (cached - full.float()).square().mean().sqrt()
    amp_error = (full.float() - reference).square().mean().sqrt()
    assert cache_error <= amp_error + 1e-6
    assert cache_error / reference.square().mean().sqrt() < 0.015
    assert (cached.argmax(-1) == full.argmax(-1)).float().mean() >= 0.99
