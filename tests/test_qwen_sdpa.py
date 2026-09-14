"""Dense Qwen kernel parity and a bounded, explicitly requested CUDA benchmark.

Run the CUDA measurement only on an allocated GPU:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python tests/test_qwen_sdpa.py
"""

import copy
import json
import statistics

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_qwen_prepass import model_and_window

from minifrontier.models.miniqwen4 import MiniQwen4Cache, MiniQwen4Config, MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.attention import configure_dense_attention
from minifrontier.models.miniqwen4.modeling import MiniQwen4StageIndexer
from minifrontier.models.miniqwen4.upstream_decoder import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def relative_error(actual, expected):
    difference = actual.detach().float() - expected.detach().float()
    return float(difference.norm() / expected.detach().float().norm().clamp_min(1e-12))


def gradient_error(actual, expected):
    differences, references = [], []
    for (name, param), (other, reference) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other
        assert (param.grad is None) == (reference.grad is None), name
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
            differences.append((param.grad.float() - reference.grad.float()).square().sum())
            references.append(reference.grad.float().square().sum())
    return float(torch.stack(differences).sum().sqrt() / torch.stack(references).sum().sqrt())


def attention_pair(config, device, batch, length, padded):
    args = config.upstream_config()
    eager = Qwen4ExpTextAttention(args, 0).to(device).train()
    eager.indexer = MiniQwen4StageIndexer(args, 0).to(device)
    eager.indexer.requires_grad_(False)
    sdpa = copy.deepcopy(eager)
    configure_dense_attention(sdpa, "sdpa")
    hidden = torch.randn(batch, length, config.hidden_size, device=device)
    rotary = Qwen4ExpTextRotaryEmbedding(args).to(device)
    positions = torch.arange(length, device=device)[None, None].expand(3, batch, -1).clone()
    # Separate visual spatial positions from the temporal/text channel.
    positions[1, :, 2:6] = 2
    positions[2, :, 2:6] = torch.tensor([2, 3, 2, 3], device=device)
    embeddings = rotary(hidden, positions)
    valid = torch.ones(batch, length, device=device, dtype=torch.bool)
    if padded:
        valid[-1, length * 3 // 4 :] = False
    legal = torch.ones(length, length, device=device, dtype=torch.bool).tril()[None]
    legal = legal & valid[:, None]
    # Match the MTP mask: invalid queries still have a diagonal key.
    legal |= torch.eye(length, device=device, dtype=torch.bool)[None]
    mask = torch.zeros_like(legal, dtype=hidden.dtype).masked_fill(~legal, float("-inf"))[:, None]
    return eager, sdpa, hidden, embeddings, mask


def compare_attention(config, device, batch, length, padded, autocast):
    torch.manual_seed(2941)
    eager, sdpa, hidden, embeddings, mask = attention_pair(config, device, batch, length, padded)
    reference_input = hidden.clone().requires_grad_()
    optimized_input = hidden.clone().requires_grad_()
    upstream_gradient = torch.randn_like(hidden)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        expected, probabilities = eager(reference_input, embeddings, mask)
        actual, unused_probabilities = sdpa(optimized_input, embeddings, mask)
    assert probabilities is not None and unused_probabilities is None
    expected.backward(upstream_gradient)
    actual.backward(upstream_gradient)
    metrics = {
        "output_relative_l2": relative_error(actual, expected),
        "input_gradient_relative_l2": relative_error(optimized_input.grad, reference_input.grad),
        "parameter_gradient_relative_l2": gradient_error(sdpa, eager),
    }
    limit = 0.02 if autocast else 2e-5
    assert max(metrics.values()) < limit, metrics
    if not autocast:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    return metrics, (eager, sdpa, hidden, embeddings, mask, upstream_gradient)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_attention_outputs_and_gradients_match(autocast, padded):
    compare_attention(tiny_config(), torch.device("cpu"), 2, 32, padded, autocast)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("media_kind", [None, "image", "video"])
def test_full_model_mtp_and_media_gradients(autocast, checkpointing, media_kind):
    torch.manual_seed(4128)
    eager, window = model_and_window(checkpointing, media_kind)
    sdpa = copy.deepcopy(eager)
    params = {name: id(p) for name, p in sdpa.named_parameters()}
    sdpa.set_dense_attention_backend("sdpa")
    assert {name: id(p) for name, p in sdpa.named_parameters()} == params
    assert eager.state_dict().keys() == sdpa.state_dict().keys()
    for ids, labels, media in window:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            expected = eager(ids, ids.ne(0), labels, media=media, return_logits=False)
            actual = sdpa(ids, ids.ne(0), labels, media=media, return_logits=False)
        for key in ("loss", "lm_loss", "aux_loss", "mtp_loss", "mtp_aux_loss"):
            torch.testing.assert_close(
                getattr(actual, key),
                getattr(expected, key),
                atol=2e-4 if autocast else 1e-6,
                rtol=2e-3 if autocast else 1e-5,
            )
        expected.loss.backward()
        actual.loss.backward()
    assert gradient_error(sdpa, eager) < (0.02 if autocast else 2e-5)


def test_phase_boundaries_restore_probabilities_for_main_and_mtp():
    model = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=True))
    ids = torch.tensor([[11, 12, 13, 14, 15, 16]])
    modules = [m for m in model.modules() if isinstance(m, Qwen4ExpTextAttention)]
    assert len(modules) == 2
    assert not any(m.dense_sdpa_enabled for m in modules)
    for phase in ("dense_pretrain", "dense_distill", "sparse_cpt"):
        if model.training_phase != phase:
            model.transition_training_phase(phase)
        model.set_dense_attention_backend("sdpa")
        assert all(m.dense_sdpa_enabled == (phase == "dense_pretrain") for m in modules)
        output = model(ids, labels=ids, return_logits=False)
        assert torch.isfinite(output.loss)
        if phase != "dense_pretrain":
            assert output.indexer_loss is not None and torch.isfinite(output.indexer_loss)
        output.loss.backward()


def test_backend_switch_keeps_checkpoint_keys_and_rng():
    model = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=True))
    original = copy.deepcopy(model)
    before = torch.get_rng_state()
    model.set_dense_attention_backend("sdpa")
    model.set_dense_attention_backend("eager")
    assert torch.equal(before, torch.get_rng_state())
    original.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[11, 12, 13, 14]])
    torch.testing.assert_close(model(ids).logits, original(ids).logits, atol=0, rtol=0)
    with pytest.raises(ValueError, match="eager or sdpa"):
        model.set_dense_attention_backend("unknown")


@torch.no_grad()
def test_opt_in_keeps_cached_inference_on_eager():
    eager = MiniQwen4ForCausalLM(tiny_config()).eval()
    sdpa = copy.deepcopy(eager)
    sdpa.set_dense_attention_backend("sdpa")
    old_cache, new_cache = MiniQwen4Cache(), MiniQwen4Cache()
    for ids in (torch.tensor([[11, 12, 13]]), torch.tensor([[14]])):
        expected = eager(ids, cache=old_cache).logits
        actual = sdpa(ids, cache=new_cache).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert old_cache.length == new_cache.length == 4


def benchmark_cuda():
    """Five timed operator updates per kernel; no optimizer or training artifact."""
    device = torch.device("cuda")
    torch.set_num_threads(4)
    metrics, values = compare_attention(MiniQwen4Config(), device, 8, 2048, True, True)
    eager, sdpa, hidden, embeddings, mask, upstream_gradient = values
    results = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "seed": 2941,
        "shape": {"batch": 8, "sequence": 2048, "hidden": 512, "query_heads": 8, "kv_heads": 2},
        "precision": "fp32 parameters and inputs, bfloat16 autocast",
        "mask": "causal, right-padded last sample, MTP diagonal allowance",
        "scope": "one Qwen dense attention forward and backward; no optimizer",
        "warmup_per_backend": 2,
        "timed_repeats_per_backend": 5,
        "comparison": metrics,
    }
    for name, module in (("eager", eager), ("sdpa", sdpa)):
        timings = []
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        for iteration in range(7):
            module.zero_grad(set_to_none=True)
            x = hidden.detach().requires_grad_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = module(x, embeddings, mask)
            output.backward(upstream_gradient)
            end.record()
            end.synchronize()
            if iteration >= 2:
                timings.append(start.elapsed_time(end))
            del output, _
        results[name] = {
            "forward_backward_ms": statistics.median(timings),
            "peak_above_baseline_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    benchmark_cuda()
