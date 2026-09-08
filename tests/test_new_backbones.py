"""Causality, gradients, source arithmetic and CUDA KDA acceptance."""

import ast
import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Config, MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.attention import Compressor
from minifrontier.models.minideepseekv4.kernels import hc_split_sinkhorn
from minifrontier.models.minikimik3 import MiniKimiK3Config, MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3 import upstream_layers as kimi
from minifrontier.models.minikimik3.kernels import chunk_kda, reference_kda

ROOT = Path(__file__).resolve().parents[1]


def tiny_kimi(**kwargs):
    values = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        kda_head_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        max_position_embeddings=256,
    )
    return MiniKimiK3Config(**dict(values, **kwargs))


def tiny_deepseek(**kwargs):
    values = dict(
        vocab_size=64,
        dim=32,
        n_layers=3,
        n_heads=2,
        head_dim=16,
        rope_head_dim=8,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=16,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        n_hash_layers=1,
        compress_ratios=(0, 4, 128),
        index_n_heads=2,
        index_head_dim=16,
        max_seq_len=256,
    )
    return MiniDeepSeekV4Config(**dict(values, **kwargs))


@pytest.mark.parametrize("family", ["kimi", "deepseek"])
def test_backbone_is_causal_and_has_finite_trainable_gradients(family):
    torch.manual_seed(82)
    model = (
        MiniKimiK3ForCausalLM(tiny_kimi())
        if family == "kimi"
        else MiniDeepSeekV4ForCausalLM(tiny_deepseek())
    )
    x = torch.randint(3, 64, (2, 20))
    model.eval()
    changed = x.clone()
    changed[:, 11:] = torch.randint(3, 64, (2, 9))
    with torch.no_grad():
        torch.testing.assert_close(
            model(x).logits[:, :11], model(changed).logits[:, :11], atol=1e-6, rtol=1e-5
        )
    model.train()
    result = model(x, labels=x)
    result.loss.backward()
    missing = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
        elif p.requires_grad and ".compressor." not in name:
            missing.append(name)
    assert not missing, missing
    before = copy.deepcopy(model.state_dict())
    torch.optim.AdamW(model.parameters(), lr=0.001).step()
    assert any(
        p.is_floating_point() and not torch.equal(p, before[name])
        for name, p in model.state_dict().items()
    )


def test_deepseek_indexer_phases_freeze_backbone_and_train_compression():
    torch.manual_seed(14)
    model = MiniDeepSeekV4ForCausalLM(tiny_deepseek())
    x = torch.randint(3, 64, (2, 20))
    before = copy.deepcopy(model.state_dict())
    model.configure_training_phase("dense_distill")
    model(x, labels=x).loss.backward()
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01).step()
    changed = [name for name, p in model.state_dict().items() if not torch.equal(p, before[name])]
    assert changed and all(".indexer." in name for name in changed)
    model.configure_training_phase("sparse_cpt")
    result = model(x, labels=x)
    result.loss.backward()
    assert result.indexer_loss >= 0
    assert model.layers[1].attn.compressor.wkv.weight.grad.abs().sum() > 0
    assert model.layers[1].attn.indexer.wq_b.weight.grad.abs().sum() > 0


def kimi_oracle():
    path = ROOT / "third_party/upstream/kimi-k3-c5d1dd4/modeling_kimi_linear.py"
    names = {
        "SituAndMul",
        "_get_situ_activation_params",
        "KimiRMSNorm",
        "KimiBlockSparseMLP",
        "KimiMLP",
        "KimiMoEGate",
        "KimiSparseMoeBlock",
        "repeat_kv",
        "eager_attention_forward",
        "KimiMLAAttention",
    }
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    import math

    namespace = dict(torch=torch, nn=nn, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_kimi_mla_and_latent_moe_match_original_inference():
    torch.manual_seed(27)
    config = tiny_kimi().upstream_config()
    oracle = kimi_oracle()
    for cls_name in ("KimiMLAAttention", "KimiSparseMoeBlock"):
        extra = (0,) if cls_name == "KimiMLAAttention" else ()
        local = getattr(kimi, cls_name)(config, *extra).eval()
        if cls_name == "KimiSparseMoeBlock":
            with torch.no_grad():
                local.gate.e_score_correction_bias.zero_()
        original = oracle[cls_name](config, *extra).eval()
        original.load_state_dict(local.state_dict())
        x = torch.randn(2, 9, 32)
        torch.testing.assert_close(local(x), original(x), rtol=1e-5, atol=1e-6)
        local.train()
        x.requires_grad_()
        local(x).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        if cls_name == "KimiSparseMoeBlock":
            assert local.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_matches_pinned_prefill_without_quantization(ratio):
    import math
    from contextlib import contextmanager
    from functools import lru_cache

    from minifrontier.models.minideepseekv4 import upstream_layers as ds

    path = ROOT / "third_party/upstream/deepseek-v4-60d8d70/model.py"
    names = {"Compressor", "apply_rotary_emb"}
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        contextmanager=contextmanager,
        lru_cache=lru_cache,
        Linear=ds.Linear,
        RMSNorm=ds.RMSNorm,
        act_quant=lambda *args: None,
        scale_fmt=None,
        scale_dtype=torch.float32,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    config = tiny_deepseek().upstream_config()
    local = Compressor(config, ratio, 16)
    original = namespace["Compressor"](config, ratio, 16)
    original.load_state_dict(local.state_dict(), strict=False)
    freqs = ds.precompute_freqs_cis(8, 256, 0, 40000, 1, 32, 1)
    original.freqs_cis = freqs
    original.kv_cache = torch.zeros(1, 256 // ratio, 16)
    x = torch.randn(1, 2 * ratio, 32)
    with torch.no_grad():
        expected = original(x.clone(), 0)
    torch.testing.assert_close(local(x, freqs), expected, rtol=1e-5, atol=1e-6)
    local(x.requires_grad_(), freqs).square().mean().backward()
    assert x.grad.isfinite().all()


def test_sinkhorn_stochasticity_and_gradients():
    x = torch.randn(2, 7, 24, requires_grad=True)
    pre, post, matrix = hc_split_sinkhorn(x, torch.ones(3), torch.zeros(24))
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 7, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2, 7, 4), atol=1e-4, rtol=0)
    (pre.square().sum() + post.square().sum() + matrix.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.cuda
def test_cuda_kda_matches_cpu_recurrence_and_gradients():
    torch.manual_seed(713)
    args = dict(
        q=torch.randn(1, 32, 2, 32),
        k=torch.randn(1, 32, 2, 32),
        v=torch.randn(1, 32, 2, 32),
        g=torch.randn(1, 32, 2, 32) * 0.3,
        beta=torch.randn(1, 32, 2),
        A_log=torch.zeros(2),
        dt_bias=torch.full((64,), -2.0),
    )
    device = {
        k: v.cuda()
        .to(torch.bfloat16 if k in {"q", "k", "v", "g"} else torch.float32)
        .requires_grad_()
        for k, v in args.items()
    }
    cpu = {k: v.detach().cpu().float().requires_grad_() for k, v in device.items()}
    expected, _ = reference_kda(**cpu)
    actual, _ = chunk_kda(
        **device,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.03, atol=0.005)
    actual.float().square().mean().backward()
    expected.square().mean().backward()
    for name, value in device.items():
        assert value.grad is not None and value.grad.isfinite().all()
        torch.testing.assert_close(value.grad.float().cpu(), cpu[name].grad, rtol=0.08, atol=3e-5)
