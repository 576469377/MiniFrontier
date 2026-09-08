"""MX golden encodings, nonuniform levels, scale blocks, STE and module scope."""

import pytest
import torch
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.training.mx_quant import (
    MXLinear,
    fake_mx,
    fp4_codes,
    fp4_values,
    hadamard,
    quantize_mx,
)


def test_e2m1_golden_all_codes_ties_and_saturation():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    torch.testing.assert_close(fp4_codes(values), torch.arange(8, dtype=torch.uint8))
    torch.testing.assert_close(fp4_codes(-values), torch.arange(8, 16, dtype=torch.uint8))
    torch.testing.assert_close(
        fp4_values(torch.arange(16, dtype=torch.uint8)), torch.cat((values, -values))
    )
    midpoint = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 10.0])
    torch.testing.assert_close(
        fp4_codes(midpoint), torch.tensor([0, 2, 2, 4, 4, 6, 6, 7], dtype=torch.uint8)
    )


def test_e8m0_scale_axis_and_ste():
    x = torch.linspace(-6, 6, 64).reshape(2, 32).requires_grad_()
    decoded, _, scale = quantize_mx(x)
    assert torch.equal(scale, torch.full((2, 1), 127, dtype=torch.uint8))
    torch.testing.assert_close(decoded, fp4_values(fp4_codes(x)))
    fake_mx(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    transposed = fake_mx(x.detach().T, axis=0)
    torch.testing.assert_close(transposed.T, decoded)
    torch.testing.assert_close(hadamard(hadamard(x)), x, rtol=1e-5, atol=2e-6)
    zero, _, scale = quantize_mx(torch.zeros(2, 32))
    assert zero.count_nonzero() == 0 and (scale == 1).all()
    fp8, _, _ = quantize_mx(x, bits=8)
    assert torch.isfinite(fp8).all()


def test_kimi_qat_scope_preserves_weights_and_shared_experts():
    config = tiny_kimi(
        hidden_size=32,
        routed_expert_hidden_size=32,
        moe_intermediate_size=32,
        qat_scheme="mxfp4-mxfp8-v1",
    )
    model = MiniKimiK3ForCausalLM(config)
    names = [name for name, module in model.named_modules() if isinstance(module, MXLinear)]
    assert names and all(".experts." in n and "shared" not in n for n in names)
    x = torch.randint(10, 60, (1, 8))
    model(x, labels=x).loss.backward()
    assert model.get_submodule(names[0]).weight.dtype == torch.float32
    assert any(p.grad is not None for n, p in model.named_parameters() if ".experts." in n)


def test_deepseek_qat_cache_indexer_matches_full():
    from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache

    config = tiny_deepseek(
        dim=32,
        moe_inter_dim=32,
        index_head_dim=32,
        qat_scheme="mxfp4-indexer-v1",
        max_seq_len=256,
        window_size=128,
    )
    model = MiniDeepSeekV4ForCausalLM(config, training_phase="sparse_cpt").eval()
    ids = torch.randint(10, 60, (1, 135))
    cache = MiniDeepSeekV4Cache()
    with torch.no_grad():
        full = model(ids).logits
        a = model(ids[:, :127], cache=cache).logits
        b = model(ids[:, 127:], cache=cache).logits
    torch.testing.assert_close(torch.cat((a, b), 1), full, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["kimi", "deepseek"])
def test_qat_mtp_bf16_cuda_gradients(family):
    if family == "kimi":
        config = tiny_kimi(
            hidden_size=32,
            routed_expert_hidden_size=32,
            moe_intermediate_size=32,
            mtp_enabled=True,
            qat_scheme="mxfp4-mxfp8-v1",
        )
        model = MiniKimiK3ForCausalLM(config).cuda()
    else:
        config = tiny_deepseek(
            dim=32,
            moe_inter_dim=32,
            index_head_dim=32,
            mtp_enabled=True,
            qat_scheme="mxfp4-indexer-v1",
        )
        model = MiniDeepSeekV4ForCausalLM(config).cuda()
    ids = torch.randint(10, 60, (1, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(ids, labels=ids)
    output.loss.backward()
    assert torch.isfinite(output.loss) and output.mtp_tokens > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.mtp.parameters())
