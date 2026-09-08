"""MTP causality, source fusion oracles, independent denominators and optimizer coverage."""

import copy

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.training.kimi_quantile_balance import KimiQuantileBalance
from minifrontier.training.minideepseekv4_optim import MiniDeepSeekV4Optimizer
from minifrontier.training.minikimik3_optim import MiniKimiK3Optimizer
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer
from minifrontier.training.mtp import shifted_batch
from minifrontier.training.qwen_balance import QwenWindowBalance, normalized_router_loss

FAMILIES = [
    (MiniKimiK3ForCausalLM, tiny_kimi, MiniKimiK3Optimizer),
    (MiniDeepSeekV4ForCausalLM, tiny_deepseek, MiniDeepSeekV4Optimizer),
    (MiniQwen4ForCausalLM, tiny_config, MiniQwen4Optimizer),
]


def test_future_shift_masks_do_not_cross_eos_images_or_padding():
    ids = torch.tensor([[1, 11, 12, 2, 21, 7, 22, 23, 2, 0, 0]])
    shifted, target, mask = shifted_batch(ids, ids, vocab_size=32)
    assert target.tolist() == [[12, 2, -100, -100, -100, -100, 2, -100, -100, -100, -100]]
    assert shifted[0, mask[0]].tolist() == [11, 12, 23]
    assert target.ne(-100).sum() == 3


@pytest.mark.parametrize("cls,config,optimizer", FAMILIES)
def test_mtp_zero_weight_identity_gradient_and_future_causality(cls, config, optimizer):
    torch.manual_seed(33)
    base = cls(config())
    torch.manual_seed(33)
    model = cls(config(mtp_enabled=True, mtp_loss_coef=0.0))
    ids = torch.randint(10, model.config.vocab_size, (2, 16))
    base.eval()
    model.eval()
    torch.testing.assert_close(base(ids).logits, model(ids).logits, rtol=0, atol=0)
    model.config.mtp_loss_coef = 0.1
    if hasattr(model, "configure_training_phase"):
        model.configure_training_phase("dense_pretrain")
    elif hasattr(model, "_configure_training_phase"):
        model._configure_training_phase("dense_pretrain")
    else:
        model.mtp.requires_grad_(True)
        for p in model.mtp.parameters():
            if p.ndim == 1 and p.shape[0] == model.config.num_experts:
                p.requires_grad_(False)
    # Capture the draft's pre-head predictions, not its aggregated scalar loss.
    seen = []
    handle = model.mtp.register_forward_hook(
        lambda module, args, output: seen.append(output[0] if isinstance(output, tuple) else output)
    )
    model(ids, labels=ids)
    changed = ids.clone()
    changed[:, 7] = (ids[:, 7] - 9) % (model.config.vocab_size - 10) + 10
    model(changed, labels=changed)
    handle.remove()
    torch.testing.assert_close(seen[0][:, :6], seen[1][:, :6])
    model.train()
    result = model(ids, labels=ids, return_logits=False)
    assert result.mtp_tokens == 28 and torch.isfinite(result.mtp_loss)
    result.loss.backward()
    kwargs = dict(lr=1e-3, adam_lr=1e-3) if optimizer != MiniDeepSeekV4Optimizer else dict(lr=1e-3)
    opt = optimizer(model, **kwargs)
    opt.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mtp.parameters())


def test_deepseek_concat_fusion_is_v4_separate_projection_per_stream():
    model = MiniDeepSeekV4ForCausalLM(tiny_deepseek(mtp_enabled=True))
    module = model.mtp
    h = torch.randn(2, 7, model.config.hc_mult, model.config.dim)
    e = torch.randn(2, 7, model.config.dim)
    nh, ne = module.hnorm(h), module.enorm(e)[:, :, None].expand_as(h)
    w_h, w_e = module.fusion.weight.chunk(2, -1)
    fused = module.fusion(torch.cat((nh, ne), -1))
    separate = torch.nn.functional.linear(nh, w_h) + torch.nn.functional.linear(ne, w_e)
    torch.testing.assert_close(fused, separate)
    assert not torch.equal(fused[:, :, 0], fused[:, :, 1])
    assert not any("embed.weight" in n or n.endswith("head.weight") for n in module.state_dict())


def test_qwen_keeps_multistream_layout_and_independent_prediction_head():
    model = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=True))
    mtp = model.mtp
    assert mtp.pre_fc_norm_hidden.weight.numel() == model.config.hc_count * model.config.hidden_size
    assert mtp.shared_head.head.weight is not model.lm_head.weight
    assert mtp.block.ple is None and mtp.block.layer_type == "full_attention"
    model.transition_training_phase("dense_distill")
    assert mtp.block.self_attn.indexer.index_qk_proj.weight.requires_grad
    ids = torch.randint(10, model.config.vocab_size, (2, 16))
    result = model(ids, labels=ids)
    assert result.mtp_aux_loss is not None and result.mtp_loss is None
    result.loss.backward()
    assert all(".indexer." in name for name, p in model.named_parameters() if p.grad is not None)


def test_qwen_window_frequency_gives_large_batch_source_value_and_gradient():
    torch.manual_seed(4)
    x = torch.randn(2, 13, 5, requires_grad=True)
    valid = torch.ones((2, 13), dtype=torch.bool)
    valid[1, 4:] = False
    routers = tuple(v.reshape(-1, 5) for v in x)
    expected = normalized_router_loss(routers, 5, 2, valid[0])
    # Distinct per-microbatch lengths with two independent layers.
    chunks = [
        (x[:, :4], torch.ones(4, dtype=torch.bool)),
        (x[:, 4:], torch.ones(9, dtype=torch.bool)),
    ]
    counts = torch.zeros(5)
    for logits, _mask in chunks:
        counts += torch.bincount(
            logits.detach().softmax(-1).topk(2, -1).indices.flatten(), minlength=5
        )
    frequency = counts / 26
    total = 0
    for logits, mask in chunks:
        total = (
            total
            + normalized_router_loss(tuple(logits), 5, 2, mask, frequency) * logits.shape[1] / 13
        )
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(
        torch.autograd.grad(total, x)[0], torch.autograd.grad(expected, x)[0]
    )


def test_qwen_window_replay_and_kimi_qb_include_mtp_without_recount():
    qwen = MiniQwen4ForCausalLM(tiny_config(mtp_enabled=True))
    x = torch.randint(10, qwen.config.vocab_size, (2, 12))
    balance = QwenWindowBalance(qwen)
    with torch.no_grad(), balance.capture(x.ne(0)):
        qwen(x, labels=x)
    assert balance.totals["main"][-1] == x.numel() * qwen.config.num_hidden_layers
    assert balance.totals["mtp"][-1] == 20
    balance.finalize()
    qwen(x, labels=x).loss.backward()
    balance.clear()
    kimi = MiniKimiK3ForCausalLM(tiny_kimi(mtp_enabled=True))
    qb = KimiQuantileBalance(kimi)
    with qb.capture(x.ne(0)):
        result = kimi(x, labels=x)
    before = copy.deepcopy(qb.state_dict())
    result.loss.backward()
    for name, _, histogram in qb.gates:
        assert int(histogram.counts[0].sum()) == (20 if name.startswith("mtp.") else 24)
        torch.testing.assert_close(histogram.counts, before[name]["counts"])
