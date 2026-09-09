"""Independent mathematical contracts required before strategy pilots."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_new_backbones import tiny_deepseek, tiny_kimi
from torch import nn

from minifrontier.inference.runtime import generate_ids
from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.attention import Attention
from minifrontier.models.minideepseekv4.expert import sequence_balance_loss
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.storage import StorageLimitError, require_space
from minifrontier.training.budgets import scaled_ce
from minifrontier.training.deepseek_opd import full_vocab_reverse_kl, reverse_kl_from_logits
from minifrontier.training.kimi_quantile_balance import (
    KimiQuantileBalance,
    QuantileHistogram,
    exact_quantile_bias,
)
from minifrontier.training.losses import causal_lm_loss
from minifrontier.training.minideepseekv4_optim import HYBRID_COEFFICIENTS, MiniDeepSeekV4Optimizer
from minifrontier.training.minikimik3_optim import MiniKimiK3Optimizer
from minifrontier.training.posttrain import token_log_probs
from minifrontier.training.semantic_optim import newton_schulz


def test_histogram_matches_exact_and_accumulation_resume_with_ties():
    torch.manual_seed(81)
    scores = torch.rand(513, 8)
    scores[:17] = 0.5
    old = torch.linspace(-0.13, 0.13, 8)
    expected = exact_quantile_bias(scores, old, 2)
    bias = old.clone()
    histogram = QuantileHistogram(bias, 2, bins=4096)
    selected = (scores + old).topk(2, dim=-1).indices
    histogram.add(scores[:200], selected[:200])
    saved = copy.deepcopy(histogram.state_dict())
    resumed = QuantileHistogram(old.clone(), 2, bins=4096)
    resumed.load_state_dict(saved)
    for h in (histogram, resumed):
        h.add(scores[200:], selected[200:])
        metrics = h.update()
        assert metrics["tokens"] == 513 and metrics["overflow"] == 0
        torch.testing.assert_close(h.bias, expected, atol=2 * metrics["bin_width"], rtol=0)
    torch.testing.assert_close(histogram.bias, resumed.bias, rtol=0, atol=0)
    assert histogram.counts.sum() == 0
    assert exact_quantile_bias(torch.full((3, 8), 0.5), torch.zeros(8), 2).abs().max() == 0


def test_qb_excludes_padding_and_checkpoint_replay():
    m = MiniKimiK3ForCausalLM(tiny_kimi())
    balance = KimiQuantileBalance(m)
    x = torch.randint(3, 64, (2, 12))
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[1, 7:] = False
    with balance.capture(mask):
        result = m(x, labels=x, attention_mask=mask)
    result.loss.backward()
    for _, _, histogram in balance.gates:
        assert histogram.counts[0].sum() == 19
    before = [g.e_score_correction_bias.clone() for _, g, _ in balance.gates]
    balance.update()
    assert any(
        not torch.equal(p, g.e_score_correction_bias)
        for p, (_, g, _) in zip(before, balance.gates, strict=True)
    )


def test_hybrid_ns_against_svd_for_well_conditioned_matrix_and_zero():
    torch.manual_seed(4)
    x = torch.randn(8, 12, dtype=torch.float64)
    u, _, v = torch.linalg.svd(x, full_matrices=False)
    expected = u @ v
    actual = newton_schulz(x, HYBRID_COEFFICIENTS)
    torch.testing.assert_close(actual.double(), expected, atol=0.002, rtol=0.002)
    assert torch.count_nonzero(newton_schulz(torch.zeros(3, 4), HYBRID_COEFFICIENTS)) == 0


@pytest.mark.parametrize("kind", ["kimi", "deepseek"])
def test_semantic_optimizer_coverage_head_layout_and_exact_resume(kind):
    m = (
        MiniKimiK3ForCausalLM(tiny_kimi())
        if kind == "kimi"
        else MiniDeepSeekV4ForCausalLM(tiny_deepseek())
    )
    cls = MiniKimiK3Optimizer if kind == "kimi" else MiniDeepSeekV4Optimizer
    opt = cls(m, lr=0.003, adam_lr=0.0003)
    assert len(opt.param_groups) == sum(p.requires_grad for p in m.parameters())
    if kind == "kimi":
        group = next(g for g in opt.param_groups if g["name"] == "layers.0.self_attn.q_proj.weight")
        assert group["blocks"] == ((0, 16), (16, 32))
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 0.01
    opt.step()
    weights, state = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    opt.step()
    expected = copy.deepcopy(m.state_dict())
    m.load_state_dict(weights)
    opt.load_state_dict(state)
    opt.step()
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


def test_sequence_balance_uses_unbiased_scores_and_real_sequences():
    scores = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0]] * 3, [[10.0, 1.0, 1.0, 1.0]] * 3], requires_grad=True
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    loss = sequence_balance_loss(scores, mask, 1)
    expected = (1 + 4 * 10 / 13) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert scores.grad[~mask].abs().sum() == 0
    assert sequence_balance_loss(scores, torch.zeros_like(mask), 1) == 0


def test_variable_length_ce_accumulation_equals_large_batch_and_ddp_average():
    torch.manual_seed(32)
    head = nn.Linear(7, 13)
    x = torch.randn(3, 9, 7)
    labels = torch.randint(0, 13, (3, 9))
    labels[0, :7] = -100
    labels[1, 5:] = -100
    causal_lm_loss(head(x), labels).backward()
    expected = head.weight.grad.clone()
    head.zero_grad()
    total = int(labels[:, 1:].ne(-100).sum())
    for rank_ids in ([0], [1, 2]):
        for i in rank_ids:
            y = labels[i : i + 1]
            # Simulated two-rank gradient averaging after local scaled sums.
            (
                scaled_ce(causal_lm_loss(head(x[i : i + 1]), y), y[:, 1:].ne(-100).sum(), total, 2)
                / 2
            ).backward()
    torch.testing.assert_close(head.weight.grad, expected)


def test_behavior_logp_matches_policy_recomputation_including_eos():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(12, 12)
            self.config = SimpleNamespace(max_seq_len=32)

        def forward(self, x):
            return SimpleNamespace(logits=self.embed(x))

    torch.manual_seed(29)
    m = Toy()
    prompt = torch.tensor([[1, 4, 5], [1, 4, 5]])
    ids, behavior, active = generate_ids(
        m, prompt, max_new_tokens=8, temperature=1, top_p=1, return_behavior=True
    )
    labels = ids.clone()
    labels[:, :3] = -100
    labels[:, 3:] = labels[:, 3:].masked_fill(~active, -100)
    recomputed, _ = token_log_probs(m(ids).logits, labels, actions=True)
    torch.testing.assert_close(recomputed[:, 2:][active], behavior[active])


def test_position_chunked_full_vocab_opd_matches_dense_value_and_gradient():
    torch.manual_seed(77)
    head, teacher_head = nn.Linear(8, 17), nn.Linear(8, 17)
    teacher_head.requires_grad_(False)
    s = torch.randn(2, 7, 8, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    dense = reverse_kl_from_logits(head(s), teacher_head(t))[mask].mean()
    gradients = torch.autograd.grad(dense, (s, head.weight), retain_graph=True)
    chunked = full_vocab_reverse_kl(s, t, head, teacher_head, mask, chunk_size=3)
    actual = torch.autograd.grad(chunked, (s, head.weight))
    torch.testing.assert_close(chunked, dense)
    for a, e in zip(actual, gradients, strict=True):
        torch.testing.assert_close(a, e)
    assert t.grad is None and teacher_head.weight.grad is None


def test_disk_reserve_counts_atomic_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "minifrontier.storage.shutil.disk_usage", lambda p: SimpleNamespace(free=100)
    )
    assert require_space(tmp_path, 60, reserve_bytes=40) == 100
    with pytest.raises(StorageLimitError):
        require_space(tmp_path, 61, reserve_bytes=40)


def test_hca_window128_restores_early_token_direct_path():
    torch.manual_seed(37)
    cfg = tiny_deepseek(window_size=128).upstream_config()
    a = Attention(2, cfg)
    b = copy.deepcopy(a)
    b.window_size = 64
    # Source adapters are initialized by the enclosing model, not Attention.
    for p in a.parameters():
        nn.init.normal_(p, std=0.1)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 97, cfg.dim)
    changed = x.clone()
    changed[:, 0] += torch.randn(cfg.dim) * 3
    torch.testing.assert_close(b(x)[:, -1], b(changed)[:, -1], atol=0, rtol=0)
    assert (a(x)[:, -1] - a(changed)[:, -1]).abs().max() > 1e-7
