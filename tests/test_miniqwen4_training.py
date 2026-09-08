"""Source LM training contracts. These tests are not formal model training."""

import ast
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from test_miniqwen4 import ROOT, tiny_config, upstream_namespace
from torch import nn

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM


def official_balance_loss():
    path = ROOT / "third_party/upstream/qwen4_exp-4177486/modeling_qwen4_exp.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if getattr(n, "name", None) == "load_balancing_loss_func"
    )
    port = ast.parse((ROOT / "minifrontier/models/miniqwen4/upstream_loss.py").read_text())
    copied = next(n for n in port.body if getattr(n, "name", None) == node.name)
    assert ast.dump(node) == ast.dump(copied)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_dense_lm_loss_and_all_gradients_match_source(checkpointing, autocast):
    torch.manual_seed(113)
    cfg = tiny_config(gradient_checkpointing=checkpointing)
    local = MiniQwen4ForCausalLM(cfg).train()
    reference = upstream_namespace()["Qwen4ExpTextModel"](cfg.upstream_config()).train()
    reference.load_state_dict(local.model.state_dict(), strict=True)
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    head.load_state_dict(local.lm_head.state_dict())
    captured = []
    for layer in reference.layers:
        if hasattr(layer, "self_attn"):
            # The report's full-attention stage has no sparse selection. Keep
            # all original parameters and all remaining attention computations.
            layer.self_attn.indexer.forward = lambda hidden, rotary, mask, cache: torch.zeros_like(
                mask
            )
            layer.self_attn.indexer.requires_grad_(False)
        layer.mlp.gate.register_forward_hook(
            lambda module, args, output: captured.append(output[0])
        )
    ids = torch.tensor([[5, 6, 2, 7, 8, 9, 10], [9, 2, 2, 5, 6, 0, 0]])
    valid = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool)
    visible = torch.ones(7, 7, dtype=torch.bool).tril()[None, None] & valid[:, None, None, :]
    mask = torch.zeros(2, 1, 7, 7).masked_fill(~visible, torch.finfo(torch.float32).min)
    labels = ids.masked_fill(~valid, -100)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = local(ids, valid, labels=ids)
        hidden = reference(
            ids,
            attention_mask={"qwen_sparse_attention": mask, "linear_attention": valid},
            use_cache=False,
        ).last_hidden_state
        logits = head(hidden)
        lm_loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, cfg.vocab_size), labels[:, 1:].reshape(-1)
        )
        aux = official_balance_loss()(
            tuple(captured), cfg.num_experts, cfg.num_experts_per_tok, valid
        )
        expected = lm_loss + local.router_aux_loss_coef * aux
    torch.testing.assert_close(actual.logits, logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual.aux_loss, aux, atol=0, rtol=0)
    actual.loss.backward()
    expected.backward()
    reference_parameters = {"model." + name: p for name, p in reference.named_parameters()}
    reference_parameters["lm_head.weight"] = head.weight
    for name, p in local.named_parameters():
        q = reference_parameters[name]
        if not p.requires_grad:
            assert ".self_attn.indexer." in name and p.grad is None and q.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
    assert len(actual.router_logits) == cfg.num_hidden_layers
    assert all(not layer.mlp.gate._forward_hooks for layer in local.model.layers)


def test_dense_lm_real_updates_and_exact_optimizer_resume():
    # AdamW here is only an independent optimizer/gradient plumbing test,
    # not the report's Muon recipe or an advertised training experiment.
    torch.manual_seed(127)
    cfg = tiny_config(gradient_checkpointing=True)
    model = MiniQwen4ForCausalLM(cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
    initial = model.lm_head.weight.detach().clone()

    def update(m, opt):
        opt.zero_grad(set_to_none=True)
        loss = m(ids, labels=ids).loss
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
        return loss.item()

    update(model, optimizer)
    assert not torch.equal(initial, model.lm_head.weight)
    saved_model = deepcopy(model.state_dict())
    saved_optimizer = deepcopy(optimizer.state_dict())
    expected_loss = update(model, optimizer)
    resumed = MiniQwen4ForCausalLM(cfg)
    resumed.load_state_dict(saved_model, strict=True)
    resumed_optimizer = torch.optim.AdamW(
        (p for p in resumed.parameters() if p.requires_grad), lr=0.001
    )
    resumed_optimizer.load_state_dict(saved_optimizer)
    assert update(resumed, resumed_optimizer) == expected_loss
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
