from copy import deepcopy

import pytest
import torch
from reference_muon import ReferenceMuon
from test_miniqwen4 import tiny_config
from torch import nn

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer, source_muon_blocks


def test_source_semantic_groups_include_ple_and_split_fused_gate():
    model = MiniQwen4ForCausalLM(tiny_config())
    specs = source_muon_blocks(model)
    for layer in model.model.layers:
        experts = layer.mlp.experts
        assert len(specs[id(experts.gate_up_proj)]) == 2 * model.config.num_experts
        if layer.ple is not None:
            assert id(layer.ple.key_proj.weight) in specs
            assert id(layer.ple.value_proj.weight) in specs
        if hasattr(layer, "self_attn"):
            blocks = specs[id(layer.self_attn.q_proj.weight)]
            assert [block[-1] for block in blocks] == [
                "muon",
                "adamw",
            ] * model.config.num_attention_heads
        assert id(layer.mlp.gate.weight) not in specs
    optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0001)
    tables = [g for g in optimizer.param_groups if "ngram_embedding.weight" in g["name"]]
    assert tables and all(g["weight_decay"] == 0 for g in tables)
    assert not any(".self_attn.indexer." in g["name"] for g in optimizer.param_groups)


def test_fused_optimizer_equals_independent_muon_and_adam_blocks():
    torch.manual_seed(131)
    model = MiniQwen4ForCausalLM(tiny_config())
    optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002, weight_decay=0.1)
    references, muon, adam = [], [], []
    for group in optimizer.param_groups:
        source = group["params"][0]
        descriptors = group["blocks"] or [(-1, 0, 0, "adamw")]
        for expert, start, end, algorithm in descriptors:
            index = (expert, slice(start, end)) if source.ndim == 3 else (slice(start, end),)
            if group["blocks"] is None:
                index = (...,)
            reference = nn.Parameter(source[index].detach().clone())
            references.append((source, index, reference))
            if algorithm == "muon":
                muon.append(reference)
            else:
                adam.append(dict(params=[reference], weight_decay=group["weight_decay"]))
    reference_muon = ReferenceMuon(
        muon,
        lr=0.001,
        weight_decay=0.1,
    )
    reference_adam = torch.optim.AdamW(adam, lr=0.0002, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    for _ in range(3):
        for group in optimizer.param_groups:
            p = group["params"][0]
            p.grad = torch.randn_like(p)
        for source, index, reference in references:
            reference.grad = source.grad[index].clone()
        optimizer.step()
        reference_muon.step()
        reference_adam.step()
        for source, index, reference in references:
            torch.testing.assert_close(source[index], reference, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_source_optimizer_resume_preserves_fp32_states(dtype):
    torch.manual_seed(137)
    cfg = tiny_config()
    model = MiniQwen4ForCausalLM(cfg).to(dtype)
    optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
    for group in optimizer.param_groups:
        group["params"][0].grad = torch.randn_like(group["params"][0])
    optimizer.step()
    saved_model = deepcopy(model.state_dict())
    saved_optimizer = deepcopy(optimizer.state_dict())
    resumed = MiniQwen4ForCausalLM(cfg).to(dtype)
    resumed.load_state_dict(saved_model, strict=True)
    resumed_optimizer = MiniQwen4Optimizer(resumed, lr=0.001, adam_lr=0.0002)
    resumed_optimizer.load_state_dict(saved_optimizer)
    for a, b in zip(optimizer.param_groups, resumed_optimizer.param_groups, strict=True):
        p, q = a["params"][0], b["params"][0]
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        for name, state in optimizer.state[p].items():
            if isinstance(state, torch.Tensor):
                assert resumed_optimizer.state[q][name].dtype == torch.float32
                torch.testing.assert_close(state, resumed_optimizer.state[q][name], atol=0, rtol=0)
    optimizer.step()
    resumed_optimizer.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)


def test_resume_rejects_changed_semantic_blocks():
    model = MiniQwen4ForCausalLM(tiny_config())
    optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
    state = deepcopy(optimizer.state_dict())
    next(g for g in state["param_groups"] if g["blocks"])["blocks"] = ()
    with pytest.raises(ValueError, match="semantic"):
        optimizer.load_state_dict(state)
