"""New source stack against the original upstream TextModel, not old MiniFrontier."""

import ast
import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minifrontier.models.miniqwen4 import MiniQwen4Config, MiniQwen4TextModel

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**overrides):
    return replace(
        MiniQwen4Config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            hc_lowrank=4,
            ple_embed_dim=16,
            ngram_vocab_size_base=16,
            indexer_n_heads=2,
            indexer_head_dim=4,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            gradient_checkpointing=False,
        ),
        **overrides,
    )


def upstream_namespace():
    path = ROOT / "third_party/upstream/qwen4_exp-4177486/modeling_qwen4_exp.py"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "2e36ee6a1bc4f43fa0434ae197c2de9eba27a71aec84ee1ff111dc3a812e8283"
    )
    tree = ast.parse(source)
    names = {"Qwen4ExpTextModel"}
    for filename in ("upstream_ple.py", "upstream_core.py", "upstream_decoder.py"):
        port = ast.parse((ROOT / "minifrontier/models/miniqwen4" / filename).read_text())
        original = {getattr(n, "name", None): n for n in tree.body}
        for n in port.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in original:
                assert ast.dump(n) == ast.dump(original[n.name]), n.name
                names.add(n.name)
    names.update({"_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1"})
    nodes = ast.parse("from __future__ import annotations").body
    for n in tree.body:
        name = getattr(n, "name", None)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        if name in names:
            nodes.append(n)

    class HFBase(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass  # Every parameter is subsequently loaded from the tested stack.

    def direct(target):
        return target

    def factory(*args, **kwargs):
        return direct

    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        ACT2FN={"silu": F.silu, "sigmoid": torch.sigmoid},
        use_kernel_forward_from_hub=factory,
        use_kernel_func_from_hub_with_fallback=factory,
        use_kernelized_func=factory,
        force_accelerate_hooks=factory,
        deprecate_kwarg=factory,
        use_experts_implementation=direct,
        is_torchdynamo_exporting=lambda: False,
        maybe_autocast=torch.autocast,
        GradientCheckpointingLayer=nn.Module,
        ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name, fallback: fallback),
        auto_docstring=direct,
        merge_with_config_defaults=direct,
        capture_outputs=direct,
        Qwen4ExpPreTrainedModel=HFBase,
        OutputRecorder=lambda *a, **k: None,
        Qwen4ExpModelOutputWithPast=SimpleNamespace,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_whole_text_stack_matches_original_forward_and_gradients(autocast, checkpointing, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(107)
    cfg = tiny_config(gradient_checkpointing=checkpointing)
    local = MiniQwen4TextModel(cfg).to(device).train()
    oracle = upstream_namespace()["Qwen4ExpTextModel"](cfg.upstream_config()).to(device).train()
    oracle.load_state_dict(local.state_dict(), strict=True)
    ids = torch.tensor([[5, 6, 2, 7, 8, 9, 10], [9, 2, 2, 5, 6, 0, 0]], device=device)
    valid = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool, device=device)
    visible = (
        torch.ones(7, 7, dtype=torch.bool, device=device).tril()[None, None]
        & valid[:, None, None, :]
    )
    mask = torch.zeros(2, 1, 7, 7, device=device).masked_fill(
        ~visible, torch.finfo(torch.float32).min
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actual = local(ids, valid)
        expected = oracle(
            ids,
            attention_mask={"qwen_sparse_attention": mask, "linear_attention": valid},
            use_cache=False,
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    reference = dict(oracle.named_parameters())
    disconnected = []
    for name, parameter in local.named_parameters():
        other = reference[name]
        if parameter.grad is None:
            assert other.grad is None
            disconnected.append(name)
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0, msg=name)
    # Published inference top-k is discrete: it does NOT supply indexer training.
    assert disconnected and all(".self_attn.indexer." in n for n in disconnected)
    assert local.training_ready is True


def test_new_stack_does_not_import_retired_model():
    for filename in ("modeling.py", "upstream_decoder.py"):
        tree = ast.parse((ROOT / "minifrontier/models/miniqwen4" / filename).read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any("qwen_flash_next" in name for name in imports)


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexer_kv_heads": 2},
        {"ple_layer_ids": (4,)},
        {"ngram_size": 2},
        {"indexer_budget": 3},
        {"output_gate_type": "relu"},
        {"seed": -1},
    ],
)
def test_invalid_new_config_rejected(overrides):
    with pytest.raises(ValueError):
        MiniQwen4TextModel(tiny_config(**overrides))
