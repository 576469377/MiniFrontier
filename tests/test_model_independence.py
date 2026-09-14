"""Model packages own computation; training controls still recognize local routers."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

FAMILIES = (
    "minikimik3",
    "miniqwen4",
    "minideepseekv4",
    "minideepseekv41",
    "minifrontier1",
    "minifrontier11",
)
ROOT = Path(__file__).parents[1]


def v41_tiny_values():
    return dict(
        vocab_size=64,
        dim=32,
        n_layers=6,
        n_encoder_layers=3,
        n_heads=2,
        head_dim=16,
        rope_head_dim=4,
        q_lora_rank=8,
        o_groups=1,
        o_lora_rank=8,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        window_size=4,
        compress_ratios=(0, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 5),
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        index_candidate_source_layer=3,
        index_candidate_topk_blocks=1,
        index_candidate_block_size=2,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        max_seq_len=64,
        engram_layer_ids=(1,),
        engram_vocab_size=13,
        engram_n_heads=2,
        engram_head_dim=4,
        gradient_checkpointing=True,
        expert_execution="batched",
        attention_chunk_size=4,
    )


# Fresh interpreters catch indirect imports hidden by pytest's shared module cache.
IMPORT_GUARD = r"""
import importlib
import importlib.abc
import json
import sys
import types
import torch

torch.set_num_threads(1)
family = sys.argv[1]
families = set(json.loads(sys.argv[2]))
forbidden = families - {family}
def other_model(name):
    bits = name.split('.')
    return len(bits) >= 3 and bits[:2] == ['minifrontier', 'models'] and bits[2] in forbidden
class RejectOtherModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if other_model(fullname):
            raise AssertionError(f'{family} computation imported sibling {fullname}')
sys.meta_path.insert(0, RejectOtherModels())
package = importlib.import_module('minifrontier.models.' + family)
from minifrontier.models.factory import model_classes
config_type, model_type = model_classes(family)
assert config_type.__module__.startswith('minifrontier.models.' + family + '.')
assert model_type.__module__.startswith('minifrontier.models.' + family + '.')
if family in {'minifrontier1', 'minifrontier11', 'minideepseekv41'}:
    config = config_type(**json.loads(sys.argv[3])) if family == 'minideepseekv41' else config_type.tiny(vocab_size=64)
    model = model_type(config)
    grouped_calls = []
    if family == 'minideepseekv41':
        adapter = importlib.import_module('minifrontier.models.' + family + '.batched_experts')
        assert any(isinstance(m, adapter.BatchedDeepSeekMoE) for m in model.modules())
    else:
        # The normal CPU forward uses scalar experts. Exercise the very same
        # production grouped implementation on CPU, without pretending to run CUDA.
        def grouped_reference(module, latent, selected, weights):
            result = module.grouped(latent, selected, weights)
            assert result is not None, 'tiny dispatch unexpectedly used fallback'
            grouped_calls.append(True)
            return result
        for module in model.modules():
            if type(module).__name__ == 'LatentMoE':
                module.reference = types.MethodType(grouped_reference, module)
    ids = torch.randint(24, 64, (2, 9))
    result = model(ids, labels=ids, return_logits=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if family != 'minideepseekv41':
        assert grouped_calls
assert not any(other_model(name) for name in sys.modules)
print(f'{family}: independent')
"""


@pytest.mark.parametrize("family", FAMILIES)
def test_model_imports_and_grouped_training_are_independent(family):
    import json

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            IMPORT_GUARD,
            family,
            json.dumps(FAMILIES),
            json.dumps(v41_tiny_values()),
        ],
        cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("family", FAMILIES)
def test_model_package_has_no_sibling_imports(family):
    root = ROOT / "minifrontier/models" / family
    assert root.is_dir()
    siblings = set(FAMILIES) - {family}
    for path in root.rglob("*.py"):
        package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module or ""
                )
                names = [base, *(base + "." + alias.name for alias in node.names)]
            for name in names:
                assert not set(name.split(".")) & siblings, (path, name)


@pytest.fixture
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_v41_local_router_balance_counts_update_and_restore(tmp_path, one_cpu_thread):
    from minifrontier.models.minideepseekv41 import (
        MiniDeepSeekV41Config,
        MiniDeepSeekV41ForCausalLM,
    )
    from minifrontier.training.runtime import RouterBalance

    config = MiniDeepSeekV41Config(**v41_tiny_values())
    model = MiniDeepSeekV41ForCausalLM(config)
    balance = RouterBalance(model)
    assert len(balance.gates) == config.n_layers
    with torch.no_grad():
        for _, bias, _ in balance.gates:
            bias.copy_(torch.arange(bias.numel()) * 10)
    ids = torch.randint(24, 64, (2, 9))
    valid = torch.ones_like(ids, dtype=torch.bool)
    with balance.capture(valid):
        output = model(ids, labels=ids)
    counts = [count.clone() for count in balance.counts]
    assert all(count.sum() == valid.sum() * config.n_activated_experts for count in counts)
    output.loss.backward()
    for before, after in zip(counts, balance.counts, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    path = tmp_path / "v41-router.pt"
    torch.save(dict(model=model.state_dict(), balance=balance.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = MiniDeepSeekV41ForCausalLM(config)
    restored.load_state_dict(saved["model"], strict=True)
    restored_balance = RouterBalance(restored)
    restored_balance.load_state_dict(saved["balance"])
    old_biases = [bias.clone() for _, bias, _ in balance.gates]
    balance.update(0.001)
    restored_balance.update(0.001)
    assert any(
        not torch.equal(old, now)
        for old, (_, now, _) in zip(old_biases, balance.gates, strict=True)
    )
    for (_, a, _), (_, b, _) in zip(balance.gates, restored_balance.gates, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(count.sum() == 0 for count in restored_balance.counts)


def test_mf11_local_quantile_balance_counts_update_and_restore(tmp_path, one_cpu_thread):
    from minifrontier.models.minifrontier11 import MiniFrontier11Config, MiniFrontier11ForCausalLM
    from minifrontier.training.minifrontier1_optim import QuantileBalance

    config = MiniFrontier11Config.tiny(vocab_size=64)
    model = MiniFrontier11ForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ == "Router":
                module.correction_bias.copy_(torch.linspace(-0.2, 0.2, config.num_experts))
    balance = QuantileBalance(model, bins=32, warmup_updates=1)
    assert len(balance.routers) == config.num_hidden_layers
    ids = torch.randint(24, 64, (2, 9))
    valid = torch.ones_like(ids, dtype=torch.bool)
    with balance.capture(valid):
        output = model(ids, labels=ids, return_logits=False)
    assert all(
        hist.counts.sum() == valid.sum() * config.num_experts for _, _, hist in balance.routers
    )
    assert all(
        hist.load.sum() == valid.sum() * config.num_experts_per_token
        for _, _, hist in balance.routers
    )
    output.loss.backward()
    path = tmp_path / "mf11-router.pt"
    torch.save(dict(model=model.state_dict(), balance=balance.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = MiniFrontier11ForCausalLM(config)
    restored.load_state_dict(saved["model"], strict=True)
    restored_balance = QuantileBalance(restored, bins=32, warmup_updates=1)
    restored_balance.load_state_dict(saved["balance"])
    records, restored_records = balance.update(), restored_balance.update()
    assert records and records == restored_records
    assert all(record["tokens"] == int(valid.sum()) for record in records.values())
    assert restored_balance.updates == 1
    for (name, a, hist), (other, b, other_hist) in zip(
        balance.routers, restored_balance.routers, strict=True
    ):
        assert name == other
        torch.testing.assert_close(a.correction_bias, b.correction_bias, atol=0, rtol=0)
        torch.testing.assert_close(balance.ema[name], restored_balance.ema[name], atol=0, rtol=0)
        assert hist.counts.sum() == other_hist.counts.sum() == 0
