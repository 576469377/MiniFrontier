"""QSA score oracle, report KL reductions, phase gradients and boundaries."""

import ast
from copy import deepcopy

import pytest
import torch
from test_miniqwen4 import ROOT, tiny_config, upstream_namespace

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM, MiniQwen4TextModel
from minifrontier.models.miniqwen4.qsa import indexer_kl_loss, indexer_score_rows
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer


def instrumented_original_indexer(config):
    namespace = upstream_namespace()
    tree = ast.parse(
        (ROOT / "third_party/upstream/qwen4_exp-4177486/modeling_qwen4_exp.py").read_text()
    )
    node = next(n for n in tree.body if getattr(n, "name", None) == "Qwen4ExpTextQSAIndexer")
    captured = []

    class ObserveScores(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(
                isinstance(t, ast.Name) and t.id == "selected_block_indices" for t in node.targets
            ):
                observer = ast.parse(
                    "observe(batch_idx, query_idx, block_token_indices, scores, selected_block_indices)"
                ).body[0]
                return [node, observer]
            return node

    namespace["observe"] = lambda *args: captured.append(args)
    nodes = [*ast.parse("from __future__ import annotations").body, ObserveScores().visit(node)]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "original-indexer-with-observer",
            "exec",
        ),
        namespace,
    )
    return namespace["Qwen4ExpTextQSAIndexer"](config, 3), captured


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_scores_and_all_gradients_match_instrumented_original(autocast, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(163)
    cfg = tiny_config()
    model = MiniQwen4TextModel(cfg).to(device)
    indexer = model.layers[3].self_attn.indexer
    original, captured = instrumented_original_indexer(cfg.upstream_config())
    original = original.to(device)
    original.load_state_dict(indexer.state_dict(), strict=True)
    hidden = torch.randn(2, 9, cfg.hidden_size, device=device, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    valid = torch.tensor([[1] * 9, [1] * 6 + [0] * 3], device=device, dtype=torch.bool)
    visible = (
        torch.ones(9, 9, device=device, dtype=torch.bool).tril()[None, None]
        & valid[:, None, None, :]
    )
    mask = torch.zeros(2, 1, 9, 9, device=device).masked_fill(
        ~visible, torch.finfo(torch.float32).min
    )
    positions = torch.arange(9, device=device).view(1, 1, -1).expand(3, 2, -1)
    rotary = model.rotary_emb(hidden, positions)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actual = indexer_score_rows(indexer, hidden, rotary, mask)
        original(other, rotary, mask, None)
    assert len(actual) == len(captured)
    for row, (batch, query, blocks, scores, selected) in zip(actual, captured, strict=True):
        assert (row.batch, row.query) == (batch, query)
        for left, right in (
            (row.block_tokens, blocks),
            (row.scores, scores),
            (row.selected_blocks, selected),
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    sum(row.scores.square().sum() for row in actual).backward()
    sum(row[3].square().sum() for row in captured).backward()
    torch.testing.assert_close(hidden.grad, other.grad, atol=0, rtol=0)
    for p, q in zip(indexer.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("selected_only", [False, True])
def test_report_kl_reduction_masks_tail_queries_and_stops_teacher_gradient(selected_only):
    torch.manual_seed(167)
    model = MiniQwen4TextModel(tiny_config())
    indexer = model.layers[3].self_attn.indexer
    hidden = torch.randn(2, 9, 16, requires_grad=True)
    valid = torch.tensor([[1] * 9, [1] * 6 + [0] * 3], dtype=torch.bool)
    visible = torch.ones(9, 9, dtype=torch.bool).tril()[None, None] & valid[:, None, None, :]
    mask = torch.zeros(2, 1, 9, 9).masked_fill(~visible, torch.finfo(torch.float32).min)
    rotary = model.rotary_emb(hidden, torch.arange(9).view(1, 1, -1).expand(3, 2, -1))
    teacher = torch.softmax(torch.randn(2, 4, 9, 9) + mask, -1).requires_grad_()
    actual = indexer_kl_loss(
        indexer, hidden, rotary, mask, teacher, valid, selected_only=selected_only
    )
    rows = indexer_score_rows(indexer, hidden.detach(), rotary, mask)
    terms = []
    for row in rows:
        if not valid[row.batch, row.query]:
            continue
        tokens = teacher.detach()[row.batch, :, row.query].sum(0)
        tokens = tokens / tokens.sum()
        probabilities = torch.stack([tokens[block].max() for block in row.block_tokens])
        scores = row.scores
        if selected_only:
            probabilities = probabilities[row.selected_blocks]
            scores = scores[row.selected_blocks]
        probabilities = probabilities / probabilities.sum()
        positive = probabilities > 0
        terms.append(
            (
                probabilities[positive]
                * (probabilities[positive].log() - scores.log_softmax(0)[positive])
            ).sum()
        )
    expected = torch.stack(terms).sum() / valid.sum()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert hidden.grad is None and teacher.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in indexer.parameters())


@pytest.mark.parametrize("phase", ["dense_distill", "sparse_cpt"])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_phase_checkpointed_forward_and_gradients(phase, autocast, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(173)
    model = MiniQwen4ForCausalLM(tiny_config(), training_phase=phase).to(device)
    reference = MiniQwen4ForCausalLM(
        tiny_config(gradient_checkpointing=True), training_phase=phase
    ).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[5, 6, 2, 7, 8, 9, 10, 11, 12]], device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actual, expected = model(ids, labels=ids), reference(ids, labels=ids)
    assert actual.indexer_loss > 0
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    actual.loss.backward()
    expected.loss.backward()
    for (name, p), q in zip(model.named_parameters(), reference.parameters(), strict=True):
        if phase == "dense_distill" and ".self_attn.indexer." not in name:
            assert not p.requires_grad and p.grad is None and q.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
    assert all(not module._forward_hooks for module in reference.modules())


def test_phase_transitions_preserve_parameters_reject_stale_optimizer_and_update_indexer():
    torch.manual_seed(179)
    model = MiniQwen4ForCausalLM(tiny_config(gradient_checkpointing=True))
    ids = torch.tensor([[5, 6, 2, 7, 8, 9, 10, 11, 12]])
    parameters = {name: id(p) for name, p in model.named_parameters()}
    old_optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
    with pytest.raises(ValueError, match="expected"):
        model.transition_training_phase("sparse_cpt")
    model.transition_training_phase("dense_distill")
    frozen = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="rebuild"):
        old_optimizer.step()
    warmup = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.001, weight_decay=0)
    loss = model(ids, labels=ids).loss
    loss.backward()
    warmup.step()
    changed = [n for n, p in model.state_dict().items() if not torch.equal(p, frozen[n])]
    assert changed and all(".self_attn.indexer." in n for n in changed)
    model.transition_training_phase("sparse_cpt")
    assert parameters == {name: id(p) for name, p in model.named_parameters()}
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="rebuild"):
        warmup.step()
    joint = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
    model(ids, labels=ids).loss.backward()
    joint.step()
    assert not torch.equal(model.lm_head.weight, frozen["lm_head.weight"])


def test_no_complete_blocks_is_connected_zero_not_nan():
    model = MiniQwen4ForCausalLM(
        tiny_config(indexer_compress_ratio=4), training_phase="dense_distill"
    )
    ids = torch.tensor([[5, 6]])
    result = model(ids)
    assert result.loss.item() == 0
    result.loss.backward()
    for name, p in model.named_parameters():
        if ".self_attn.indexer." in name:
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0
