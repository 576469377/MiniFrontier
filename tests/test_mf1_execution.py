"""Independent scalar oracles and batching invariants for MF1 execution changes."""

import copy
from dataclasses import replace

import pytest
import torch

from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.csa import Compressor
from minifrontier.models.minifrontier1.indexer import block_registry, mrope, rope, select_blocks
from minifrontier.models.minifrontier1.kda import KDA
from minifrontier.models.minifrontier1.moe import LatentMoE
from minifrontier.models.minifrontier1.processing import token_metadata
from minifrontier.models.minifrontier1.qsa_mla import QSAMLA
from minifrontier.training.minifrontier1_curriculum import (
    collate_records,
    microbatches,
    pack_records,
)


def assert_gradients(left, right, *, atol=2e-5, rtol=2e-4):
    for (name, a), (other, b) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert name == other
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=rtol, msg=name)


def scalar_pool(module, x, blocks, offset):
    kv, gates = module.kv(x.float()), module.gate(x.float())
    values = []
    for i, block in enumerate(blocks):
        ids = [t - offset for t in block.member_indices]
        v, g = (
            kv[ids, module.dim :],
            gates[ids, module.dim :] + module.ape[: len(ids), module.dim :],
        )
        if i:
            old = blocks[i - 1]
            if (old.segment, old.modality, old.media_id) == (
                block.segment,
                block.modality,
                block.media_id,
            ):
                prev = [t - offset for t in old.member_indices]
                v = torch.cat((kv[prev, : module.dim], v))
                g = torch.cat(
                    (gates[prev, : module.dim] + module.ape[: len(prev), : module.dim], g)
                )
        values.append((v * g.softmax(0)).sum(0))
    return module.norm(torch.stack(values))


def test_compressor_vectorization_matches_overlap_oracle_and_all_gradients():
    torch.manual_seed(731)
    a = Compressor(12, 8, 1e-5)
    b = copy.deepcopy(a)
    seg = torch.tensor([0] * 13 + [1] * 7 + [-1] * 2)
    mod = torch.tensor([0] * 5 + [1] * 3 + [0] * 14)
    blocks = block_registry(seg, mod, torch.where(mod.bool(), 2, -1), offset=17)
    x = torch.randn(22, 12, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = a(x, blocks, 17), scalar_pool(b, y, blocks, 17)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
    assert_gradients(a, b)


def scalar_qsa(module, x, metadata):
    """Per-query gather oracle, deliberately without the chunked execution helpers."""
    c = module.config
    rows = []
    for batch, z in enumerate(x):
        seg, mod, media = (metadata[k][batch] for k in ("segment_ids", "modality", "media_ids"))
        pos = metadata["position_ids"][:, batch]
        q = module.q_up(module.q_norm(module.q_down(z))).unflatten(
            -1, (c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim)
        )
        qc, qr = q.split((c.qk_nope_head_dim, c.qk_rope_head_dim), -1)
        qr = mrope(qr, pos, c.mrope_sections, c.rope_theta)
        latent, kr = module.kv_down(z).split((c.kv_lora_rank, c.qk_rope_head_dim), -1)
        kr = mrope(kr, pos, c.mrope_sections, c.rope_theta)
        kc, v = (
            module.kv_up(module.kv_norm(latent))
            .unflatten(-1, (c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim))
            .split((c.qk_nope_head_dim, c.v_head_dim), -1)
        )
        blocks = [b for b in block_registry(seg, mod, media) if b.complete_at is not None]
        raw = module.indexer.key(z.detach())
        keys = torch.stack([raw[list(b.member_indices)].mean(0) for b in blocks])
        keys = rope(
            keys,
            metadata["linear_positions"][batch][[b.start for b in blocks]],
            c.index_rope_dim,
            c.rope_theta,
        )
        scores = module.indexer.scores(z.detach(), metadata["linear_positions"][batch], keys)
        outputs = []
        for i in range(len(z)):
            visible = torch.tensor(
                [b.segment == int(seg[i]) and b.complete_at <= i and seg[i] >= 0 for b in blocks]
            )
            chosen = select_blocks(scores[i : i + 1], visible[None], c.top_blocks)[0]
            allowed = {t for j, b in enumerate(blocks) if chosen[j] for t in b.member_indices}
            ids = [
                t
                for t in range(i + 1)
                if seg[t] == seg[i]
                and seg[i] >= 0
                and (
                    module.training_phase != "sparse_cpt"
                    or i - t < c.window_size
                    or (c.protect_media and mod[t] != 0)
                    or t in allowed
                )
            ]
            if not ids:
                outputs.append(z.new_zeros(c.num_attention_heads, c.v_head_dim))
                continue
            logits = (
                torch.einsum("hd,khd->hk", qc[i], kc[ids])
                + torch.einsum("hd,kd->hk", qr[i], kr[ids])
            ) * (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5
            outputs.append(torch.einsum("hk,khd->hd", logits.softmax(-1), v[ids]))
        rows.append(module.out(torch.stack(outputs).flatten(-2) * module.gate(z).sigmoid()))
    return torch.stack(rows)


@pytest.mark.parametrize("phase", ["dense_pretrain", "sparse_cpt"])
def test_qsa_chunked_output_and_gradients_match_per_query_oracle(phase):
    torch.manual_seed(82)
    c = replace(MiniFrontier1Config.tiny(), query_chunk_size=5, top_blocks=1)
    a = QSAMLA(c)
    a.training_phase, a.indexer_loss_enabled = phase, False
    b = copy.deepcopy(a)
    ids = torch.randint(24, 200, (2, 29))
    ids[1, -3:] = 0
    segments = torch.zeros_like(ids)
    segments[0, 17:] = 1
    meta = token_metadata(ids, c, segment_ids=segments)
    meta["modality"][:, 4:7] = 1
    meta["media_ids"][:, 4:7] = 0
    x = torch.randn(2, 29, c.hidden_size, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = a(x, meta)[0], scalar_qsa(b, y, meta)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
    assert_gradients(a, b)


def test_grouped_experts_keep_unused_gradients_none_and_match_optimizer_step():
    torch.manual_seed(52)
    c = MiniFrontier1Config.tiny()
    a, x = LatentMoE(c), torch.randn(9, c.routed_expert_hidden_size, requires_grad=True)
    b, y = copy.deepcopy(a), x.detach().clone().requires_grad_()
    ids = torch.tensor([[2, 0], [0, 2], [2, 0]] * 3)
    weights = torch.rand(9, 2)
    actual, expected = a.grouped(x, ids, weights), b.reference(y, ids, weights)
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
    assert_gradients(a, b)
    for index in (1, 3):
        assert all(p.grad is None for p in a.experts[index].parameters())
    for model in (a, b):
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1).step()
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_kda_batched_prefill_preserves_packed_resets_outputs_and_gradients():
    torch.manual_seed(61)
    a = KDA(MiniFrontier1Config.tiny())
    a.core.A_log.data.zero_()
    a.core.dt_bias.data.zero_()
    b = copy.deepcopy(a)
    x = torch.randn(3, 13, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    seg = torch.tensor([[0] * 6 + [1] * 7, [0] * 6 + [1] * 5 + [-1] * 2, [0] + [1] * 12])
    actual, states = a(x, seg)
    expected, other = b(y, seg, state=[{}, {}, {}])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert states[1] == other[1] == {}
    for i in (0, 2):
        torch.testing.assert_close(states[i]["recurrent"], other[i]["recurrent"])
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
    assert_gradients(a, b)


@pytest.mark.cuda
def test_cuda_one_token_kda_retains_projection_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    c = replace(MiniFrontier1Config.tiny(), kda_head_dim=16)
    model = KDA(c).cuda()
    model.core.A_log.data.zero_()
    model.core.dt_bias.data.zero_()
    x = torch.randn(2, 1, 32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = model(x, torch.zeros(2, 1, dtype=torch.long, device="cuda"))
    output.float().square().sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "b_proj"):
        grad = getattr(model.core, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_collation_preserves_packed_boundaries_ce_mtp_and_checkpoint_gradients():
    torch.manual_seed(62)
    c = replace(MiniFrontier1Config.tiny(), gradient_checkpointing=True)
    model = MiniFrontier1ForCausalLM(c).train()

    def item(length):
        ids = torch.randint(24, 200, (1, length))
        labels = ids.clone()
        labels[:, :2] = -100
        return dict(input_ids=ids, labels=labels, media=[], media_hashes=[], sample_id=str(length))

    inputs = [item(7), pack_records([item(5), item(9)], 20), item(11)]
    grouped = collate_records(inputs, c.pad_token_id)
    ce = sum(int(i["labels"][:, 1:].ne(-100).sum()) for i in inputs)
    before = [
        model(
            i["input_ids"],
            labels=i["labels"],
            segment_ids=i.get("segment_ids"),
            return_logits=False,
        )
        for i in inputs
    ]
    after = model(
        grouped["input_ids"],
        labels=grouped["labels"],
        segment_ids=grouped["segment_ids"],
        return_logits=False,
    )
    expected = (
        sum(
            r.lm_loss * int(i["labels"][:, 1:].ne(-100).sum())
            for r, i in zip(before, inputs, strict=True)
        )
        / ce
    )
    torch.testing.assert_close(after.lm_loss, expected, atol=2e-6, rtol=2e-5)
    assert after.mtp_tokens == sum(r.mtp_tokens for r in before)
    after.loss.backward()
    assert model.embed_tokens.weight.grad.abs().sum() > 0
    batches = list(microbatches(inputs, batch_size=8, max_padded_tokens=28, pad_token_id=0))
    assert all(b["input_ids"].numel() <= 28 for b in batches)
    assert sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in batches) == ce
