"""CPU checks of causal sharing, mixing, hash barriers and differentiable training."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minifrontier.models.deepseek_v41_layers import (
    Compressor,
    CSA2Attention,
    CSA2State,
    SinglePassHC,
)
from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minideepseekv41 import (
    MiniDeepSeekV41Cache,
    MiniDeepSeekV41Config,
    MiniDeepSeekV41ForCausalLM,
)


def tiny(**changes):
    values = dict(
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
        gradient_checkpointing=False,
        expert_execution="loop",
        attention_chunk_size=4,
    )
    values.update(changes)
    return MiniDeepSeekV41Config(**values)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_source_modes_causal_prefix_and_padding():
    torch.manual_seed(12)
    model = MiniDeepSeekV41ForCausalLM(tiny()).eval()
    assert [layer.attn.mode for layer in model.layers] == [
        "swa",
        "full",
        "reuse",
        "full",
        "reuse",
        "reindex",
    ]
    ids = torch.randint(3, 64, (2, 11))
    with torch.no_grad():
        complete = model(ids).logits
        prefix = model(ids[:, :7]).logits
        mask = torch.ones_like(ids)
        mask[1, 7:] = 0
        padded = model(ids, attention_mask=mask).logits
    torch.testing.assert_close(complete[:, :7], prefix, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(padded[1, :7], prefix[1], atol=2e-6, rtol=2e-5)


def test_shared_kv_reindex_and_reuse_keep_owner_and_gradients():
    torch.manual_seed(13)
    config = tiny()
    full, reuse, reindex = [CSA2Attention(config, i) for i in (3, 4, 5)]
    x = torch.randn(2, 9, 32, requires_grad=True)
    encoder = torch.randn(2, 9, 32, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    _, state, _ = full(x, CSA2State(), mask, encoder)
    kv = state.kv
    _, reused, _ = reuse(x, state, mask, encoder)
    out, indexed, loss = reindex(x, reused, mask, encoder)
    assert indexed.kv is kv and reused.indices is state.indices
    assert indexed.index_key is state.index_key and indexed.owner == 3
    assert reindex.compressor is None and reuse.indexer is None
    (out.square().mean() + loss).backward()
    assert encoder.grad.abs().sum() > 0
    assert full.compressor.wkv.weight.grad.abs().sum() > 0
    assert full.indexer.wk.weight.grad.abs().sum() > 0
    assert reindex.indexer.wq_b.weight.grad.abs().sum() > 0


def test_hierarchical_reindex_uses_causal_candidate_blocks():
    config = tiny(hierarchical_indexing=True)
    full, reindex = CSA2Attention(config, 3), CSA2Attention(config, 5)
    x, mask = torch.randn(1, 13, 32), torch.ones(1, 13, dtype=torch.bool)
    _, state, _ = full(x, CSA2State(), mask, x)
    assert state.candidates.sum(-1).max() <= 2
    assert not torch.triu(state.candidates[0], diagonal=1).any()
    _, next_state, _ = reindex(x, state, mask, x)
    assert state.candidates.gather(-1, next_state.indices.clamp_min(0))[
        next_state.indices >= 0
    ].all()


def test_single_pass_uses_previous_coefficients_and_source_destination_orientation():
    torch.manual_seed(3)
    mixer = SinglePassHC(tiny())
    stream, output = torch.randn(1, 2, 2, 32), torch.randn(1, 2, 32)
    incoming = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    pre, post, comb = mixer.coefficients(stream)
    actual = mixer.combine(output, stream, post, comb)
    expected = post[..., None] * output[..., None, :] + (
        comb[..., None] * stream.unsqueeze(-2)
    ).sum(2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        mixer.mix(stream, incoming), torch.stack((stream[0, 0, 0], stream[0, 1, 1]))[None]
    )
    assert not torch.allclose(mixer.mix(stream, incoming), mixer.mix(stream, pre))


def test_engram_image_barrier_and_compressed_identity_checkpoint():
    model = MiniDeepSeekV41ForCausalLM(tiny())
    engram = model.layers[1].engram
    mapping = list(range(64))
    mapping[12] = mapping[11]
    engram.bind_token_map(mapping)
    ids = torch.tensor([[5, 6, 0, 11, 9]])
    mask = torch.tensor([[True, True, False, True, True]])
    other = ids.clone()
    other[0, :2], other[0, 3] = torch.tensor([30, 31]), 12
    torch.testing.assert_close(engram.hashes(ids, mask)[:, 3:], engram.hashes(other, mask)[:, 3:])
    clone = MiniDeepSeekV41ForCausalLM(tiny())
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.layers[1].engram.hashes(ids, mask), engram.hashes(ids, mask))
    stream = torch.randn(1, 5, 2, 32)
    torch.testing.assert_close(engram(stream, ids, mask)[:, 2], stream[:, 2])


def test_checkpoint_recomputation_matches_gradients_and_kl_trains_indexer():
    torch.manual_seed(19)
    plain = MiniDeepSeekV41ForCausalLM(tiny())
    recomputed = MiniDeepSeekV41ForCausalLM(tiny(gradient_checkpointing=True))
    recomputed.load_state_dict(plain.state_dict())
    ids = torch.randint(3, 64, (2, 13))
    a, b = plain(ids, labels=ids), recomputed(ids, labels=ids)
    assert a.indexer_loss > 0 and a.mtp_loss is None
    a.loss.backward()
    b.loss.backward()
    torch.testing.assert_close(a.loss, b.loss)
    for (name, p), (_, q) in zip(
        plain.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=2e-4, msg=name)
    assert plain.layers[1].attn.indexer.wq_b.weight.grad.abs().sum() > 0
    assert plain.layers[1].engram.embed.weight.grad.abs().sum() > 0


def test_batched_experts_matches_loop_training():
    torch.manual_seed(21)
    loop = MiniDeepSeekV41ForCausalLM(tiny())
    batched = MiniDeepSeekV41ForCausalLM(tiny(expert_execution="batched"))
    batched.load_state_dict(loop.state_dict())
    ids = torch.randint(3, 64, (2, 9))
    a, b = loop(ids, labels=ids), batched(ids, labels=ids)
    torch.testing.assert_close(a.loss, b.loss)
    b.loss.backward()
    assert batched.layers[0].ffn.experts[0].w1.weight.grad is not None


def test_native_visual_gradient_and_image_ce_mask():
    vision = DeepSeekVisionConfig(
        depth=1, hidden_size=16, num_heads=2, intermediate_size=24, patch_size=2, output_size=32
    )
    model = MiniDeepSeekV41ForCausalLM(tiny(vision_config=vision))
    ids = torch.tensor([[8, 64, 66, 68, 10, 11]])
    labels = ids.clone()
    labels[:, 1:4] = -100
    media = [
        dict(
            batch_index=0,
            start=1,
            types=torch.tensor([0, 2, 4]),
            patches=torch.randn(9, 3, 2, 2),
            n_vit_h=3,
            n_vit_w=3,
        )
    ]
    out = model(ids, labels=labels, media=media)
    out.loss.backward()
    assert model.vision.vit.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.vision.aligner.w1.weight.grad.abs().sum() > 0
    assert model.layers[0].ffn.gate.bias_vl is not None
    with pytest.raises(ValueError, match="labels outside vocabulary"):
        model(ids, labels=ids, media=media)


def test_exact_reference_cache_matches_full_sequence_and_rejects_mutation():
    model = MiniDeepSeekV41ForCausalLM(tiny()).eval()
    ids = torch.randint(3, 64, (1, 9))
    cache = MiniDeepSeekV41Cache()
    with torch.no_grad():
        complete = model(ids).logits
        prefill = model(ids[:, :7], cache=cache).logits
        end = model(ids[:, 7:], cache=cache).logits
        torch.testing.assert_close(torch.cat((prefill, end), 1), complete, atol=2e-6, rtol=2e-5)
        model.head.weight.add_(0.1)
        with pytest.raises(ValueError, match="different weights"):
            model(ids[:, -1:], cache=cache)


def test_optimizer_metadata_and_backbone_stage_contract():
    model = MiniDeepSeekV41ForCausalLM(tiny())
    meta = model.optimizer_metadata()
    assert meta["embed.weight"]["kind"] == "sinkhorn"
    assert meta["layers.1.engram.embed.weight"]["lr_scale"] == 5
    assert meta["layers.1.attn.wq_b.weight"]["heads"] == 2
    assert meta["layers.1.engram.q_weight"]["kind"] == "adamw"
    with pytest.raises(ValueError, match="dense warmup"):
        model.configure_training_phase("dense_pretrain")
    with pytest.raises(ValueError, match="omits MTP"):
        MiniDeepSeekV41ForCausalLM(tiny(mtp_enabled=True))


def official_method(class_name, method_name):
    """Execute the pinned released method, without importing inference CUDA kernels."""
    path = (
        Path(__file__).parents[1] / "third_party/upstream/deepseek-v4.1-dba1be0/inference/model.py"
    )
    source = ast.parse(path.read_text())
    cls = next(
        item for item in source.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    method = next(
        item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == method_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_single_pass_post_matches_pinned_official_method():
    post = official_method("Block", "hc_post")
    stream, output = torch.randn(2, 5, 2, 32), torch.randn(2, 5, 32)
    coefficients = torch.randn(2, 5, 2, 2)
    gate = torch.randn(2, 5, 2)
    torch.testing.assert_close(
        SinglePassHC.combine(output, stream, gate, coefficients),
        post(None, output, stream, gate, coefficients),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = SinglePassHC.combine(output.to(torch.bfloat16), stream, gate, coefficients)
        expected = post(None, output.to(torch.bfloat16), stream, gate, coefficients)
    torch.testing.assert_close(mixed, expected, atol=0, rtol=0)


def test_compressor_matches_pinned_official_forward_and_backward():
    forward = official_method("Compressor", "forward")
    compressor = Compressor(tiny(), ratio=2)
    source = SimpleNamespace(
        compress_ratio=2,
        wkv=compressor.wkv,
        wgate=compressor.wgate,
        norm=compressor.norm,
        kv_state=torch.zeros(2, 2, 16),
        score_state=torch.zeros(2, 2, 16),
    )
    x = torch.randn(2, 9, 32, requires_grad=True)
    actual, expected = compressor(x), forward(source, x, 0)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(a, b)


def test_engram_matches_pinned_official_forward_and_backward():
    forward = official_method("Engram", "forward")
    model = MiniDeepSeekV41ForCausalLM(tiny())
    engram = model.layers[1].engram
    source = SimpleNamespace(
        dim=32,
        hc_mult=2,
        eps=model.config.norm_eps,
        clamp_value=1e-6,
        wkv=engram.wkv,
        embed=engram.embed,
        q_weight=engram.q_weight,
        k_weight=engram.k_weight,
    )
    ids = torch.randint(3, 64, (2, 9))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 4:6] = False
    stream = torch.randn(2, 9, 2, 32, requires_grad=True)
    actual = engram(stream, ids, mask)
    expected = forward(source, stream, engram.hashes(ids, mask), mask)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), stream, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), stream)[0]
    torch.testing.assert_close(a, b)


def test_bfloat16_training_is_finite():
    model = MiniDeepSeekV41ForCausalLM(
        tiny(gradient_checkpointing=True, expert_execution="batched")
    )
    ids = torch.randint(3, 64, (2, 11))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(ids, labels=ids, return_logits=False)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
