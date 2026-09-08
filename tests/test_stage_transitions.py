"""Stage changes cannot alter capacity; V1 keeps all inherited text tensors frozen."""

from dataclasses import asdict

import pytest
import torch
from test_new_backbones import tiny_deepseek, tiny_kimi
from test_strategy_processing import color_grid, vision_model

from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.migration import configure_visual_warmup
from minifrontier.models.minideepseekv4.processing import process_image
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.training.minideepseekv4_optim import MiniDeepSeekV4Optimizer
from minifrontier.training.transitions import load_previous, set_visual_rates


@pytest.mark.parametrize("name", ["kimi", "deepseek"])
def test_explicit_qat_transition_preserves_master_state_and_resume_is_exact(name):
    if name == "kimi":
        config = tiny_kimi(hidden_size=32, routed_expert_hidden_size=32, moe_intermediate_size=32)
        kind, scheme = MiniKimiK3ForCausalLM, "mxfp4-mxfp8-v1"
    else:
        config = tiny_deepseek(dim=32, moe_inter_dim=32, index_head_dim=32)
        kind, scheme = MiniDeepSeekV4ForCausalLM, "mxfp4-indexer-v1"
    model = kind(config)
    saved = dict(config=asdict(config), model=model.state_dict())
    config.qat_scheme = scheme
    qat = kind(config)
    with pytest.raises(ValueError, match="capacity"):
        load_previous(qat, saved)
    with pytest.raises(ValueError, match="entering SFT"):
        load_previous(qat, saved, transition="qat")
    load_previous(qat, saved, transition="qat", stage="sft")
    for key, value in qat.state_dict().items():
        torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="capacity"):
        load_previous(qat, saved, transition="qat", stage="sft", resume=True)


def test_native_visual_warmup_updates_random_vision_without_text_drift():
    text = MiniDeepSeekV4ForCausalLM(tiny_deepseek(max_seq_len=128))
    visual = vision_model()
    saved = dict(model_name="minideepseekv4", config=asdict(text.config), model=text.state_dict())
    load_previous(visual, saved, transition="text-to-vision")
    configure_visual_warmup(visual)
    optimizer = MiniDeepSeekV4Optimizer(visual, lr=1e-4, eps=1e-8)
    set_visual_rates(optimizer, visual, vision_lr=1e-4, projector_lr=3e-4)
    assert {group["visual_base_lr"] for group in optimizer.param_groups} == {1e-4, 3e-4}
    before = {key: value.clone() for key, value in visual.state_dict().items()}
    media = process_image(color_grid(), start=2, max_features=4, min_pixels=1764)
    ids = torch.tensor([[1, 20, *(visual.config.vocab_size + media["types"]).tolist(), 21, 22, 2]])
    labels = ids.clone()
    labels[labels >= visual.config.vocab_size] = -100
    visual(ids, labels=labels, media=[dict(media, start=2, batch_index=0)]).loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in visual.vision.parameters())
    optimizer.step()
    updated = [
        key for key, value in visual.state_dict().items() if not torch.equal(value, before[key])
    ]
    assert updated and all(key.startswith(("vision.", "image_")) for key in updated)
