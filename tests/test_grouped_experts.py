"""Loop oracle: full loss, input/parameter gradients, route weighting and dtype order."""

import copy

import pytest
import torch
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_deepseek, tiny_kimi

from minifrontier.models.grouped_experts import configure
from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM


@pytest.mark.parametrize("family", ["kimi", "deepseek", "qwen"])
@pytest.mark.parametrize("qat", [False, True])
def test_batched_experts_match_source_loop_and_all_parameter_gradients(family, qat):
    if family == "qwen" and qat:
        pytest.skip("Qwen has no declared native MX QAT recipe")
    torch.manual_seed(718)
    if family == "kimi":
        model = MiniKimiK3ForCausalLM(
            tiny_kimi(
                mtp_enabled=True,
                moe_intermediate_size=32,
                routed_expert_hidden_size=32,
                qat_scheme="mxfp4-mxfp8-v1" if qat else "bf16",
            )
        )
    elif family == "deepseek":
        model = MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(
                mtp_enabled=True,
                moe_inter_dim=32,
                index_head_dim=32,
                qat_scheme="mxfp4-indexer-v1" if qat else "bf16",
            )
        )
    else:
        model = MiniQwen4ForCausalLM(tiny_config(vocab_size=64, mtp_enabled=True))
    alternative = copy.deepcopy(model)
    identities = {name: id(p) for name, p in alternative.named_parameters()}
    configure(alternative, "batched")
    assert identities == {name: id(p) for name, p in alternative.named_parameters()}
    assert model.state_dict().keys() == alternative.state_dict().keys()
    ids = torch.randint(10, 60, (2, 17))
    expected, actual = model(ids, labels=ids), alternative(ids, labels=ids)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-6, atol=1e-6)
    expected.loss.backward()
    actual.loss.backward()
    for (name, p), (other_name, q) in zip(
        model.named_parameters(), alternative.named_parameters(), strict=True
    ):
        assert name == other_name
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(q.grad, p.grad, rtol=2e-3, atol=2e-6, msg=name)
