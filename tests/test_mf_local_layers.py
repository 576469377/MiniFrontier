"""MF-local copies preserve source outputs, gradients and checkpoint parameters."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from minifrontier.models.minideepseekv41.residual import SinglePassHC
from minifrontier.models.minifrontier1.configuration import MiniFrontier1Config
from minifrontier.models.minifrontier1.kda import KDA
from minifrontier.models.minifrontier1.residual import GatedResidual
from minifrontier.models.minifrontier11.configuration import MiniFrontier11Config
from minifrontier.models.minifrontier11.kda import KDA as KDA11
from minifrontier.models.minifrontier11.residual import SinglePassMHC
from minifrontier.models.minikimik3.upstream_layers import KimiDeltaAttention
from minifrontier.models.miniqwen4.upstream_core import Qwen4ExpTextGatedResidual


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def assert_parameters_and_gradients(local, source, rename=lambda name: name):
    expected = dict(source.named_parameters())
    actual = {rename(name): parameter for name, parameter in local.named_parameters()}
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
        assert parameter.grad is not None and expected[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, expected[name].grad, atol=0, rtol=0)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("backend", ["reference", "auto"])
@pytest.mark.parametrize(
    "config_class, kda_class", [(MiniFrontier1Config, KDA), (MiniFrontier11Config, KDA11)]
)
def test_kda_local_projection_kernel_and_gradients_match_source(
    autocast, backend, config_class, kda_class
):
    config = replace(config_class.tiny(), kda_backend=backend)
    local = kda_class(config)
    source = KimiDeltaAttention(local.core.config, 0)
    with torch.no_grad():
        source.dt_bias.zero_()
    assert list(local.core.state_dict()) == list(source.state_dict())
    local.core.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 7, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    segments = torch.zeros(x.shape[:2], dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual, _ = local(x, segments, cache_output=False)
        expected = source(reference_x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    assert_parameters_and_gradients(local.core, source)


@pytest.mark.parametrize(
    "config_class, kda_class", [(MiniFrontier1Config, KDA), (MiniFrontier11Config, KDA11)]
)
def test_kda_local_cache_layout_matches_source_prefill_and_continuation(config_class, kda_class):
    config = config_class.tiny()
    local = kda_class(config).eval()
    source = KimiDeltaAttention(local.core.config, 0).eval()
    with torch.no_grad():
        source.dt_bias.zero_()
    local.core.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 6, config.hidden_size)
    cache = SimpleNamespace(conv_states=[None], recurrent_states=[None])
    with torch.no_grad():
        prefix, conv, recurrent = local._run(x[:, :5], {})
        expected = source(x[:, :5], cache_params=cache)
        torch.testing.assert_close(prefix, expected, atol=0, rtol=0)
        torch.testing.assert_close(recurrent, cache.recurrent_states[0], atol=0, rtol=0)
        for actual_conv, expected_conv in zip(conv, cache.conv_states[0], strict=True):
            torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
        actual, conv, recurrent = local._run(x[:, 5:], dict(conv=conv, recurrent=recurrent))
        expected = source(x[:, 5:], cache_params=cache)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(recurrent, cache.recurrent_states[0], atol=0, rtol=0)
    for actual_conv, expected_conv in zip(conv, cache.conv_states[0], strict=True):
        torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_local_gr_matches_source_outputs_gradients_and_state(read_only, autocast):
    config = MiniFrontier1Config.tiny()
    local = GatedResidual(config, read_only=read_only)
    source = Qwen4ExpTextGatedResidual(config, use_combine=not read_only)
    assert list(local.primitive.state_dict()) == list(source.state_dict())
    local.primitive.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 3, config.hc_count, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    update = torch.randn(2, 3, config.hidden_size)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual, weights = local.read(x)
        raw = source(reference_x.flatten(-2))
        expected = raw if read_only else raw[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if not read_only:
            actual = actual + local.inject(x, update, weights).sum(-2)
            expected = expected + (reference_x + raw[2][..., None] * update[..., None, :]).sum(-2)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    assert_parameters_and_gradients(local.primitive, source)


@pytest.mark.parametrize("autocast", [False, True])
def test_local_mhc_matches_source_outputs_gradients_and_precision(autocast):
    config = MiniFrontier11Config.tiny()
    local = SinglePassMHC(config)
    source = SinglePassHC(
        SimpleNamespace(
            hc_mult=local.streams,
            hc_eps=local.eps,
            norm_eps=local.norm_eps,
            hc_sinkhorn_iters=local.sinkhorn_iters,
            dim=config.hidden_size,
            initializer_range=config.initializer_range,
        )
    )
    source.load_state_dict(
        {
            "projection" if name == "fn" else name: value
            for name, value in local.state_dict().items()
        },
        strict=True,
    )
    dtype = torch.bfloat16 if autocast else torch.float32
    x = torch.randn(2, 3, config.hc_count, config.hidden_size, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    update = torch.randn(2, 3, config.hidden_size, dtype=dtype, requires_grad=True)
    reference_update = update.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        pre, post, matrix = local.coefficients(x)
        ref_pre, ref_post, ref_matrix = source.coefficients(reference_x)
        actual = local.mix(x, pre) + local.inject(x, update, post, matrix).sum(-2)
        expected = source.mix(reference_x, ref_pre) + source.combine(
            reference_update, reference_x, ref_post, ref_matrix
        ).sum(-2)
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(update.grad, reference_update.grad, atol=0, rtol=0)
    assert_parameters_and_gradients(
        local, source, lambda name: "projection" if name == "fn" else name
    )
