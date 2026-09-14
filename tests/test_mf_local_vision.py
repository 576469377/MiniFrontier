"""The MF-owned visual dependency closure preserves the original computation."""

import ast
import importlib
import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.models.miniqwen4 import upstream_vision as original_vision
from minifrontier.models.miniqwen4.processing import process_frames as original_frames


@pytest.fixture(params=["minifrontier1", "minifrontier11"])
def implementation(request):
    prefix = "minifrontier.models." + request.param
    package = importlib.import_module(prefix)
    config = (
        package.MF1VisionConfig if request.param == "minifrontier1" else package.MF11VisionConfig
    )
    return (
        importlib.import_module(prefix + ".processing"),
        config,
        importlib.import_module(prefix + ".vision").NativeVision,
    )


def frames(count, width=37, height=23):
    generator = np.random.default_rng(13)
    return [
        Image.fromarray(generator.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for _ in range(count)
    ]


def assert_sample(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("count", [1, 3, 4])
def test_local_image_and_video_processing_matches_original(monkeypatch, count, implementation):
    processing, _, _ = implementation
    images = frames(count)
    kwargs = dict(patch_size=4, max_features=24)
    if count > 1:
        kwargs["timestamps"] = [0.13 + 0.37 * index for index in range(count)]
    actual = processing.process_frames(images, **kwargs)
    monkeypatch.setattr(processing, "source_frames", original_frames)
    expected = processing.process_frames(images, **kwargs)
    assert_sample(actual, expected)


def test_local_document_processing_matches_original(monkeypatch, implementation):
    processing, _, _ = implementation
    image = frames(1, width=473, height=491)[0]
    actual = processing.process_document(image, patch_size=4)
    monkeypatch.setattr(processing, "source_frames", original_frames)
    expected = processing.process_document(image, patch_size=4)
    assert len(actual) == len(expected) == 5
    for local, original in zip(actual, expected, strict=True):
        assert_sample(local, original)


def original_native_vision(vision_class):
    """Use the unchanged MF wrapper with its former dependency symbols as an oracle."""
    scope = dict(
        __name__="mf_original_vision_oracle",
        asdict=asdict,
        SimpleNamespace=SimpleNamespace,
        nn=nn,
        checkpoint=checkpoint,
    )
    for name in (
        "Qwen4ExpVisionBlock",
        "Qwen4ExpVisionPatchEmbed",
        "Qwen4ExpVisionRotaryEmbedding",
        "get_vision_cu_seqlens",
        "get_vision_position_ids",
    ):
        scope[name] = getattr(original_vision, name)
    exec(inspect.getsource(vision_class), scope)
    return scope["NativeVision"]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_local_vision_preserves_parameters_output_and_gradients(
    checkpointing, autocast, implementation
):
    processing, config_type, vision_type = implementation
    config = config_type(
        depth=2,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        patch_size=4,
        output_size=32,
        gradient_checkpointing=checkpointing,
    )
    torch.manual_seed(19)
    local = vision_type(config).train()
    torch.manual_seed(19)
    original = original_native_vision(vision_type)(config).train()
    assert local.state_dict().keys() == original.state_dict().keys()
    for name, value in local.state_dict().items():
        assert torch.equal(value, original.state_dict()[name]), name
    image = processing.process_frames(frames(1), patch_size=4, max_features=12)
    video = processing.process_frames(
        frames(3), patch_size=4, max_features=24, timestamps=[0.1, 0.6, 1.3]
    )
    grid = torch.cat([image["grid_thw"], video["grid_thw"]])
    patches = torch.cat([image["patches"], video["patches"]])
    local_patches = patches.clone().requires_grad_()
    original_patches = patches.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = local(local_patches, grid)
        expected = original(original_patches, grid)
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, atol=0, rtol=0)
        sum(value.float().square().sum() for value in actual).backward()
        sum(value.float().square().sum() for value in expected).backward()
    torch.testing.assert_close(local_patches.grad, original_patches.grad, atol=0, rtol=0)
    original_parameters = dict(original.named_parameters())
    for name, value in local.named_parameters():
        torch.testing.assert_close(value.grad, original_parameters[name].grad, atol=0, rtol=0)
    assert (
        type(local.blocks[0]).__module__.rsplit(".", 1)[0]
        == vision_type.__module__.rsplit(".", 1)[0]
    )


def test_local_visual_sources_do_not_import_another_model(implementation):
    _, _, vision_type = implementation
    directory = Path(inspect.getfile(vision_type)).parent
    for filename in ("vision.py", "processing.py", "upstream_vision.py", "upstream_processing.py"):
        tree = ast.parse((directory / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, filename
                assert not (node.module or "").startswith("minifrontier.models."), filename
            elif isinstance(node, ast.Import):
                assert all(not name.name.startswith("minifrontier.models.") for name in node.names)
    names = {
        node.name
        for node in ast.parse((directory / "upstream_vision.py").read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert "Qwen4ExpVisionPatchMerger" not in names


@pytest.mark.parametrize("package", ["minifrontier1", "minifrontier11"])
def test_model_package_owns_its_computation(package):
    module = importlib.import_module("minifrontier.models." + package)
    directory = Path(inspect.getfile(module)).parent
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, path
                imports = [node.module or ""]
            elif isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            else:
                continue
            for imported in imports:
                if imported.startswith("minifrontier.models."):
                    assert imported in {
                        "minifrontier.models.cache_utils",
                        "minifrontier.models.common",
                        "minifrontier.models.grouped_experts",
                    }, path
    name = "MiniFrontier1ForCausalLM" if package == "minifrontier1" else "MiniFrontier11ForCausalLM"
    assert getattr(module, name).__bases__ == (nn.Module,)
    if package == "minifrontier11":
        assert not (directory / "mtp.py").exists()
        assert not (directory / "draft.py").exists()
        assert not hasattr(importlib.import_module(module.__name__ + ".residual"), "GatedResidual")
    else:
        assert not hasattr(importlib.import_module(module.__name__ + ".residual"), "SinglePassMHC")
