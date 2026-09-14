"""MF1 browser requests preserve input and loaded-checkpoint identity without inference."""

import base64
import hashlib
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from minifrontier.inference import minifrontier1_demo as demo


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[30 + ord(char) % 10 for char in text])

    def get_vocab_size(self):
        return 64

    def decode(self, tokens, skip_special_tokens=True):
        return "ok"


@pytest.fixture(params=["1.0-reference-v1", "1.1-reference-v1"], ids=["mf1", "mf11"])
def model(request):
    return SimpleNamespace(
        config=SimpleNamespace(
            model_version=request.param,
            max_position_embeddings=256,
            protected_media_tokens=64,
            vision_config=SimpleNamespace(patch_size=16),
        ),
        parameters=lambda: iter([torch.zeros(3)]),
    )


def test_completion_has_no_chat_tokens_and_rejects_media(model):
    payload = dict(prompt="Hello", input_mode="completion", max_new_tokens=4)
    ids, spans, plan = demo.prepare_request(payload, model, Tokenizer())
    assert ids.tolist() == [[1, *Tokenizer().encode("Hello").ids]]
    assert not spans and plan["vision_tokens"] == 0
    assert plan["remaining_tokens"] == 256 - ids.numel() - 4
    chat, _, _ = demo.prepare_request(dict(payload, input_mode="chat"), model, Tokenizer())
    assert chat[0, :2].tolist() == [1, 4]
    assert chat[0, -3:].tolist() == [2, 5, 17]
    with pytest.raises(ValueError, match="不接收媒体"):
        demo.prepare_request(dict(payload, media=[{}]), model, Tokenizer())


def test_request_identity_rejects_changed_input_or_budget(model):
    payload = dict(prompt="Hello", max_new_tokens=4)
    _, _, plan = demo.prepare_request(payload, model, Tokenizer())
    bound = dict(payload, request_id=plan["request_id"])
    assert demo.prepare_request(bound, model, Tokenizer())[2]["request_id"] == plan["request_id"]
    for change in ({"prompt": "Other"}, {"max_new_tokens": 5}, {"input_mode": "completion"}):
        with pytest.raises(ValueError, match="输入已变化"):
            demo.prepare_request(bound | change, model, Tokenizer())


@pytest.mark.parametrize(
    "media",
    [
        None,
        "image",
        [None],
        [{"kind": "image", "frames": None}],
        [{"kind": "image", "frames": [None]}],
        [{"kind": "image", "frames": "abc"}],
        [{"kind": "video", "frames": ["abc", "abc"], "timestamps": [0, 0]}],
    ],
)
def test_malformed_media_is_a_validation_error(model, media):
    with pytest.raises(ValueError):
        demo.prepare_request(dict(prompt="Hello", media=media), model, Tokenizer())


def test_preview_preserves_names_timestamps_and_processed_dimensions(model):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32)).save(stream, format="PNG")
    frame = base64.b64encode(stream.getvalue()).decode()
    media = [
        dict(kind="video", name="clip.mp4", frames=[frame] * 4, timestamps=[0.1, 0.4, 0.8, 1.2])
    ]
    _, _, plan = demo.prepare_request(dict(prompt="Hello", media=media), model, Tokenizer())
    assert plan["media"] == [
        dict(
            kind="video",
            name="clip.mp4",
            timestamps=[0.1, 0.4, 0.8, 1.2],
            frames=4,
            width=32,
            height=32,
            tokens=2,
        )
    ]


def install_loader(monkeypatch, model, family="minifrontier1"):
    monkeypatch.setattr(
        demo,
        "load_checkpoint",
        lambda path, device: (
            model,
            Tokenizer(),
            dict(model_name=family, stage="pretrain", step=8, phase="dense_pretrain"),
        ),
    )


def test_loading_rejects_checkpoint_replaced_during_load(tmp_path, monkeypatch, model):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"old")
    install_loader(monkeypatch, model)
    original = demo.load_checkpoint

    def replace_while_loading(path, device):
        replacement = tmp_path / "replacement.pt"
        replacement.write_bytes(b"new")
        replacement.replace(path)
        return original(path, device)

    monkeypatch.setattr(demo, "load_checkpoint", replace_while_loading)
    with pytest.raises(ValueError, match="加载期间"):
        demo.load_demo_checkpoint(path, "cpu")


@pytest.mark.parametrize("family", ["minifrontier1", "minifrontier11"])
def test_info_and_generation_stay_bound_to_loaded_version(tmp_path, monkeypatch, model, family):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"initial")
    checksum = hashlib.sha256(b"initial").hexdigest()
    install_loader(monkeypatch, model, family)
    captured = {}

    class Server:
        def __init__(self, address, handler):
            captured["handler"] = handler

        def serve_forever(self):
            pass

    monkeypatch.setattr(demo, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(
        demo, "generate_ids", lambda model, ids, **kwargs: torch.cat((ids, torch.tensor([[2]])), 1)
    )
    demo.serve(path, allow_unqualified=True)
    path.write_bytes(b"newer checkpoint")

    def request(route, payload=None):
        import json

        handler = object.__new__(captured["handler"])
        handler.path = route
        response = {}
        handler.reply = lambda value, code=200: response.update(value=value, code=code)
        if payload is None:
            handler.do_GET()
        else:
            raw = json.dumps(payload).encode()
            handler.headers = {"Content-Length": str(len(raw))}
            handler.rfile = io.BytesIO(raw)
            handler.do_POST()
        return response

    info = request("/api/info")["value"]
    assert info["checkpoint_sha256"] == checksum
    assert info["checkpoint"]["step"] == 8
    assert info["checkpoint"]["model_name"] == family
    payload = dict(prompt="Hello", input_mode="completion", max_new_tokens=4)
    plan = request("/api/prepare", payload)["value"]
    assert plan["checkpoint_sha256"] == checksum
    bound = payload | {key: plan[key] for key in ("request_id", "checkpoint_sha256")}
    response = request("/api/generate", bound)
    assert response["code"] == 200
    assert response["value"]["checkpoint"] == info["checkpoint"]
    assert response["value"]["request"]["prompt"] == "Hello"
    assert response["value"]["finish_reason"] == "eos"
    assert request("/api/generate", bound | {"checkpoint_sha256": "different"})["code"] == 400
    assert request("/api/prepare", dict(prompt="Hello", media=[None]))["code"] == 400
