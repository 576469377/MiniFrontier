"""MF1 browser requests preserve input and loaded-checkpoint identity without inference."""

import base64
import hashlib
import io
import json
from dataclasses import asdict, replace
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


@pytest.mark.parametrize("family", ["minifrontier1", "minifrontier11"])
def test_bound_training_pixels_match_native_cli_generic_chat_and_browser(
    tmp_path, monkeypatch, family
):
    from minifrontier.commands import minifrontier1 as cli
    from minifrontier.data import sha256
    from minifrontier.data.minifrontier1 import prepare_media, train_tokenizer
    from minifrontier.inference import minifrontier1 as native
    from minifrontier.inference import runtime
    from minifrontier.models.factory import build_model
    from minifrontier.models.minifrontier1 import MiniFrontier1Config
    from minifrontier.models.minifrontier11 import MiniFrontier11Config

    config_cls = MiniFrontier1Config if family == "minifrontier1" else MiniFrontier11Config
    config = replace(
        config_cls.tiny(300), protected_media_tokens=1024, max_position_embeddings=4096
    )
    model = build_model(family, asdict(config), phase="dense_pretrain")
    tok = train_tokenizer([], tmp_path / "tokenizer.json", 300)
    path = tmp_path / "model.pt"
    torch.save(
        dict(
            model_name=family,
            model=model.state_dict(),
            config=asdict(config),
            chat_template="mf1-control-v1",
            phase="dense_pretrain",
            stage="pretrain",
            step=1,
            run_spec={"dataset_manifest_sha256": "fixture"},
            tokenizer_sha256=sha256(tmp_path / "tokenizer.json"),
        ),
        path,
    )
    (tmp_path / "inference_media_profile.json").write_text(
        json.dumps(
            dict(dataset_manifest_sha256="fixture", image_max_features=49, video_max_features=196)
        )
    )
    model, _, _ = runtime.load_checkpoint(path)
    img = tmp_path / "one.png"
    Image.new("RGB", (640, 480), color="navy").save(img)
    image_data = base64.b64encode(img.read_bytes()).decode()
    seen = []

    def capture(_model, ids, *, media, **kwargs):
        seen.append((ids, media))
        return ids

    monkeypatch.setattr(native, "generate_ids", capture)
    monkeypatch.setattr(runtime, "generate_ids", capture)
    for kind, cap, representation in [
        ("image", 49, "standard"),
        ("image", 100, "standard"),
        ("video", 196, "standard"),
        ("image", 49, "document"),
    ]:
        video = kind == "video"
        times = [0.0, 1.0] if video else None
        frame_count = 2 if video else 1
        resource = dict(
            uri=img.name, sha256=sha256(img), max_features=cap, representation=representation
        )
        if video:
            resource.update(
                frames=[img.name] * frame_count,
                frame_sha256=[sha256(img)] * frame_count,
                timestamps=times,
            )
        expected = prepare_media(resource, config, tmp_path, 1024)
        cli_args = ["generate", "--checkpoint", str(path), "--prompt", "Q", "--max-new-tokens", "1"]
        kwargs = {}
        web = dict(kind=kind, frames=[image_data] * frame_count)
        if video:
            cli_args += ["--video-frame", str(img)] * frame_count + [
                "--timestamps",
                json.dumps(times),
            ]
            kwargs.update(video_frames=[img] * frame_count, timestamps=times)
            web.update(timestamps=times)
        else:
            cli_args += ["--image", str(img)]
            kwargs.update(images=[img])
            if cap == 100:
                cli_args += ["--max-image-features", "100"]
                kwargs.update(max_image_features=100)
                web.update(max_features=100)
            if representation == "document":
                cli_args += ["--image-representation", "document"]
                kwargs.update(image_representation="document")
                web.update(representation="document")
        cli.main(cli_args)
        runtime.respond(model, tok, "Q", **kwargs)
        web_ids, web_spans, _ = demo.prepare_request(dict(prompt="Q", media=[web]), model, tok)
        for ids, spans in [*seen, (web_ids, web_spans)]:
            assert ids.tolist() == web_ids.tolist()
            assert len(spans) == len(expected)
            for actual, reference in zip(spans, expected, strict=True):
                torch.testing.assert_close(actual["patches"], reference["patches"], atol=0, rtol=0)
                assert actual["feature_count"] == reference["feature_count"]
        seen.clear()
