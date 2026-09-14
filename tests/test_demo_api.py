"""Exercise HTTP diagnostics with metadata fixtures and mocked model loading only."""

import http.client
import json
import pickle
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from minifrontier.inference import demo


@pytest.fixture
def saved_run(tmp_path):
    folder = tmp_path / "formal" / "minikimik3" / "K1"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(
        json.dumps(
            dict(
                kind="strategy",
                model_name="minikimik3",
                stage="pretrain",
                ce_token_budget=1_000_000,
            )
        )
    )
    (folder / "status.json").write_text(
        json.dumps(
            dict(
                step=100,
                stage="pretrain",
                state="running",
                token_ledger=dict(ce_tokens=12345),
            )
        )
    )
    (folder / "checkpoint.pt").write_bytes(b"not a real checkpoint")
    return folder


@pytest.fixture
def api(tmp_path, saved_run, monkeypatch, request):
    def load(path, device):
        return (
            object(),
            object(),
            dict(
                model_name="minikimik3",
                stage="pretrain",
                step=100,
                phase="dense_pretrain",
                chat_template="legacy",
            ),
        )

    monkeypatch.setattr(demo, "load_checkpoint", load)
    monkeypatch.setattr(demo, "text_response", lambda *a, **kw: "fixture output")
    server = demo.create_server(
        root=tmp_path,
        host="127.0.0.1",
        port=0,
        include_experiments=True,
        device=getattr(request, "param", "cpu"),
    )
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    worker.start()

    def request(method, path, payload=None, *, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            body = json.dumps(payload).encode() if raw is None and payload is not None else raw
            connection.request(
                method, path, body=body, headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    yield request
    server.shutdown()
    worker.join(timeout=5)
    server.server_close()
    assert not worker.is_alive()


def selected_request(api, **overrides):
    status, rows = api("GET", "/api/models")
    assert status == 200 and len(rows) == 1
    row = rows[0]
    return dict(model=row["id"], version=row["version"], prompt="Test", **overrides)


def test_default_scope_and_response_identify_selected_save(api, monkeypatch):
    status, info = api("GET", "/api/info")
    assert status == 200
    assert info["default_stage"] == "formal" and "minifrontier1" in info["model_families"]
    assert info["modalities"] == ["text"]
    payload = selected_request(api)
    observed = []
    monkeypatch.setattr(demo, "text_response", lambda *a, **kw: observed.append(kw) or "output")
    status, result = api("POST", "/api/generate", payload)
    assert status == 200 and result["text"] == "output"
    assert result["model"] == result["run"] == payload["model"]
    assert result["version"] == payload["version"] and result["artifact"] == "checkpoint.pt"
    assert result["model_name"] == "minikimik3" and result["step"] == 100
    assert result["stage"] == "pretrain" and result["generation_mode"] == "completion"
    assert result["temperature"] == 0 and result["top_p"] == 1 and result["seed"] == 0
    assert observed[0]["chat"] is False and observed[0]["temperature"] == 0
    assert result["seconds"] >= result["load_seconds"] + result["generation_seconds"] >= 0


@pytest.mark.parametrize("change", ["before_request", "during_load", "saved_metadata"])
def test_rolling_save_conflicts_never_generate(api, saved_run, monkeypatch, change):
    payload = selected_request(api)
    monkeypatch.setattr(
        demo, "text_response", lambda *a, **kw: pytest.fail("generated wrong version")
    )
    original = demo.load_checkpoint
    if change == "before_request":
        (saved_run / "checkpoint.pt").write_bytes(b"updated save")
        monkeypatch.setattr(
            demo, "load_checkpoint", lambda *a: pytest.fail("loaded stale selection")
        )
    else:

        def replaced(path, device):
            model, tokenizer, meta = original(path, device)
            if change == "during_load":
                (saved_run / "checkpoint.pt").write_bytes(b"updated during load")
            else:
                meta["step"] = 101
            return model, tokenizer, meta

        monkeypatch.setattr(demo, "load_checkpoint", replaced)
    status, result = api("POST", "/api/generate", payload)
    assert status == 409 and result["code"] == "checkpoint_changed"


def test_version_is_checked_after_waiting_for_another_request(api, saved_run, monkeypatch):
    payload = selected_request(api)
    entered, finish = threading.Event(), threading.Event()
    calls = []
    results = []

    def generate(*args, **kwargs):
        calls.append("generate")
        entered.set()
        assert finish.wait(3)
        return "first result"

    monkeypatch.setattr(demo, "text_response", generate)
    first = threading.Thread(target=lambda: results.append(api("POST", "/api/generate", payload)))
    second = threading.Thread(target=lambda: results.append(api("POST", "/api/generate", payload)))
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        (saved_run / "checkpoint.pt").write_bytes(b"save after first model loaded")
    finally:
        finish.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert sorted(status for status, _ in results) == [200, 409]
    assert calls == ["generate"]
    first_result = next(body for status, body in results if status == 200)
    assert first_result["version"] == payload["version"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", True),
        ("temperature", "0"),
        ("temperature", -1),
        ("temperature", 3),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_p", False),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_p", None),
        ("seed", True),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", 4294967296),
        ("max_new_tokens", False),
        ("max_new_tokens", 1.5),
        ("max_new_tokens", "64"),
        ("mode", []),
        ("stage", {}),
        ("version", None),
        ("prompt", "  "),
    ],
)
def test_invalid_generation_settings_do_not_load(api, monkeypatch, field, value):
    payload = selected_request(api)
    payload[field] = value
    monkeypatch.setattr(demo, "load_checkpoint", lambda *a: pytest.fail("invalid request loaded"))
    status, result = api("POST", "/api/generate", payload)
    assert status == 400 and result["code"] == "invalid_request"


@pytest.mark.parametrize("raw", [b"null", b"[]", b"{", b"\xff"])
def test_bad_request_body_has_json_error(api, raw):
    status, result = api("POST", "/api/generate", raw=raw)
    assert status == 400 and result["error"]


def test_seeded_sampling_repeats_and_restores_rng(api, monkeypatch):
    monkeypatch.setattr(demo, "text_response", lambda *a, **kw: str(torch.rand(3).tolist()))
    payload = selected_request(api, temperature=0.8, top_p=0.9, seed=123)
    before = torch.get_rng_state()
    status, first = api("POST", "/api/generate", payload)
    assert status == 200 and torch.equal(torch.get_rng_state(), before)
    status, second = api("POST", "/api/generate", payload)
    assert status == 200 and second["text"] == first["text"]
    payload["seed"] = 124
    status, third = api("POST", "/api/generate", payload)
    assert status == 200 and third["text"] != first["text"]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RuntimeError("bad weights"), 422, "generation_failed"),
        (ValueError("tokenizer does not match checkpoint"), 422, "generation_failed"),
        (pickle.UnpicklingError("invalid checkpoint"), 422, "generation_failed"),
        (torch.cuda.OutOfMemoryError("fixture OOM"), 503, "insufficient_memory"),
    ],
)
def test_load_failures_have_friendly_status(api, monkeypatch, error, status, code):
    payload = selected_request(api)

    def fail(*args):
        raise error

    monkeypatch.setattr(demo, "load_checkpoint", fail)
    actual, result = api("POST", "/api/generate", payload)
    assert actual == status and result["code"] == code and result["error"]


@pytest.mark.parametrize("api", ["cuda:0"], indirect=True)
def test_failure_cleanup_runs_after_traceback_releases_model(api, monkeypatch):
    references, observations = [], []
    released = threading.Event()

    class FakeModel:
        pass

    def fail_load(*args):
        model = FakeModel()
        references.append(weakref.ref(model))
        raise RuntimeError("fixture load failure")

    def clear_cache():
        alive = references[0]() is not None
        observations.append(alive)
        if not alive:
            released.set()

    monkeypatch.setattr(demo, "load_checkpoint", fail_load)
    monkeypatch.setattr(torch.cuda, "empty_cache", clear_cache)
    status, result = api("POST", "/api/generate", selected_request(api))
    assert status == 422 and result["code"] == "generation_failed"
    assert released.wait(3)
    assert observations[0] is True and observations[-1] is False


@pytest.mark.parametrize("api", ["cuda:0"], indirect=True)
def test_failed_request_keeps_loading_slot_until_cleanup_finishes(api, monkeypatch):
    cleanup_started, finish_cleanup, second_loaded = (threading.Event() for _ in range(3))
    loads, cleanups = [], []

    def fail_load(*args):
        loads.append("load")
        if len(loads) == 2:
            second_loaded.set()
        raise RuntimeError("fixture failure")

    def clear_cache():
        cleanups.append("clear")
        if len(cleanups) == 2:
            cleanup_started.set()
            assert finish_cleanup.wait(3)

    monkeypatch.setattr(demo, "load_checkpoint", fail_load)
    monkeypatch.setattr(torch.cuda, "empty_cache", clear_cache)
    payload = selected_request(api)
    results = []
    first = threading.Thread(target=lambda: results.append(api("POST", "/api/generate", payload)))
    second = threading.Thread(target=lambda: results.append(api("POST", "/api/generate", payload)))
    first.start()
    try:
        assert cleanup_started.wait(3)
        second.start()
        assert not second_loaded.wait(0.05)
    finally:
        finish_cleanup.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert second_loaded.is_set()
    assert [status for status, _ in results] == [422, 422]


@pytest.mark.parametrize(
    ("chat", "expected"),
    [
        (False, [1, 40, 41]),
        (True, [1, 4, 40, 41, 2, 5, 17]),
    ],
)
@pytest.mark.parametrize("family", ["minifrontier1", "minifrontier11"])
def test_mf1_text_uses_native_prompt_protocol(monkeypatch, chat, expected, family):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace()
    tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: SimpleNamespace(ids=[40, 41]),
        get_vocab_size=lambda: 64,
        decode=lambda ids, **kw: "output" if ids == [42] else "wrong output",
    )

    def generate(model, ids, **kwargs):
        assert ids.tolist() == [expected]
        assert kwargs["temperature"] == 0 and kwargs["top_p"] == 1
        return torch.tensor([[*expected, 42]])

    monkeypatch.setattr(demo, "generate_ids", generate)
    monkeypatch.setattr(demo, "respond", lambda *a, **kw: pytest.fail("used source chat template"))
    assert (
        demo.text_response(
            model,
            tokenizer,
            dict(model_name=family),
            "Text",
            chat=chat,
            max_new_tokens=4,
            temperature=0,
            top_p=1,
        )
        == "output"
    )
