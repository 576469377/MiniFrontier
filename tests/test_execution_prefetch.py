import copy
import json
from types import SimpleNamespace

import pytest
import torch

from minifrontier.data import sha256
from minifrontier.multimodal import TrainingBatch
from minifrontier.training.execution_upgrade import resume_matches
from minifrontier.training.mixture import TokenMixtureCursor
from minifrontier.training.prefetch import OrderedPrefetch, native_window


def test_native_lookahead_discards_unconsumed_window_on_resume():
    class Dataset:
        def __init__(self):
            self.domains = ["text"] * 12
            self.input_counts = [3 + i % 4 for i in range(12)]
            self.ce_counts = [n - 1 for n in self.input_counts]

        def __getitem__(self, index):
            ids = torch.full((1, self.input_counts[index]), index + 1, dtype=torch.long)
            return TrainingBatch(ids, ids.clone())

    data = Dataset()

    def cursor():
        return TokenMixtureCursor(data, {"text": 1.0}, batch_size=4, seed=19)

    def producer(stream):
        return lambda: native_window(data, stream, 4, 25, 8)

    serial, ahead = cursor(), cursor()
    queue = OrderedPrefetch(producer(ahead), ahead.state_dict())
    try:
        for _ in range(3):
            expected, state = producer(serial)()
            actual = queue.next()
            assert actual[1] == expected[1]
            for a, b in zip(actual[0], expected[0], strict=True):
                torch.testing.assert_close(a.input_ids, b.input_ids, atol=0, rtol=0)
                torch.testing.assert_close(a.labels, b.labels, atol=0, rtol=0)
            assert queue.committed == state
        saved = copy.deepcopy(queue.committed)
        queue.future.result()  # A full unconsumed window exists at checkpoint time.
        assert ahead.offset > saved["offset"]
    finally:
        queue.close()
    resumed = cursor()
    resumed.load_state_dict(saved)
    a, state_a = producer(serial)()
    b, state_b = producer(resumed)()
    assert state_a == state_b
    for x, y in zip(a[0], b[0], strict=True):
        torch.testing.assert_close(x.input_ids, y.input_ids, atol=0, rtol=0)


def test_prefetch_delivers_errors_at_the_consumed_window():
    state = SimpleNamespace(n=0)

    def produce():
        state.n += 1
        if state.n == 2:
            raise ValueError("corrupt next image")
        return "first", {"offset": state.n}

    queue = OrderedPrefetch(produce, {"offset": 0})
    try:
        assert queue.next() == "first"
        with pytest.raises(ValueError, match="corrupt next image"):
            queue.next()
        assert queue.committed == {"offset": 1}
    finally:
        queue.close()


def test_execution_upgrade_changes_only_bound_source_and_evidence(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact original checkpoint")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps(dict(passed=True, calculation_unchanged=True)))
    old = dict(
        model_name="miniqwen4",
        source=dict(commit="old", dirty=False),
        batch_size=32,
        strategy=dict(phase="Q1", plan_sha256="plan", evidence_sha256="old-evidence"),
    )
    new = copy.deepcopy(old)
    new["source"]["commit"] = "new"
    new["strategy"]["evidence_sha256"] = "new-evidence"
    record = dict(
        kind="execution_only_resume",
        authorization="calculation unchanged",
        model="miniqwen4",
        previous_source=old["source"],
        current_source=new["source"],
        previous_evidence_sha256="old-evidence",
        current_evidence_sha256="new-evidence",
        checkpoint_sha256=sha256(checkpoint),
        verification=dict(path=str(proof), sha256=sha256(proof)),
    )
    path = tmp_path / "upgrade.json"
    path.write_text(json.dumps(record))
    assert not resume_matches(old, new, checkpoint)
    monkeypatch.setenv("MINIFRONTIER_EXECUTION_UPGRADE", str(path))
    assert resume_matches(old, new, checkpoint)
    assert not resume_matches(old, dict(new, batch_size=64), checkpoint)
    assert not resume_matches(
        old, dict(new, source=dict(commit="unknown", dirty=False)), checkpoint
    )
    checkpoint.write_bytes(b"other checkpoint")
    assert not resume_matches(old, new, checkpoint)
