import json

from minifrontier.data import sha256
from minifrontier.training.strategy_gate import initial_start_authorized


def test_initial_permission_is_bound_and_never_qualifies_a_later_phase(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"historical_result": "retained"}))
    actual = dict(
        model="minifrontier1",
        phase="p0",
        data_sha256="data",
        config_sha256="config",
        source_commit="source",
    )
    report = tmp_path / "authorization.json"
    report.write_text(
        json.dumps(
            dict(
                kind="maintainer_initial_training_authorization",
                status="authorized",
                scope="first_phase_without_additional_preflight_runs",
                authorization=dict(user_statement="Start without additional preflight runs"),
                qualification_passed=False,
                runtime_checks_retained=True,
                bindings=[actual],
                prior_evidence=[dict(path=str(prior), sha256=sha256(prior))],
            )
        )
    )
    evidence = dict(initial_training_authorization=dict(path=str(report), sha256=sha256(report)))
    assert initial_start_authorized(evidence, **actual)
    for field in ("phase", "model", "data_sha256", "config_sha256", "source_commit"):
        assert not initial_start_authorized(evidence, **dict(actual, **{field: "changed"}))
    prior.write_text("{}")
    assert not initial_start_authorized(evidence, **actual)
