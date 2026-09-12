import json

from minifrontier.data import sha256
from minifrontier.training.strategy_gate import continuous_phase_profile, initial_start_authorized


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


def test_continuous_main_phase_observation_does_not_skip_attention_conversion():
    phases = {
        "D1": dict(budget_scope="main", objective="ce_tokens", attention_phase="dense_pretrain"),
        "D2": dict(
            budget_scope="main",
            objective="ce_tokens",
            attention_phase="dense_pretrain",
            depends_on=["D1"],
        ),
        "D3": dict(
            budget_scope="indexer",
            objective="input_tokens",
            attention_phase="dense_distill",
            depends_on=["D2"],
        ),
        "D4": dict(
            budget_scope="main",
            objective="ce_tokens",
            attention_phase="sparse_cpt",
            depends_on=["D3"],
        ),
    }
    plan = dict(performance=dict(continuous_main_phases="observe_during_training"))
    assert continuous_phase_profile(plan, phases["D2"], phases)
    assert not continuous_phase_profile({}, phases["D2"], phases)
    assert not continuous_phase_profile(plan, phases["D3"], phases)
    assert not continuous_phase_profile(plan, phases["D4"], phases)
    phases["D4"]["depends_on"] = ["D2"]
    assert not continuous_phase_profile(plan, phases["D4"], phases)


def test_continuation_still_requires_parent_quality_and_current_data(tmp_path, monkeypatch):
    from minifrontier.training import strategy_gate

    document = tmp_path / "strategy.md"
    document.write_text("strategy")
    plan = dict(
        model="minideepseekv4",
        source_document=document.name,
        source_document_sha256=sha256(document),
        min_disk_free_gib=0,
        performance=dict(continuous_main_phases="observe_during_training"),
        phases=[
            dict(
                id="D1",
                budget_scope="main",
                objective="ce_tokens",
                attention_phase="dense_pretrain",
            ),
            dict(
                id="D2",
                budget_scope="main",
                objective="ce_tokens",
                attention_phase="dense_pretrain",
                depends_on=["D1"],
                required_evidence=[],
                image_occurrences=0,
            ),
        ],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps(dict(tokenizer=dict(vocab_size=65536))))
    config = tmp_path / "config.json"
    config.write_text("{}")
    tokenizer = tmp_path / "tokenizer-freeze.json"
    tokenizer.write_text(json.dumps(dict(frozen=True, selected=65536)))
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent")
    evidence = dict(
        source_commit="current",
        data_sha256=sha256(data / "manifest.json"),
        config_sha256=sha256(config),
        tokenizer_comparison=str(tokenizer),
        completed_phases=dict(
            D1=dict(
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                quality_passed=True,
            )
        ),
        data_audit=dict(
            source_licenses_reviewed=True,
            split_groups_disjoint=True,
            sealed_test=True,
            readable_200_passed=True,
            minimum_source_holdout_fraction=0.01,
            periodic_validation_ce_tokens=1000000,
            phase_end_validation_ce_tokens=5000000,
        ),
    )
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(strategy_gate, "require_source_checkout", lambda: tmp_path)
    monkeypatch.setattr(
        strategy_gate, "source_identity", lambda: dict(commit="current", dirty=False)
    )
    monkeypatch.setattr(strategy_gate, "require_space", lambda *a, **k: None)

    def check():
        evidence_path.write_text(json.dumps(evidence))
        return strategy_gate.check(
            plan_path, "D2", evidence_path, data=data, config=config, output=tmp_path
        )

    result = check()
    assert result["allowed"] and result["profile_during_formal_updates"]
    assert not result["independent_production_qualification_passed"]
    evidence["completed_phases"]["D1"]["quality_passed"] = False
    assert any("quality pass" in e for e in check()["errors"])
    evidence["completed_phases"]["D1"]["quality_passed"] = True
    evidence["data_sha256"] = "wrong"
    assert any("data manifest" in e for e in check()["errors"])
