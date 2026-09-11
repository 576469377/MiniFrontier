"""Fail closed on missing evidence before a strategy's formal training phase."""

import json
from pathlib import Path

from minifrontier.data import sha256
from minifrontier.provenance import require_source_checkout, source_identity
from minifrontier.storage import GIB, require_space


def _bound_report(reference):
    path = Path(reference.get("path", ""))
    if not path.is_file() or sha256(path) != reference.get("sha256"):
        return {}
    return json.loads(path.read_text())


def initial_start_authorized(evidence, *, model, phase, data_sha256, config_sha256, source_commit):
    """A bound maintainer decision defers repeat preflight runs for first phases only.

    This is permission to start learning, not a passing performance or capability
    result. Runtime correctness, immutable inputs and later phase gates still apply.
    """
    first_phases = {"minifrontier1": "p0", "minikimik3": "K1", "miniqwen4": "Q1"}
    report = _bound_report(evidence.get("initial_training_authorization", {}))
    expected = dict(
        model=model,
        phase=phase,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
        source_commit=source_commit,
    )
    return bool(
        first_phases.get(model) == phase
        and report.get("kind") == "maintainer_initial_training_authorization"
        and report.get("status") == "authorized"
        and report.get("scope") == "first_phase_without_additional_preflight_runs"
        and report.get("authorization", {}).get("user_statement")
        and report.get("qualification_passed") is False
        and report.get("runtime_checks_retained") is True
        and expected in report.get("bindings", [])
        and report.get("prior_evidence")
        and all(_bound_report(ref) for ref in report["prior_evidence"])
    )


def _human_review_waived(evidence, model, phase, data_sha256):
    """A maintainer exception covers manual review only, for named artifacts."""
    report = _bound_report(evidence.get("human_review_waiver", {}))
    return (
        report.get("kind") == "maintainer_human_review_waiver"
        and report.get("status") == "authorized"
        and report.get("scope") == "learning_project_pretraining_manual_review_only"
        and report.get("human_review_completed") is False
        and bool(report.get("authorization", {}).get("user_statement"))
        and phase["budget_scope"] == "main"
        and dict(model=model, phase=phase["id"], data_sha256=data_sha256)
        in report.get("bindings", [])
    )


def _pilot_recipe_passed(completed, dependency, phase, phases, commit):
    """Recipe pilots qualify choices; their generation verdict remains separate."""
    if (
        phases[dependency]["budget_scope"] != "recipe_pilot"
        or phase.get("dependency_quality_scopes", {}).get(dependency) != "recipe_viability"
    ):
        return False
    report = _bound_report(completed.get("recipe_viability", {}))
    return (
        report.get("kind") == "recipe_pilot_viability"
        and report.get("source_commit") == commit
        and report.get("checkpoint_sha256") == completed.get("checkpoint_sha256")
        and report.get("passed") is True
        and all(
            report.get("checks", {}).get(key) is True
            for key in (
                "finite_updates",
                "heldout_learnability",
                "optimizer_mtp_choices_bound",
                "current_numerical_contracts",
                "generation_review_recorded",
            )
        )
    )


def check(plan_path, phase_id, evidence_path, *, data, config, output):
    plan_path, data = Path(plan_path).resolve(), Path(data).resolve()
    root = require_source_checkout()
    plan = json.loads(plan_path.read_text())
    if sha256(root / plan["source_document"]) != plan["source_document_sha256"]:
        raise ValueError("strategy document changed; review and regenerate its plan")
    phases = {p["id"]: p for p in plan["phases"]}
    if phase_id not in phases:
        raise ValueError("unknown strategy phase")
    phase = phases[phase_id]
    evidence = json.loads(Path(evidence_path).read_text())
    identity = source_identity()
    errors = []
    if not identity["commit"] or identity["dirty"]:
        errors.append("formal execution requires a clean source commit")
    if evidence.get("source_commit") != identity["commit"]:
        errors.append("evidence belongs to a different source commit")
    manifest = json.loads((data / "manifest.json").read_text())
    direct_start = initial_start_authorized(
        evidence,
        model=plan["model"],
        phase=phase_id,
        data_sha256=sha256(data / "manifest.json"),
        config_sha256=sha256(config),
        source_commit=identity["commit"],
    )
    if evidence.get("data_sha256") != sha256(data / "manifest.json"):
        errors.append("audited data manifest is missing or different")
    if evidence.get("config_sha256") != sha256(config):
        errors.append("audited model configuration is missing or different")
    tokenizer = (
        json.loads(Path(evidence["tokenizer_comparison"]).read_text())
        if evidence.get("tokenizer_comparison")
        else {}
    )
    if (
        not tokenizer.get("frozen")
        or tokenizer.get("selected") != manifest["tokenizer"]["vocab_size"]
    ):
        errors.append("frozen tokenizer evidence is missing or incompatible")
    for dependency in [] if direct_start else phase["depends_on"]:
        completed = evidence.get("completed_phases", {}).get(dependency, {})
        path = completed.get("checkpoint")
        if (
            not path
            or not Path(path).is_file()
            or sha256(path) != completed.get("checkpoint_sha256")
        ):
            errors.append(f"missing immutable checkpoint for {dependency}")
        if not completed.get("quality_passed") and not _pilot_recipe_passed(
            completed, dependency, phase, phases, identity["commit"]
        ):
            errors.append(
                f"{dependency} has no quality pass; token completion alone is insufficient"
            )
    for gate in [] if direct_start else phase["required_evidence"]:
        path = evidence.get("reports", {}).get(gate)
        report = json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}
        if report.get("source_commit") != identity["commit"] or not report.get("passed"):
            errors.append(f"missing current passing evidence: {gate}")
    if phase["budget_scope"] not in {"diagnostic", "recipe_pilot"}:
        audit = evidence.get("data_audit", {})
        for key in (
            "source_licenses_reviewed",
            "split_groups_disjoint",
            "sealed_test",
        ):
            if not audit.get(key):
                errors.append(f"data admission missing {key}")
        if not audit.get("readable_200_passed") and not _human_review_waived(
            evidence, plan["model"], phase, sha256(data / "manifest.json")
        ):
            errors.append("data admission missing readable_200_passed or bound maintainer waiver")
        if audit.get("minimum_source_holdout_fraction", 0) < 0.005:
            errors.append("some source has less than 0.5% group holdout")
        base_pretraining = phase["budget_scope"] == "main"
        minimum_periodic = 1_000_000 if base_pretraining else 5_000_000
        if audit.get("periodic_validation_ce_tokens", 0) < minimum_periodic:
            errors.append(f"periodic validation is smaller than {minimum_periodic} CE tokens")
        if base_pretraining and audit.get("phase_end_validation_ce_tokens", 0) < 5_000_000:
            errors.append("phase-end pretraining validation is smaller than 5M CE tokens")
        if (
            phase["budget_scope"]
            in {
                "posttraining",
                "teachers_9",
                "teachers_12",
                "draft",
                "release",
            }
            and audit.get("sft_unique_validation_questions", 0) < 2000
        ):
            errors.append("SFT validation needs 2000 independent questions")
        if (
            phase.get("image_occurrences")
            or json.loads(Path(config).read_text()).get("vision_config")
        ) and audit.get("vision_unique_validation_groups", 0) < 1000:
            errors.append("vision validation needs 1000 independent media groups")
        if not direct_start:
            profile_path = evidence.get("performance")
            profile = (
                json.loads(Path(profile_path).read_text())
                if profile_path and Path(profile_path).is_file()
                else {}
            )
            if profile.get("measured_updates", 0) < 200:
                errors.append("missing 200 real updates after profile warmup")
            if profile.get("recipe", {}).get("performance_profile", {}).get("warmup", 0) < 50:
                errors.append("performance profile has fewer than 50 warmup updates")
            measured = profile.get("recipe", {})
            if measured.get("data_sha256") != sha256(data / "manifest.json"):
                errors.append("performance profile uses a different corpus")
            if measured.get("phase") != phase["attention_phase"]:
                errors.append("performance profile uses a different attention phase")
            if measured.get("source", {}).get("commit") != identity["commit"]:
                errors.append("performance profile uses a different implementation")
            requested_config = json.loads(Path(config).read_text())
            if any(
                measured.get("config", {}).get(key) != value
                for key, value in requested_config.items()
            ):
                errors.append("performance profile uses a different model configuration")
    require_space(output, 0, reserve_bytes=plan["min_disk_free_gib"] * GIB)
    return dict(
        allowed=not errors,
        initial_start_authorized=direct_start,
        independent_production_qualification_passed=not direct_start and not errors,
        errors=errors,
        model=plan["model"],
        phase=phase,
        source=identity,
        data_sha256=sha256(data / "manifest.json"),
        plan_sha256=sha256(plan_path),
    )


def validate_arguments(args):
    if not all((args.strategy_plan, args.strategy_phase, args.strategy_evidence, args.config)):
        raise ValueError("strategy runs require plan, phase, immutable evidence and config")
    report = check(
        args.strategy_plan,
        args.strategy_phase,
        args.strategy_evidence,
        data=args.data,
        config=args.config,
        output=args.output,
    )
    phase = report["phase"]
    budget = args.ce_tokens or args.input_tokens or args.response_tokens
    if args.model != report["model"]:
        report["errors"].append("model and strategy disagree")
    if budget is None or not phase["budget_min"] <= budget <= phase["budget_max"]:
        report["errors"].append("requested actual token budget is outside strategy bounds")
    if args.sequence_length not in phase["sequence_lengths"]:
        report["errors"].append("context bucket is not part of this phase")
    if phase["budget_scope"] == "main" and not args.pretraining_eval:
        report["errors"].append(
            "base pretraining requires fixed CE-budget validation and main-CE cadence"
        )
    objective = phase["objective"]
    if objective == "ce_tokens" and (
        args.stage not in {"pretrain", "sparse_cpt"} or args.ce_tokens is None
    ):
        report["errors"].append("main CE phase requires its language-model objective")
    if objective == "input_tokens" and args.stage != "dense_distill":
        report["errors"].append("input-only phase requires the frozen indexer objective")
    if objective == "assistant_tokens" and args.stage != "sft":
        report["errors"].append("assistant-token phase requires SFT")
    if phase["id"] == "V1" and not args.visual_warmup:
        report["errors"].append("V1 must freeze learned text and train native vision/aligner")
    if args.stage == "sft" and (
        args.max_data_epochs is None
        or args.max_data_epochs > 2
        or not (args.token_mixture or args.media_mixture)
    ):
        report["errors"].append("formal SFT needs a resumable sampler with at most two data epochs")
    if phase["budget_scope"] not in {"diagnostic", "recipe_pilot", "vision_diagnostic"}:
        if not report["initial_start_authorized"]:
            evidence = json.loads(Path(args.strategy_evidence).read_text())
            profile_path = Path(evidence.get("performance", ""))
            measured = (
                json.loads(profile_path.read_text()).get("recipe", {})
                if profile_path.is_file()
                else {}
            )
            if getattr(args, "pretraining_program", None):
                from .pretraining import PretrainingProgram

                binding = PretrainingProgram(args).binding
                qualified = measured.get("pretraining_program", {})
                if any(
                    qualified.get(k) != binding[k]
                    for k in ("program_id", "model", "recipe_sha256", "phase")
                ):
                    report["errors"].append(
                        "performance profile does not measure this frozen pretraining program"
                    )
            for key in (
                "sequence_length",
                "batch_size",
                "grad_accum",
                "input_batch_tokens",
                "input_batch_policy",
                "input_batch_schedule",
                "visual_warmup",
                "vision_lr",
                "projector_lr",
            ):
                if measured.get(key) != getattr(args, key):
                    report["errors"].append(f"performance profile does not measure this {key}")
            plan = json.loads(Path(args.strategy_plan).read_text())
            if measured.get("world_size") != len(plan["gpu_ids"]):
                report["errors"].append("performance profile has a different device count")
            for key in ("token_mixture", "media_mixture"):
                path = getattr(args, key)
                if measured.get(key) != (json.loads(Path(path).read_text()) if path else None):
                    report["errors"].append(
                        "performance profile uses a different modality/domain mixture"
                    )
        if phase["image_occurrences"] or phase["video_examples"]:
            mixture = json.loads(Path(args.media_mixture).read_text()) if args.media_mixture else {}
            if (
                mixture.get("image_occurrences") != phase["image_occurrences"]
                or mixture.get("video_examples", 0) != phase["video_examples"]
            ):
                report["errors"].append(
                    "formal media phase requires the planned image/video occurrence quotas"
                )
    if report["initial_start_authorized"] and getattr(args, "init", None):
        report["errors"].append("direct initial pretraining starts require random initialization")
    if objective == "response_tokens" and args.stage not in {"grpo", "mopd", "opd"}:
        report["errors"].append("response-token phase requires its on-policy objective")
    if objective not in {"ce_tokens", "input_tokens", "assistant_tokens", "response_tokens"}:
        report["errors"].append("this phase requires a separate teacher/draft/release executor")
    if report["errors"]:
        raise ValueError("strategy phase blocked:\n" + "\n".join(report["errors"]))
    return report


def validate_runtime(plan_path, phase_id, attention_phase, world_size):
    """The phase inherited from init/resume must agree with the admitted profile."""
    plan = json.loads(Path(plan_path).read_text())
    phase = next(p for p in plan["phases"] if p["id"] == phase_id)
    if attention_phase != phase["attention_phase"]:
        raise ValueError("effective checkpoint attention phase differs from strategy")
    if world_size != len(plan["gpu_ids"]):
        raise ValueError("actual distributed device count differs from strategy profile")
