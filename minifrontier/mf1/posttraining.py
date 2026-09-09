"""Native on-policy RL/OPD and target-frozen draft objectives with replayable evidence."""

import copy
import json
import random
import shutil
from pathlib import Path
from typing import Any, cast

import torch

from minifrontier.data import sha256
from minifrontier.inference import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1 import MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.draft import MF1Draft
from minifrontier.multimodal import move
from minifrontier.training.deepseek_opd import full_vocab_reverse_kl
from minifrontier.training.distributions import forbidden_actions
from minifrontier.training.posttrain import dpo_loss, grouped_advantages, token_log_probs
from minifrontier.training.runtime import atomic_save, restore_rng, rng_state
from minifrontier.training.tool_environment import ToolEnvironment

from .data import RecordDataset, encode_record, safe_text, write_json
from .optim import make_optimizer
from .strategy import PHASES, TEACHER_SLOTS, bindings, validate_gate


def verify_answer(record, text, tool_trace=None):
    expected = record.get("expected")
    if record.get("supervision", {}).get("verifier") == "python":
        from minifrontier.training.verifiers import check_python

        task = record["verification"]
        result = check_python(text, task["entry_point"], task["tests"])
        return float(result["score"]), dict(kind="python", result=result)
    if expected is None:
        return 0.0, dict(status="unverifiable", reason="no qualified verifier/expected output")
    correct = text.strip() == str(expected).strip()
    if record.get("tool_environment"):
        correct = (
            correct
            and bool(tool_trace)
            and all("error" not in entry for turn in tool_trace for entry in turn["result"])
        )
    return float(correct), dict(
        status="verified",
        kind="exact",
        correct=correct,
        format_reward=0.0,
        success_reward=float(correct),
    )


@torch.no_grad()
def rollout(model, dataset, record, *, max_tokens=256, max_tool_turns=3):
    device = next(model.parameters()).device
    prepared = encode_record(
        record, dataset.tokenizer, model.config, dataset.media_root, generation_prompt=True
    )
    ids = prepared["input_ids"].to(device)
    media = move(prepared["media"], device)
    labels = torch.full_like(ids, -100)
    behavior_logp = torch.zeros((1, ids.shape[1] - 1), device=device)
    trace, generated = [], 0
    environment = (
        ToolEnvironment(record["tool_environment"]) if record.get("tool_environment") else None
    )
    for _ in range(max_tool_turns if environment else 1):
        remaining = min(max_tokens - generated, model.config.max_position_embeddings - ids.shape[1])
        if remaining < 1:
            break
        result, sampled_logp, _ = generate_ids(
            model,
            ids,
            media=media,
            max_new_tokens=remaining,
            temperature=1,
            top_p=1,
            vocab_size=dataset.tokenizer.get_vocab_size(),
            return_behavior=True,
        )
        actions = result[:, ids.shape[1] :]
        labels = torch.cat((labels, actions), 1)
        behavior_logp = torch.cat((behavior_logp, sampled_logp), 1)
        ids = result
        generated += actions.numel()
        values = actions[0].tolist()
        if environment is None or 18 not in values:
            break
        start = values.index(18) + 1
        payload = dataset.tokenizer.decode(
            [v for v in values[start:] if v != 2], skip_special_tokens=True
        )
        observation = environment.execute(payload)
        trace.append(dict(action=payload, result=observation))
        encoded = [6, 19, *safe_text(dataset.tokenizer, json.dumps(observation)), 2, 5]
        if len(encoded) + ids.shape[1] >= model.config.max_position_embeddings:
            break
        ids = torch.cat((ids, torch.tensor([encoded], device=device)), 1)
        labels = torch.cat(
            (labels, torch.full((1, len(encoded)), -100, device=device, dtype=torch.long)), 1
        )
        behavior_logp = torch.cat((behavior_logp, torch.zeros((1, len(encoded)), device=device)), 1)
    answer = (
        dataset.tokenizer.decode(actions[0].tolist(), skip_special_tokens=True) if generated else ""
    )
    reward, verification = verify_answer(record, answer, trace)
    return dict(
        input_ids=ids,
        labels=labels,
        media=media,
        generated_tokens=generated,
        reward=reward,
        verification=verification,
        tool_trace=trace,
        sample_id=record["sample_id"],
        text=answer,
        media_hashes=prepared["media_hashes"],
        behavior_logp=behavior_logp,
    )


def response_logp(model, trajectory, vocab):
    out = model(trajectory["input_ids"], media=trajectory["media"])
    return token_log_probs(
        out.logits,
        trajectory["labels"],
        actions=True,
        vocab_size=vocab,
        forbidden_ids=forbidden_actions(model),
    )


def grpo_objective(
    model,
    reference,
    trajectories,
    *,
    vocab_size,
    clip=0.2,
    kl_beta=0.01,
    backward=False,
    normalization_denominator=None,
):
    rewards = torch.tensor(
        [r["reward"] for r in trajectories], device=next(model.parameters()).device
    )
    advantages = grouped_advantages(rewards, len(trajectories))
    denominator = sum(int(t["labels"][:, 1:].ne(-100).sum()) for t in trajectories)
    # Behavior probabilities are measured before any update, with identical precision/actions.
    with torch.no_grad():
        old = [
            t["behavior_logp"].detach()
            if "behavior_logp" in t
            else response_logp(model, t, vocab_size)[0].detach()
            for t in trajectories
        ]
        refs = [response_logp(reference, t, vocab_size)[0].detach() for t in trajectories]
    terms = []
    for i, trajectory in enumerate(trajectories):
        logp, mask = response_logp(model, trajectory, vocab_size)
        ratio = (logp - old[i]).exp()
        policy = -torch.minimum(
            ratio * advantages[i], ratio.clamp(1 - clip, 1 + clip) * advantages[i]
        )
        delta = refs[i] - logp
        term = ((policy + kl_beta * (delta.exp() - delta - 1)) * mask).sum() / max(
            1, normalization_denominator if normalization_denominator is not None else denominator
        )
        if backward:
            term.backward()
        else:
            terms.append(term)
    return terms, dict(
        zero_variance=bool(rewards.std(unbiased=False) == 0),
        rewards=rewards.tolist(),
        response_positions=denominator,
        reduction="sum over valid assistant tokens / window assistant tokens",
    )


def rl_window(model, reference, dataset, records, *, group_size, max_tokens):
    """One fresh policy window, sequential graphs, normalization over all response tokens."""
    model.eval()
    groups = [
        [rollout(model, dataset, record, max_tokens=max_tokens) for _ in range(group_size)]
        for record in records
    ]
    traces = [trace for group in groups for trace in group]
    denominator = sum(int(t["labels"][:, 1:].ne(-100).sum()) for t in traces)
    model.train()
    reports = []
    for group in groups:
        _, report = grpo_objective(
            model,
            reference,
            group,
            vocab_size=dataset.tokenizer.get_vocab_size(),
            backward=True,
            normalization_denominator=denominator,
        )
        reports.append(report)
    zero = sum(r["zero_variance"] for r in reports)
    return traces, dict(
        zero_variance=zero == len(groups),
        zero_variance_groups=zero,
        prompts=len(groups),
        groups=reports,
        response_positions=denominator,
        reduction="sum over all window assistant tokens / all window assistant tokens",
    )


def opd_objective(student, teacher, trajectory, *, vocab_size, chunk_size=64):
    with torch.no_grad():
        target = teacher(
            trajectory["input_ids"],
            media=trajectory["media"],
            return_hidden=True,
            return_logits=False,
        )
    output = student(
        trajectory["input_ids"], media=trajectory["media"], return_hidden=True, return_logits=False
    )
    mask = trajectory["labels"][:, 1:].ne(-100)
    return full_vocab_reverse_kl(
        output.hidden_states[:, :-1],
        target.hidden_states[:, :-1],
        student.lm_head,
        teacher.lm_head,
        mask,
        chunk_size=chunk_size,
        vocab_size=vocab_size,
        forbidden_ids=forbidden_actions(student),
    )


def qualified_teacher(registry_path, slot, student_saved, *, diagnostic=False):
    registry = json.loads(Path(registry_path).read_text())
    if slot not in TEACHER_SLOTS or slot not in registry.get("teachers", {}):
        raise ValueError("teacher registry has no matching domain/effort slot")
    entry = registry["teachers"][slot]
    checkpoint = Path(registry_path).parent / entry["checkpoint"]
    if sha256(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("teacher checkpoint differs from qualified artifact")
    if entry.get("status") != "qualified" and not diagnostic:
        raise ValueError("unqualified teacher cannot participate in formal OPD")
    if not diagnostic:
        evaluation = Path(registry_path).parent / entry["evaluation"]
        if sha256(evaluation) != entry["evaluation_sha256"]:
            raise ValueError("teacher qualification evaluation changed")
        value = json.loads(evaluation.read_text())
        if (
            value.get("checkpoint_sha256") != entry["checkpoint_sha256"]
            or value.get("slot") != slot
            or value.get("tokenizer_sha256") != student_saved["tokenizer_sha256"]
            or value.get("metrics", {}).get("qualified") is not True
        ):
            raise ValueError("teacher evaluation does not qualify the actual checkpoint")
    teacher_saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        teacher_saved["tokenizer_sha256"] != student_saved["tokenizer_sha256"]
        or teacher_saved["config"] != student_saved["config"]
    ):
        raise ValueError("OPD needs exactly the same vocabulary mapping and architecture")
    for key in ("processor_sha256", "template_sha256"):
        if teacher_saved["run_spec"].get(key) != student_saved["run_spec"].get(key):
            raise ValueError(f"teacher/student {key} mismatch")
    return checkpoint, entry


def train_post(
    *,
    phase,
    checkpoint,
    data,
    output,
    device="cpu",
    steps=None,
    max_tokens=None,
    group_size=4,
    teacher_registry=None,
    teacher_slot=None,
    run_kind="acceptance",
    evidence=None,
    resume=None,
    seed=42,
    token_budget=None,
    prompts_per_update=None,
):
    if phase not in {"rl", "teacher", "opd", "draft", "dpo"}:
        raise ValueError("posttrain phase must be rl, teacher, opd, draft or dpo")
    steps = (
        steps if steps is not None else (2 if run_kind == "acceptance" else PHASES[phase]["budget"])
    )
    max_tokens = max_tokens if max_tokens is not None else (32 if run_kind == "acceptance" else 256)
    if (
        steps < 1
        or (run_kind == "acceptance" and steps > 10000)
        or not 1 <= max_tokens <= 1024
        or not 2 <= group_size <= 8
    ):
        raise ValueError("invalid posttraining update/rollout budget")
    if prompts_per_update is None:
        prompts_per_update = 32 if run_kind == "strategy" and phase in {"rl", "teacher"} else 1
    if not 1 <= prompts_per_update <= 64 or (
        phase not in {"rl", "teacher"} and prompts_per_update != 1
    ):
        raise ValueError("prompt accumulation is supported for RL/teacher windows only")
    device, output = torch.device(device), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and not resume:
        raise FileExistsError("posttraining run exists; resume its exact state")
    torch.manual_seed(seed)
    random.seed(seed)
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    if type(model) is not MiniFrontier1ForCausalLM:
        raise ValueError("posttraining requires an MF1 checkpoint")
    base = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dataset = RecordDataset(data, "train", model.config)
    if base["tokenizer_sha256"] != dataset.manifest["tokenizer_sha256"]:
        raise ValueError("posttraining dataset/tokenizer mismatch")
    bound = bindings(model.config, data, checkpoint)
    if run_kind == "strategy":
        if token_budget != PHASES[phase]["budget"] or evidence is None:
            raise ValueError("formal posttraining needs its planned token budget and stage gate")
        validate_gate(phase, json.loads(Path(evidence).read_text()), bound, dataset.manifest, base)
    elif run_kind != "acceptance" or (token_budget is not None and token_budget > 2_000_000):
        raise ValueError("invalid diagnostic posttraining budget")
    if phase == "teacher" and teacher_slot not in TEACHER_SLOTS:
        raise ValueError("teacher run must identify its actual domain/effort slot")
    from .teachers import slot_for

    eligible = [
        i
        for i in range(len(dataset))
        if phase != "teacher" or slot_for(dataset.record(i)) == teacher_slot
    ]
    if not eligible:
        raise ValueError("no training records match the requested teacher domain/effort")
    if phase == "dpo" and run_kind == "strategy":
        steps = min(steps, len(eligible))  # The selected DPO recipe permits at most one epoch.
    target = model
    learner: Any
    reference = (
        copy.deepcopy(model).eval().requires_grad_(False)
        if phase in {"rl", "teacher", "dpo"}
        else None
    )
    teacher_entry = None
    teacher_paths = {}
    active_teacher_slot = None
    if phase == "opd":
        if teacher_registry is None:
            raise ValueError("OPD requires a hash-bound qualified teacher registry and slot")
        if teacher_slot in {None, "auto"}:
            registry = json.loads(Path(teacher_registry).read_text())
            teacher_entry = {}
            for slot, entry in registry.get("teachers", {}).items():
                if entry.get("status") == "qualified":
                    path, admitted = qualified_teacher(teacher_registry, slot, base)
                    teacher_paths[slot] = path
                    teacher_entry[slot] = admitted
            if not teacher_paths:
                raise ValueError("automatic multi-teacher OPD has no qualified teachers")
            eligible = [i for i in eligible if slot_for(dataset.record(i)) in teacher_paths]
            if not eligible:
                raise ValueError("no prompts match the qualified teacher slots")
        else:
            path, teacher_entry = qualified_teacher(
                teacher_registry, teacher_slot, base, diagnostic=run_kind == "acceptance"
            )
            reference, _, _ = load_checkpoint(path, device)
            reference.eval().requires_grad_(False)
    if phase == "draft":
        learner = MF1Draft(target)
        opt = torch.optim.AdamW(learner.parameters(), lr=1e-4, betas=(0.9, 0.95))
    else:
        learner = model
        model.requires_grad_(True)
        for name, p in model.named_parameters():
            if (
                ".indexer." in name
                or name.startswith("mtp.")
                or (
                    phase == "teacher"
                    and teacher_slot.split(":")[0] != "vision"
                    and name.startswith("vision.")
                )
            ):
                p.requires_grad_(False)
        opt = make_optimizer(
            model, lr=3e-6 if phase == "opd" else 1e-6, scalar_lr=1e-6, vision_lr=1e-6
        )
    run = dict(
        format="mf1-posttrain-v1",
        phase=phase,
        **bound,
        steps=steps,
        max_tokens=max_tokens,
        group_size=group_size,
        prompts_per_update=prompts_per_update,
        seed=seed,
        kind=run_kind,
        token_budget=token_budget,
        teacher_registry_sha256=sha256(teacher_registry) if teacher_registry else None,
        teacher_slot=teacher_slot,
        teacher=teacher_entry,
        mtp="frozen" if phase != "draft" else "draft_only",
    )
    ledger = dict(
        generated_tokens=0, response_positions=0, optimizer_updates=0, zero_variance_groups=0
    )
    start = 0
    if resume:
        saved = torch.load(resume, weights_only=True, map_location="cpu")
        if saved["posttrain_spec"] != run:
            raise ValueError(
                "exact posttraining resume requires identical target/data/teacher/source/budget"
            )
        learner.load_state_dict(saved["draft"] if phase == "draft" else saved["model"])
        opt.load_state_dict(saved["optimizer"])
        ledger, start = saved["ledger"], saved["step"]
        restore_rng(saved["rng"], device)
    else:
        shutil.copyfile(Path(checkpoint).parent / "tokenizer.json", output / "tokenizer.json")
        write_json(output / "run.json", run)
    for step in range(start, steps):
        if (
            token_budget is not None
            and ledger["generated_tokens" if phase != "dpo" else "response_positions"]
            >= token_budget
        ):
            break
        record = dataset.record(eligible[(step * prompts_per_update) % len(eligible)])
        if teacher_paths:
            slot = slot_for(record)
            if slot != active_teacher_slot:
                reference = None  # Release the previous teacher before loading the next one.
                reference, _, _ = load_checkpoint(teacher_paths[slot], device)
                reference.eval().requires_grad_(False)
                active_teacher_slot = slot
        opt.zero_grad(set_to_none=True)
        model.eval()
        traces = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if phase == "draft":
                prepared = encode_record(
                    record, tokenizer, model.config, dataset.media_root, generation_prompt=True
                )
                loss, metrics = learner.unroll(
                    prepared["input_ids"].to(device),
                    media=move(prepared["media"], device),
                    steps=min(6, max(2, max_tokens)),
                    vocab_size=tokenizer.get_vocab_size(),
                )
                ledger["generated_tokens"] += metrics["positions"]
                loss.backward()
            elif phase == "dpo":
                if "rejected_messages" not in record or not record.get("preference_verified"):
                    raise ValueError(
                        "DPO requires verified chosen/rejected messages on the same prompt/media"
                    )
                chosen = encode_record(record, tokenizer, model.config, dataset.media_root)
                rejected_record = dict(record, messages=record["rejected_messages"])
                if record["messages"][:-1] != rejected_record["messages"][:-1]:
                    raise ValueError("preference pair prompts differ")
                rejected = encode_record(
                    rejected_record, tokenizer, model.config, dataset.media_root
                )
                width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
                policy_logits, reference_logits, ys = [], [], []
                for item in (chosen, rejected):
                    item = move(item, device)
                    policy_logits.append(
                        torch.nn.functional.pad(
                            model(item["input_ids"], media=item["media"]).logits,
                            (0, 0, 0, width - item["input_ids"].shape[1]),
                        )
                    )
                    with torch.no_grad():
                        reference_logits.append(
                            torch.nn.functional.pad(
                                cast(Any, reference)(item["input_ids"], media=item["media"]).logits,
                                (0, 0, 0, width - item["input_ids"].shape[1]),
                            )
                        )
                    ys.append(
                        torch.nn.functional.pad(
                            item["labels"], (0, width - item["labels"].shape[1]), value=-100
                        )
                    )
                loss, accuracy = dpo_loss(
                    torch.cat(policy_logits), torch.cat(reference_logits), torch.cat(ys)
                )
                metrics = dict(preference_accuracy=float(accuracy))
                ledger["response_positions"] += sum(int(y[:, 1:].ne(-100).sum()) for y in ys)
                loss.backward()
            elif phase in {"rl", "teacher"}:
                records = [
                    dataset.record(eligible[(step * prompts_per_update + offset) % len(eligible)])
                    for offset in range(prompts_per_update)
                ]
                traces, metrics = rl_window(
                    model,
                    reference,
                    dataset,
                    records,
                    group_size=group_size,
                    max_tokens=max_tokens,
                )
                ledger["zero_variance_groups"] += metrics["zero_variance_groups"]
                ledger["generated_tokens"] += sum(t["generated_tokens"] for t in traces)
                ledger["response_positions"] += metrics["response_positions"]
            else:
                traces = [rollout(model, dataset, record, max_tokens=max_tokens)]
                model.train()
                loss = opd_objective(
                    model, reference, traces[0], vocab_size=tokenizer.get_vocab_size()
                )
                metrics = dict(
                    full_vocabulary_reverse_kl=float(loss.detach()),
                    teacher_slot=active_teacher_slot or teacher_slot,
                )
                loss.backward()
                ledger["generated_tokens"] += sum(t["generated_tokens"] for t in traces)
                ledger["response_positions"] += sum(
                    int(t["labels"][:, 1:].ne(-100).sum()) for t in traces
                )
        # A zero-variance group has no policy learning signal; do not perform a fake update.
        updated = not metrics.get("zero_variance", False)
        if updated:
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            ledger["optimizer_updates"] += 1
        with (output / "rollouts.jsonl").open("a") as handle:
            for trajectory in traces:
                handle.write(
                    json.dumps(
                        dict(
                            step=step + 1,
                            policy_version=ledger["optimizer_updates"] - int(updated),
                            checkpoint_series=bound["actual_init_checkpoint_sha256"],
                            input_ids=trajectory["input_ids"].tolist(),
                            labels=trajectory["labels"].tolist(),
                            sample_id=trajectory["sample_id"],
                            media_hashes=trajectory["media_hashes"],
                            reward=trajectory["reward"],
                            verification=trajectory["verification"],
                            tool_trace=trajectory["tool_trace"],
                            generated_tokens=trajectory["generated_tokens"],
                            behavior_logp=trajectory["behavior_logp"].tolist(),
                        )
                    )
                    + "\n"
                )
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(step=step + 1, updated=updated) | metrics | ledger) + "\n")
        artifact = dict(
            base,
            mf1_phase=phase,
            stage=phase,
            step=step + 1,
            posttrain_spec=run,
            model=target.state_dict(),
            optimizer=opt.state_dict(),
            rng=rng_state(device),
            ledger=ledger,
            capability_qualified=False,
        )
        if phase == "draft":
            artifact.update(
                draft=learner.state_dict(),
                target_sha256=sha256(checkpoint),
                draft_rule="fixed-anchor-v1",
            )
        atomic_save(artifact, output / "checkpoint.pt")
        write_json(
            output / "status.json",
            dict(
                state="budget_complete_unqualified"
                if (token_budget is not None and ledger[PHASES[phase]["unit"]] >= token_budget)
                else "updates_complete_unqualified"
                if step + 1 == steps
                else "running",
                phase=phase,
                step=step + 1,
                ledger=ledger,
            ),
        )
    return dict(
        phase=phase,
        ledger=ledger,
        checkpoint=str(output / "checkpoint.pt"),
        capability_qualified=False,
    )
