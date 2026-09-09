"""On-policy text, native media and bounded local tool trajectories."""

from __future__ import annotations

import copy
import json
import random
import re
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from minifrontier.data import chat_tokens, fingerprint, split_for
from minifrontier.inference.runtime import generate_ids
from minifrontier.training.distributions import forbidden_actions
from minifrontier.training.posttrain import (
    grouped_advantages,
    mopd_advantages,
    policy_loss,
    token_log_probs,
)
from minifrontier.training.teachers import TeacherRegistry


def prepare_tasks(output, count=10000, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new directory for generated RL tasks")
    if not 100 <= count <= 200000:
        raise ValueError("task count must be in [100, 200000]")
    output.mkdir(parents=True)
    rng = random.Random(seed)
    splits: dict[str, list] = {"train": [], "val": []}
    seen = set()
    while sum(map(len, splits.values())) < count:
        a, b = rng.randrange(1000), rng.randrange(1000)
        op = rng.choice(["+", "-"])
        key = f"{a}{op}{b}"
        if key in seen:
            continue
        seen.add(key)
        answer = a + b if op == "+" else a - b
        effort = rng.choice(["low", "high"])
        task = dict(
            prompt=f"Calculate {a} {op} {b}. Reply with the integer answer only.",
            answer=str(answer),
            domain="arithmetic",
            effort=effort,
        )
        splits[split_for(fingerprint(key), seed)].append(task)
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                generator="integer-add-sub-v1",
                seed=seed,
                provenance="locally generated, exact arithmetic reward",
                counts={k: len(v) for k, v in splits.items()},
            ),
            indent=2,
        )
    )


class TaskDataset:
    def __init__(
        self,
        root,
        split,
        tokenizer,
        prompt_length,
        *,
        family=None,
        model_vocab_size=None,
        max_features=64,
    ):
        self.root, self.family, self.model_vocab_size = (
            Path(root).resolve(),
            family,
            model_vocab_size,
        )
        self.max_features = max_features
        self._native_cache = {}
        self.rows = [
            json.loads(line) for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()
        ]
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        self.chat_template = manifest.get("chat_template", "legacy")
        if self.chat_template not in {"legacy", "control-v1"}:
            raise ValueError("unknown RL chat template")
        if not self.rows:
            raise ValueError("empty RL split")
        for row in self.rows:
            if row.get("environment"):
                from minifrontier.training.tool_environment import ToolEnvironment

                if self.chat_template != "control-v1":
                    raise ValueError("tool tasks require the explicit control-v1 template")
                ToolEnvironment(row["environment"])
            if row.get("media") and family is None:
                raise ValueError(
                    "multimodal RL needs the native rollout adapter; media cannot be ignored"
                )
            if any(not resource.get("rgb_sha256") for resource in row.get("media", [])):
                raise ValueError("native RL media needs immutable decoded RGB hashes")
            if "verifier" not in row and row.get("domain") != "arithmetic":
                raise ValueError("non-arithmetic tasks require an explicit verifier")
            if not row.get("media"):
                row["input_ids"] = self._text_prompt(row)
                if len(row["input_ids"]) > prompt_length:
                    raise ValueError("RL prompt exceeds context budget; increase sequence length")

    def _record(self, row):
        structured = self.chat_template == "control-v1"
        prompt = row.get("prompt", "")
        if not structured:
            prompt = f"Reasoning effort: {row['effort']}. " + prompt
        turns = row.get("turns", [{"role": "user", "content": prompt}])
        if row.get("environment") and not any(turn.get("tools") for turn in turns):
            from minifrontier.training.tool_environment import definitions

            turns = copy.deepcopy(turns)
            context = next((turn for turn in turns if turn["role"] in {"system", "user"}), None)
            if context is None:
                raise ValueError("tool task needs user/system context for its tool definitions")
            context["tools"] = definitions(row["environment"]["tools"])
        if not turns or turns[-1]["role"] == "assistant":
            raise ValueError("RL task must end before the assistant answer")
        return dict(
            stage="sft",
            turns=turns,
            media=row.get("media", []),
            mode=row.get("mode") if structured else None,
            effort=row["effort"] if structured else None,
        )

    def _text_prompt(self, row):
        record = self._record(row)
        return chat_tokens(
            record["turns"],
            self.tokenizer,
            generation_prompt=True,
            mode=record["mode"],
            effort=record["effort"],
        )[0]

    def native_prompt(self, index):
        from minifrontier.multimodal import prepare_record

        cached = self._native_cache.pop(index, None)
        if cached is not None:
            self._native_cache[index] = cached
            return cached
        prepared = prepare_record(
            self._record(self.rows[index]),
            self.tokenizer,
            self.family,
            root=self.root,
            max_features=self.max_features,
            generation_prompt=True,
            model_vocab_size=self.model_vocab_size,
        )
        if prepared.input_ids.shape[1] > self.prompt_length:
            raise ValueError(
                "native RL prompt exceeds complete text/media budget; no image is dropped"
            )
        self._native_cache[index] = prepared
        if len(self._native_cache) > 8:
            self._native_cache.pop(next(iter(self._native_cache)))
        return prepared

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids = (
            self.native_prompt(index).input_ids[0].tolist()
            if self.rows[index].get("media")
            else self.rows[index]["input_ids"]
        )
        # Second tensor carries the task index, rather than supervised labels.
        return torch.tensor(ids + [0] * (self.prompt_length - len(ids))), torch.tensor([index])


def exact_reward(text, answer):
    match = re.fullmatch(r"\s*([-+]?\d+)\s*[.!。]?\s*", text)
    return float(match is not None and int(match[1]) == int(answer))


def preserve_policy_mode(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        training = self.model.training
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.train(training)

    return call


class RolloutObjective:
    def __init__(
        self,
        model,
        reference,
        tokenizer,
        dataset,
        *,
        group_size=4,
        max_new_tokens=32,
        teacher_map=None,
        device="cpu",
        method=None,
        require_complete_teachers=False,
    ):
        self.model, self.reference, self.tokenizer, self.dataset = (
            model,
            reference,
            tokenizer,
            dataset,
        )
        self.group_size, self.max_new_tokens = group_size, max_new_tokens
        self.method = method or ("mopd" if teacher_map else "grpo")
        if self.method == "opd" and model.__class__.__name__ != "MiniDeepSeekV4ForCausalLM":
            raise ValueError("full-vocabulary OPD is the DeepSeek route")
        self.teachers: Any = {}
        if teacher_map:
            self.teachers = TeacherRegistry(
                teacher_map, tokenizer, device, require_complete=require_complete_teachers
            )
            for row in dataset.rows:
                if f"{row['domain']}:{row['effort']}" not in self.teachers:
                    raise ValueError("teacher map must cover every task domain/effort pair")
        self.last_reward = 0.0
        self.last_tokens = 0

    @preserve_policy_mode
    def prepare(self, x, task_indices):
        from minifrontier.multimodal import move, select_media

        training = self.model.training
        sequences: list[list[int]] = []
        label_rows, rewards, teacher_keys, behavior_rows = [], [], [], []
        all_media: list[dict] = []
        terminations = []
        traces = []
        opd_targets = []
        self.model.eval()
        with torch.no_grad():
            for prompt, idx in zip(x, task_indices.flatten(), strict=True):
                row = self.dataset.rows[int(idx)]
                prompt = prompt[prompt != 0][None].expand(self.group_size, -1)
                media = []
                if row.get("media"):
                    native = self.dataset.native_prompt(int(idx))
                    if not torch.equal(native.input_ids[0], prompt[0].cpu()):
                        raise ValueError("rollout prompt changed after native preprocessing")
                    media = move(
                        select_media(native.extras["media"], [0] * self.group_size), x.device
                    )
                if row.get("environment"):
                    from minifrontier.training.tool_rollouts import sample_tool_trajectory
                    from minifrontier.training.verifiers import reward

                    for sample_index in range(self.group_size):
                        sampled = sample_tool_trajectory(
                            self.model,
                            prompt[sample_index : sample_index + 1],
                            self.tokenizer,
                            row,
                            select_media(media, [sample_index]),
                            max_actions=self.max_new_tokens,
                            context_limit=self.dataset.prompt_length + self.max_new_tokens,
                        )
                        all_media.extend(
                            dict(span, batch_index=len(sequences))
                            for span in media
                            if span["batch_index"] == sample_index
                        )
                        score = (
                            reward(
                                sampled["response"],
                                row["verifier"],
                                environment=sampled["environment"],
                            )
                            if sampled["termination"] == "final"
                            else 0.0
                        )
                        sequences.append(sampled["ids"])
                        label_rows.append(sampled["labels"])
                        behavior_rows.append(sampled["behavior"])
                        rewards.append(score)
                        terminations.append(sampled["termination"])
                        teacher_keys.append(f"{row['domain']}:{row['effort']}")
                        traces.append(
                            dict(
                                task_index=int(idx),
                                prompt_ids=prompt[0].tolist(),
                                turns=sampled["trace"],
                                tool_calls=sampled["environment"].calls,
                                tool_errors=sampled["environment"].errors,
                            )
                        )
                    continue
                generated, behavior, action_mask = generate_ids(
                    self.model,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    vocab_size=self.tokenizer.get_vocab_size(),
                    temperature=1.0,
                    top_p=1.0,
                    return_behavior=True,
                    media=media,
                )
                for sample_index, seq in enumerate(generated):
                    all_media.extend(
                        dict(span, batch_index=len(sequences))
                        for span in media
                        if span["batch_index"] == sample_index
                    )
                    ids = seq.tolist()
                    end = len(ids)
                    for i in range(prompt.shape[1], end):
                        if ids[i] == 2:
                            end = i + 1
                            break
                    ids = ids[:end]
                    labels = [-100] * prompt.shape[1] + ids[prompt.shape[1] :]
                    sequences.append(ids)
                    label_rows.append(labels)
                    traces.append(dict(task_index=int(idx), prompt_ids=prompt[0].tolist()))
                    behavior_rows.append(
                        [0.0] * (prompt.shape[1] - 1)
                        + behavior[sample_index, action_mask[sample_index]].tolist()
                    )
                    response = self.tokenizer.decode(
                        ids[prompt.shape[1] :], skip_special_tokens=True
                    )
                    terminations.append("eos" if ids[-1] == 2 else "length")
                    parsed = None
                    if self.dataset.chat_template == "control-v1":
                        from minifrontier.chat_controls import parse_action

                        parsed = parse_action(
                            ids[prompt.shape[1] :], self.tokenizer, require_eos=True
                        )
                        terminations[-1] = parsed["kind"]
                        response = parsed["text"]
                    if parsed is not None and parsed["kind"] != "final":
                        rewards.append(0.0)
                        teacher_keys.append(f"{row['domain']}:{row['effort']}")
                        continue
                    if row.get("verifier"):
                        from minifrontier.training.verifiers import reward

                        rewards.append(reward(response, row["verifier"]))
                    else:
                        rewards.append(exact_reward(response, row["answer"]))
                    teacher_keys.append(f"{row['domain']}:{row['effort']}")
            length = max(map(len, sequences))
            inputs = torch.tensor([s + [0] * (length - len(s)) for s in sequences], device=x.device)
            labels = torch.tensor(
                [s + [-100] * (length - len(s)) for s in label_rows], device=x.device
            )
            old_logp = torch.tensor(
                [row + [0.0] * (length - 1 - len(row)) for row in behavior_rows], device=x.device
            )
            recomputed_logp, mask = token_log_probs(
                self.model(
                    inputs,
                    attention_mask=inputs.ne(0),
                    **(dict(media=all_media) if all_media else {}),
                ).logits,
                labels,
                actions=True,
                forbidden_ids=forbidden_actions(self.model),
                vocab_size=self.tokenizer.get_vocab_size(),
            )
            ratio_error = float(((recomputed_logp - old_logp).exp() - 1)[mask].abs().max())
            # Explicit local numeric bounds, well below PPO's 0.2 clipping band.
            # CUDA KDA chunk/recurrent kernels have different accumulation order.
            low_precision = torch.is_autocast_enabled(x.device.type) or next(
                self.model.parameters()
            ).dtype in {torch.bfloat16, torch.float16}
            ratio_tolerance = 0.02 if low_precision else 0.001 if x.is_cuda else 2e-5
            if not ratio_error <= ratio_tolerance:
                raise FloatingPointError(
                    f"frozen rollout ratio error {ratio_error:.6g} exceeds {ratio_tolerance}; "
                    "check precision/cache/attention support before RL"
                )
            if self.method == "opd":
                for key in sorted(set(teacher_keys)):
                    indices = torch.tensor(
                        [i for i, item in enumerate(teacher_keys) if item == key], device=x.device
                    )
                    teacher = self.teachers[key]
                    self.check_teacher_template(teacher)
                    self.check_media_teacher(teacher, all_media)
                    if teacher.config.qat_scheme != self.model.config.qat_scheme:
                        raise ValueError("OPD teacher and student QAT schemes must match")
                    features = teacher(
                        inputs[indices],
                        attention_mask=inputs[indices].ne(0),
                        return_logits=False,
                        return_hidden=True,
                        **(dict(media=select_media(all_media, indices)) if all_media else {}),
                    ).hidden_states
                    opd_targets.append(
                        dict(
                            indices=indices.cpu(),
                            hidden=features.detach().cpu(),
                            weight=teacher.head.weight.detach().cpu().clone(),
                        )
                    )
                    del features, teacher
                self.teachers.unload()
                advantages, ref_logp = torch.zeros_like(old_logp), None
            elif self.teachers:
                teacher_logp = torch.empty_like(old_logp)
                for key in sorted(set(teacher_keys)):
                    indices = torch.tensor(
                        [i for i, item in enumerate(teacher_keys) if item == key], device=x.device
                    )
                    teacher = self.teachers[key]
                    self.check_teacher_template(teacher)
                    self.check_media_teacher(teacher, all_media)
                    if getattr(teacher.config, "qat_scheme", "bf16") != getattr(
                        self.model.config, "qat_scheme", "bf16"
                    ):
                        raise ValueError("MOPD teacher and student QAT schemes must match")
                    teacher_logp[indices] = token_log_probs(
                        teacher(
                            inputs[indices],
                            attention_mask=inputs[indices].ne(0),
                            **(dict(media=select_media(all_media, indices)) if all_media else {}),
                        ).logits,
                        labels[indices],
                        actions=True,
                        forbidden_ids=forbidden_actions(self.model),
                        vocab_size=self.tokenizer.get_vocab_size(),
                    )[0]
                    del teacher
                advantages = mopd_advantages(teacher_logp, old_logp)
                ref_logp = None
                self.teachers.unload()
            else:
                advantages = grouped_advantages(
                    torch.tensor(rewards, device=x.device), self.group_size
                )
                ref_logp = token_log_probs(
                    self.reference(
                        inputs,
                        attention_mask=inputs.ne(0),
                        **(dict(media=all_media) if all_media else {}),
                    ).logits,
                    labels,
                    actions=True,
                    forbidden_ids=forbidden_actions(self.model),
                    vocab_size=self.tokenizer.get_vocab_size(),
                )[0]
        loss_mask = mask if self.teachers else mask & advantages.ne(0)[:, None]
        self.model.train(training)
        return dict(
            inputs=inputs,
            labels=labels,
            old_logp=old_logp,
            advantages=advantages,
            reference_logp=ref_logp,
            rewards=rewards,
            loss_mask=loss_mask,
            active_responses=int(loss_mask.sum())
            if self.method == "opd"
            else int(loss_mask.any(-1).sum()),
            opd_targets=opd_targets,
            media=all_media,
            media_counts=[
                sum(s.get("resource_kind", "image") == "image" for s in all_media),
                sum(s.get("resource_kind") == "video" for s in all_media),
                sum(s.get("source_frame_count", 0) for s in all_media),
                sum(s["feature_count"] for s in all_media),
            ],
            input_positions=int(inputs.ne(0).sum()),
            terminations=terminations,
            traces=traces,
            behavior_ratio_error=ratio_error,
            behavior_ratio_tolerance=ratio_tolerance,
        )

    def check_media_teacher(self, teacher, media):
        if not media:
            return
        from dataclasses import asdict

        if type(teacher) is not type(self.model) or getattr(teacher, "vision", None) is None:
            raise ValueError(
                "media teacher must have the same native model family and vision tower"
            )
        if asdict(teacher.config.vision_config) != asdict(self.model.config.vision_config):
            raise ValueError("media teacher and student native processors/configurations differ")

    def check_teacher_template(self, teacher):
        if self.dataset.chat_template != getattr(teacher, "chat_template", "legacy"):
            raise ValueError("teacher and student trajectories must share the same chat template")

    def __call__(self, wrapped, x, task_indices, *, prepared=None):
        prepared = self.prepare(x, task_indices) if prepared is None else prepared
        inputs, labels = prepared["inputs"], prepared["labels"]
        old_logp, advantages = prepared["old_logp"], prepared["advantages"]
        ref_logp, rewards = prepared["reference_logp"], prepared["rewards"]
        training = self.model.training
        self.model.eval()
        if self.method == "opd":
            try:
                mask = prepared["loss_mask"]
                loss = wrapped(
                    inputs,
                    attention_mask=inputs.ne(0),
                    return_logits=False,
                    opd_targets=prepared["opd_targets"],
                    opd_mask=mask,
                    opd_vocab_size=self.tokenizer.get_vocab_size(),
                    **(dict(media=prepared["media"]) if prepared.get("media") else {}),
                ).loss
            finally:
                self.model.train(training)
            self.last_reward = sum(rewards) / len(rewards)
            self.last_tokens = self.last_trainable_tokens = self.last_active_responses = int(
                mask.sum()
            )
            self.last_ratio_error = prepared["behavior_ratio_error"]
            return loss, loss.detach(), loss.new_tensor(self.last_reward)
        # Policy gradients are computed in eval mode too: dropout must not change
        # the probability model that sampled these actions. Autograd remains on.
        try:
            logp, mask = token_log_probs(
                wrapped(
                    inputs,
                    attention_mask=inputs.ne(0),
                    **(dict(media=prepared["media"]) if prepared.get("media") else {}),
                ).logits,
                labels,
                actions=True,
                forbidden_ids=forbidden_actions(self.model),
                vocab_size=self.tokenizer.get_vocab_size(),
            )
        finally:
            self.model.train(training)
        self.last_ratio_error = float(((logp.detach() - old_logp).exp() - 1)[mask].abs().max())
        if not self.teachers:
            active = advantages.ne(0)
            self.last_zero_variance_fraction = float((~active).float().mean())
            mask = mask & active[:, None]
        loss = policy_loss(logp, old_logp, advantages, mask, reference_logp=ref_logp)
        self.last_reward = sum(rewards) / len(rewards)
        self.last_tokens = int(labels[:, 1:].ne(-100).sum())
        self.last_trainable_tokens = int(mask.sum())
        self.last_active_responses = int(mask.any(-1).sum())
        return loss, loss.detach(), loss.new_tensor(self.last_reward)
