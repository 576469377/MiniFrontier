"""On-policy arithmetic RL and domain/effort teacher selection for educational runs.

Tasks are generated locally with exact numeric answers. No external model API,
network call or execution of generated code is used for rewards.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch

from minifrontier.data import chat_tokens, fingerprint, split_for
from minifrontier.inference import generate_ids
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
    def __init__(self, root, split, tokenizer, prompt_length):
        self.rows = [
            json.loads(line) for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()
        ]
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        if not self.rows:
            raise ValueError("empty RL split")
        for row in self.rows:
            prompt = f"Reasoning effort: {row['effort']}. " + row["prompt"]
            ids, _ = chat_tokens(
                [{"role": "user", "content": prompt}], tokenizer, generation_prompt=True
            )
            if len(ids) > prompt_length:
                raise ValueError("RL prompt exceeds context budget; increase sequence length")
            row["input_ids"] = ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids = self.rows[index]["input_ids"]
        # Second tensor carries the task index, rather than supervised labels.
        return torch.tensor(ids + [0] * (self.prompt_length - len(ids))), torch.tensor([index])


def exact_reward(text, answer):
    match = re.fullmatch(r"\s*([-+]?\d+)\s*[.!。]?\s*", text)
    return float(match is not None and int(match[1]) == int(answer))


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
    ):
        self.model, self.reference, self.tokenizer, self.dataset = (
            model,
            reference,
            tokenizer,
            dataset,
        )
        self.group_size, self.max_new_tokens = group_size, max_new_tokens
        self.teachers: Any = {}
        if teacher_map:
            self.teachers = TeacherRegistry(teacher_map, tokenizer, device)
            for row in dataset.rows:
                if f"{row['domain']}:{row['effort']}" not in self.teachers:
                    raise ValueError("teacher map must cover every task domain/effort pair")
        self.last_reward = 0.0
        self.last_tokens = 0

    def __call__(self, wrapped, x, task_indices):
        training = self.model.training
        sequences, label_rows, rewards, teacher_keys, behavior_rows = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for prompt, idx in zip(x, task_indices.flatten(), strict=True):
                row = self.dataset.rows[int(idx)]
                prompt = prompt[prompt != 0][None].expand(self.group_size, -1)
                generated, behavior, action_mask = generate_ids(
                    self.model,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    vocab_size=self.tokenizer.get_vocab_size(),
                    temperature=1.0,
                    top_p=1.0,
                    return_behavior=True,
                )
                for sample_index, seq in enumerate(generated):
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
                    behavior_rows.append(
                        [0.0] * (prompt.shape[1] - 1)
                        + behavior[sample_index, action_mask[sample_index]].tolist()
                    )
                    response = self.tokenizer.decode(
                        ids[prompt.shape[1] :], skip_special_tokens=True
                    )
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
            _, mask = token_log_probs(
                self.model(inputs, attention_mask=inputs.ne(0)).logits,
                labels,
                actions=True,
                forbidden_ids=forbidden_actions(self.model),
                vocab_size=self.tokenizer.get_vocab_size(),
            )
            if self.teachers:
                teacher_logp = torch.empty_like(old_logp)
                for key in sorted(set(teacher_keys)):
                    indices = torch.tensor(
                        [i for i, item in enumerate(teacher_keys) if item == key], device=x.device
                    )
                    teacher_logp[indices] = token_log_probs(
                        self.teachers[key](
                            inputs[indices], attention_mask=inputs[indices].ne(0)
                        ).logits,
                        labels[indices],
                        actions=True,
                        forbidden_ids=forbidden_actions(self.model),
                        vocab_size=self.tokenizer.get_vocab_size(),
                    )[0]
                advantages = mopd_advantages(teacher_logp, old_logp)
                ref_logp = None
            else:
                advantages = grouped_advantages(
                    torch.tensor(rewards, device=x.device), self.group_size
                )
                ref_logp = token_log_probs(
                    self.reference(inputs, attention_mask=inputs.ne(0)).logits,
                    labels,
                    actions=True,
                    forbidden_ids=forbidden_actions(self.model),
                    vocab_size=self.tokenizer.get_vocab_size(),
                )[0]
        # Policy gradients are computed in eval mode too: dropout must not change
        # the probability model that sampled these actions. Autograd remains on.
        try:
            logp, mask = token_log_probs(
                wrapped(inputs, attention_mask=inputs.ne(0)).logits,
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
        return loss, loss.detach(), loss.new_tensor(self.last_reward)
