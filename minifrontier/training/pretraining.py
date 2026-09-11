"""Explicit multi-phase base-training schedules and optimizer state inheritance.

The existing experiment plan owns the recipe. This module does not schedule jobs
or grant data, performance, or capability admission.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from minifrontier.data import sha256
from minifrontier.multimodal import move


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_parent_artifact(evidence_path, phase, checkpoint):
    evidence = json.loads(Path(evidence_path).read_text())
    parent = evidence.get("completed_phases", {}).get(phase, {})
    if not parent.get("quality_passed") or parent.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(
            "phase quality evidence does not bind the actual initialization checkpoint"
        )


def schedule_factor(schedule, position):
    start, warmup, decay, end = (
        schedule[k] for k in ("start", "warmup_tokens", "decay_start", "end")
    )
    floor = schedule.get("final_factor", 0.1)
    if not (0 <= start < end and 0 < warmup <= decay - start and decay <= end and 0 <= floor <= 1):
        raise ValueError("invalid pretraining schedule boundaries")
    if position < start + warmup:
        return min(1.0, max(1, position - start) / warmup)
    if position <= decay or decay == end:
        return 1.0
    fraction = min(1.0, (position - decay) / (end - decay))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))


def optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    keys = (
        "blocks",
        "recipe",
        "coefficients",
        "scaling",
        "weight_decay",
        "betas",
        "eps",
        "momentum",
    )
    return [
        dict(
            names=[names[id(p)] for p in group["params"]],
            contract={k: group[k] for k in keys if k in group},
        )
        for group in optimizer.param_groups
    ]


def restore_optimizer(model, optimizer, saved):
    """Map moments by parameter name, retaining only frozen parameters on CPU."""
    previous = saved["pretraining_state"]
    available = dict(previous["dormant_optimizer"])
    for group, description in zip(
        saved["optimizer"]["param_groups"], previous["optimizer_groups"], strict=True
    ):
        for index, name in zip(group["params"], description["names"], strict=True):
            if name in available:
                raise ValueError("duplicate pretraining optimizer parameter")
            if index in saved["optimizer"]["state"]:
                available[name] = dict(
                    state=saved["optimizer"]["state"][index], contract=description["contract"]
                )
    parameters = dict(model.named_parameters())
    if available.keys() - parameters.keys():
        raise ValueError("pretraining optimizer refers to removed parameters")
    for group, description in zip(
        optimizer.param_groups, optimizer_groups(model, optimizer), strict=True
    ):
        for name, parameter in zip(description["names"], group["params"], strict=True):
            if name not in available:
                continue  # Newly active indexers or experts without previous gradients.
            old = available.pop(name)
            if _digest(old["contract"]) != _digest(description["contract"]):
                raise ValueError(f"pretraining optimizer contract changed: {name}")
            optimizer.state[parameter] = move(old["state"], parameter.device)
            if isinstance(optimizer, torch.optim.AdamW) and "step" in optimizer.state[parameter]:
                optimizer.state[parameter]["step"] = optimizer.state[parameter]["step"].cpu()
    return available


class PretrainingProgram:
    def __init__(self, args):
        document = json.loads(Path(args.pretraining_program).read_text())["execution_program"]
        model = document["models"][args.model]
        self.recipe = model["training_recipe"]
        native_optimizer = {
            "minikimik3": "kimi_muon",
            "miniqwen4": "qwen_muon",
            "minideepseekv4": "deepseek_muon",
        }[args.model]
        if self.recipe["optimizer"] != native_optimizer:
            raise ValueError("base program requires the selected model-specific optimizer")
        self.phases = self.recipe["phases"]
        selected = next((p for p in self.phases if p["id"] == args.pretraining_phase), None)
        if selected is None:
            raise ValueError("unknown pretraining program phase")
        self.phase: dict[str, Any] = selected
        self.number = self.phases.index(self.phase)
        self.program_id, self.model, self.execution_kind = document["id"], args.model, args.run_kind
        if self.recipe.get("bindings", {}).keys() - {p["id"] for p in self.phases}:
            raise ValueError("data binding refers to an unknown pretraining phase")
        self.binding = self.binding_through(self.number)
        self.main_base = 0
        self.dormant = {}
        self.lineage = []
        self.transition = None
        if len({p["id"] for p in self.phases}) != len(self.phases):
            raise ValueError("duplicate pretraining phase identity")
        if sum(p["budget"] for p in self.phases if p["unit"] == "ce_tokens") != model["main_ce"]:
            raise ValueError("program phases do not sum to its complete main CE budget")
        for phase in self.phases:
            if phase["unit"] not in {"ce_tokens", "input_tokens"} or phase["budget"] <= 0:
                raise ValueError("invalid pretraining phase budget")
            if (phase["unit"] == "input_tokens") != (phase["stage"] == "dense_distill"):
                raise ValueError("indexer input and main CE objectives must remain separate")
            schedule_factor(phase["schedule"], 1)
            if phase["schedule"]["axis"] not in {"main_ce", "phase_input"}:
                raise ValueError("unknown pretraining schedule axis")
            if any(not math.isfinite(v) or v <= 0 for v in phase["peak_lr"].values()):
                raise ValueError("pretraining peak rates must be finite and positive")
            if (
                args.model == "minideepseekv4"
                and phase["peak_lr"]["muon"] != phase["peak_lr"]["adam"]
            ):
                raise ValueError("DeepSeek backbone Muon and Adam rates must remain shared")
        if (
            args.schedule != "program"
            or args.warmup_tokens is not None
            or args.input_batch_schedule
        ):
            raise ValueError("program owns its complete token schedule; use --schedule program")
        if (
            args.stage != self.phase["stage"]
            or (args.ce_tokens or args.input_tokens) != self.phase["budget"]
        ):
            raise ValueError("training stage/budget differs from pretraining program")
        if (
            args.seed != self.recipe["seed"]
            or args.input_batch_tokens != self.recipe["global_input_tokens"]
            or args.input_batch_policy != "sample-bounded"
            or args.optimizer not in {"auto", self.recipe["optimizer"]}
            or args.init_transition != self.phase.get("init_transition", "exact")
            or args.sequence_length not in self.phase["sequence_lengths"]
        ):
            raise ValueError(
                "seed/batch/optimizer/context/transition differs from pretraining program"
            )
        if args.strategy_phase and args.strategy_phase != self.phase["id"]:
            raise ValueError("strategy and pretraining program phases disagree")
        for key in ("weight_decay", "adam_eps", "clip_grad", "router_bias_rate"):
            if getattr(args, key) != self.recipe[key]:
                raise ValueError(f"optimizer setting differs from pretraining program: {key}")
        if (
            args.lr != self.phase["peak_lr"]["adam"]
            or args.muon_lr != self.phase["peak_lr"]["muon"]
        ):
            raise ValueError("CLI peak rates differ from pretraining phase")
        if args.run_kind == "strategy" and self.recipe.get("status") != "frozen":
            raise ValueError("formal pretraining recipe still lacks data/resource freeze")
        if args.run_kind == "strategy":
            frozen = self.recipe.get("bindings", {}).get(self.phase["id"], {})
            requested = dict(
                data_sha256=sha256(Path(args.data) / "manifest.json"),
                config_sha256=sha256(args.config),
                tokenizer_sha256=sha256(Path(args.data) / "tokenizer.json"),
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
            )
            if any(frozen.get(k) != value for k, value in requested.items()):
                raise ValueError(
                    "formal data/config/tokenizer/microbatch bindings differ from recipe"
                )

    def binding_through(self, number):
        """Keep the full schedule fixed while later phases acquire their data.

        A checkpoint binds every data/config/tokenizer/microbatch assignment up
        to its own phase. Future assignments cannot invalidate it or qualify the
        next phase; that phase must independently pass its formal bindings gate.
        """
        recipe = dict(self.recipe)
        through = {p["id"] for p in self.phases[: number + 1]}
        recipe["bindings"] = {
            key: value for key, value in recipe.get("bindings", {}).items() if key in through
        }
        return dict(
            program_id=self.program_id,
            model=self.model,
            recipe=recipe,
            recipe_sha256=_digest(recipe),
            binding_scope="data_through_checkpoint_phase_v1",
            phase=self.phases[number]["id"],
            execution_kind=self.execution_kind,
        )

    def validate_parent(self, args, saved):
        if args.resume:
            if not saved or saved.get("run_spec", {}).get("pretraining_program") != self.binding:
                raise ValueError("exact resume requires the same pretraining program and phase")
            if "pretraining_state" not in saved:
                raise ValueError("checkpoint lacks pretraining state")
            return
        if self.number == 0:
            if saved is not None:
                raise ValueError("first formal/program phase must start from random initialization")
            return
        if not saved or "pretraining_state" not in saved:
            raise ValueError("phase transition requires a full preceding program checkpoint")
        previous = saved["run_spec"].get("pretraining_program", {})
        expected = self.binding_through(self.number - 1)
        if previous != expected or not saved["pretraining_state"].get("phase_complete"):
            raise ValueError("program predecessor/recipe/kind is different or incomplete")
        prior_phase = self.phases[self.number - 1]
        prior_state = saved["pretraining_state"]
        if (
            saved["token_ledger"][prior_phase["unit"]] < prior_phase["budget"]
            or prior_state["main_ce_tokens"]
            != prior_state["main_ce_base"] + saved["token_ledger"]["ce_tokens"]
            or (prior_phase["unit"] == "input_tokens" and saved["token_ledger"]["ce_tokens"] != 0)
        ):
            raise ValueError("program predecessor token ledger is incomplete or inconsistent")
        if args.run_kind == "strategy":
            validate_parent_artifact(args.strategy_evidence, prior_phase["id"], args.init)
        self.main_base = saved["pretraining_state"]["main_ce_tokens"]
        self.lineage = [
            *saved["pretraining_state"]["lineage"],
            dict(
                phase=previous["phase"],
                checkpoint_sha256=sha256(args.init),
                token_ledger=saved["token_ledger"],
                main_ce_tokens=self.main_base,
            ),
        ]

    def inherit(self, model, optimizer, saved, *, resume):
        if saved is None:
            return
        if resume:
            previous = saved["pretraining_state"]
            self.main_base = previous["main_ce_base"]
            self.dormant, self.lineage = previous["dormant_optimizer"], previous["lineage"]
            self.transition = previous["transition"]
        else:
            self.dormant = restore_optimizer(model, optimizer, saved)

    def main_ce(self, ledger):
        return self.main_base + ledger.ce_tokens

    def rates(self, optimizer, ledger, incoming):
        schedule = self.phase["schedule"]
        position = (
            self.main_ce(ledger) if schedule["axis"] == "main_ce" else ledger.input_tokens
        ) + incoming
        factor = schedule_factor(schedule, position)
        peaks = self.phase["peak_lr"]
        for group in optimizer.param_groups:
            name = group["name"]
            rate = (
                peaks["indexer"]
                if ".indexer." in name
                else peaks.get("projector", peaks["adam"])
                if name.startswith(("image_", "vision.aligner.", "vision.merger."))
                else peaks.get("vision", peaks["adam"])
                if name.startswith("vision.")
                else None
            )
            group["lr"] = (peaks["muon"] if rate is None else rate) * factor
            group["adam_lr"] = (peaks["adam"] if rate is None else rate) * factor
        return factor

    def snapshot(self, model, optimizer, ledger, *, complete):
        return dict(
            main_ce_base=self.main_base,
            main_ce_tokens=self.main_ce(ledger),
            phase_complete=complete,
            optimizer_groups=optimizer_groups(model, optimizer),
            dormant_optimizer=self.dormant,
            lineage=self.lineage,
            transition=self.transition,
        )
