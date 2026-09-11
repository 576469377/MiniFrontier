"""Single-device token-normalized MF1 training, exact resume and continuous P0-P3 state."""

import contextlib
import copy
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import digest, write_json
from minifrontier.data.minifrontier1_encoding import evaluation_items, open_dataset
from minifrontier.models.minifrontier1 import MiniFrontier1Config, MiniFrontier1ForCausalLM
from minifrontier.models.minifrontier1.mtp import mtp_targets
from minifrontier.models.minifrontier1.processing import CONTROL_VERSION, token_metadata
from minifrontier.multimodal import move
from minifrontier.training.metrics import mf1_scalars
from minifrontier.training.minifrontier1_curriculum import (
    collate_records,
    context_length,
    microbatches,
    pack_records,
)
from minifrontier.training.minifrontier1_optim import (
    QuantileBalance,
    make_optimizer,
    parameter_report,
)
from minifrontier.training.minifrontier1_strategy import (
    PHASES,
    SFT_MIX,
    TEXT_MIX,
    VISION_MIX,
    bindings,
    scheduler_factor,
    validate_gate,
)
from minifrontier.training.runtime import atomic_save, cpu_tree, restore_rng, rng_state
from minifrontier.training.validation import NativeCEValidation, validation_due


class Sampler:
    """Deterministic shuffled stream; optional CE-deficit buckets, with all counters resumable."""

    def __init__(self, dataset, seed, weights=None, *, length_filter=False):
        self.seed, self.rng = seed, random.Random(seed)
        self.windowable = {
            i
            for i in range(len(dataset))
            if hasattr(dataset, "windowable_at") and dataset.windowable_at(i)
        }
        self.document_offsets: dict[str, tuple[int, int]] = {}
        self.lengths = (
            {
                i: dataset.length_at(i)
                if hasattr(dataset, "length_at")
                else dataset[i]["input_ids"].shape[1]
                for i in range(len(dataset))
            }
            if length_filter
            else {}
        )
        self.buckets: dict[str, list[int]] = {}
        for i in range(len(dataset)):
            domain = (
                (
                    dataset.domain_at(i)
                    if hasattr(dataset, "domain_at")
                    else dataset.record(i)["domain"]
                )
                if weights
                else "all"
            )
            self.buckets.setdefault(domain, []).append(i)
        self.weights = weights or {"all": 1.0}
        if (
            set(self.weights) != set(self.buckets)
            or not math.isclose(sum(self.weights.values()), 1)
            or min(self.weights.values()) <= 0
        ):
            raise ValueError(
                "CE mixture must name every data bucket with positive weights summing to one"
            )
        self.orders = {k: self.rng.sample(v, len(v)) for k, v in self.buckets.items()}
        self.offsets = dict.fromkeys(self.buckets, 0)
        self.ce = dict.fromkeys(self.buckets, 0)
        self.examples = dict.fromkeys(self.buckets, 0)
        self.epochs = dict.fromkeys(self.buckets, 0)

    def next(self, max_length=None):
        domain = min(self.buckets, key=lambda k: self.ce[k] / self.weights[k])
        if max_length is not None and not any(
            i in self.windowable or self.lengths[i] <= max_length for i in self.buckets[domain]
        ):
            raise ValueError(
                f"no {domain} samples fit the selected {max_length}-token curriculum bucket"
            )
        while True:
            if self.offsets[domain] == len(self.orders[domain]):
                self.orders[domain] = self.rng.sample(
                    self.buckets[domain], len(self.buckets[domain])
                )
                self.offsets[domain] = 0
                self.epochs[domain] += 1
            index = self.orders[domain][self.offsets[domain]]
            self.offsets[domain] += 1
            if max_length is None or index in self.windowable or self.lengths[index] <= max_length:
                break
        self.examples[domain] += 1
        return index, domain

    def next_item(self, dataset, max_length=None):
        domain = min(self.buckets, key=lambda k: self.ce[k] / self.weights[k])
        if domain in self.document_offsets:
            index, start = self.document_offsets[domain]
        else:
            index, domain = self.next(max_length)
            start = 0
        if index not in self.windowable:
            return dataset[index], domain
        capacity = max_length or dataset.config.max_position_embeddings
        item = dataset.window_at(index, start, capacity)
        next_start = start + item["input_ids"].shape[1] - 1
        if next_start < dataset.length_at(index) - 1:
            self.document_offsets[domain] = (index, next_start)
        else:
            self.document_offsets.pop(domain, None)
        return item, domain

    def state_dict(self):
        return dict(
            seed=self.seed,
            buckets=self.buckets,
            weights=self.weights,
            orders=self.orders.copy(),
            offsets=self.offsets.copy(),
            ce=self.ce.copy(),
            examples=self.examples.copy(),
            epochs=self.epochs.copy(),
            rng=self.rng.getstate(),
            document_offsets=copy.deepcopy(self.document_offsets),
        )

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("seed", "buckets", "weights")):
            raise ValueError("sampler data buckets/seed/mixture changed")
        for key in ("orders", "offsets", "ce", "examples", "epochs"):
            setattr(self, key, state[key].copy())
        self.document_offsets = copy.deepcopy(state.get("document_offsets", {}))
        for domain, (index, start) in self.document_offsets.items():
            if (
                domain not in self.buckets
                or index not in self.buckets[domain]
                or index not in self.windowable
                or start < 1
                or (self.lengths and start >= self.lengths[index] - 1)
            ):
                raise ValueError("sampler document cursor differs from its text inventory")
        self.rng.setstate(state["rng"])


def _optimizer_named(model, optimizer):
    return {
        name: cpu_tree(optimizer.state[p])
        for name, p in model.named_parameters()
        if p in optimizer.state
    }


def _load_named(model, optimizer, states, device):
    for name, p in model.named_parameters():
        if p.requires_grad and name in states:
            state = move(states[name], device)
            # Adam's step scalar remains on CPU unless capturable/fused is enabled.
            if "exp_avg" in state and isinstance(state.get("step"), torch.Tensor):
                state["step"] = state["step"].cpu()
            optimizer.state[p] = state


def _forward(model, item, *, labels=True, return_hidden=False):
    return model(
        item["input_ids"],
        labels=item["labels"] if labels else None,
        media=item.get("media"),
        segment_ids=item.get("segment_ids"),
        return_logits=False,
        return_hidden=return_hidden,
    )


@torch.no_grad()
def evaluate(
    model, dataset, device, *, limit=0, max_length=None, selection=None, final=False, batch_size=1
):
    if selection is not None and (
        limit or selection.dataset is not dataset or selection.max_length != max_length
    ):
        raise ValueError(
            "fixed native validation must use its bound dataset/context without a limit"
        )
    scope = "phase_end" if final else "periodic"
    started = time.monotonic()
    was_training = model.training
    model.eval()
    losses: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    wrong_losses: dict[str, float] = defaultdict(float)
    wrong_counts: Counter[str] = Counter()
    batches = 0
    attention_states = (
        [
            (layer.attention, layer.attention.training_phase)
            for layer in model.layers
            if layer.kind != "kda"
        ]
        if selection is not None
        else []
    )

    def selected_batches():
        if selection is None:
            for item in evaluation_items(dataset, limit=limit, max_length=max_length):
                yield item["domain"], [item], None
        else:
            for domain, indices in selection.batches(final=final, batch_size=batch_size):
                yield (
                    domain,
                    [selection[i] for i in indices],
                    sum(selection.counts[i] for i in indices),
                )

    try:
        # P2 training can leave a layer on its last dense replay microbatch.
        # Evaluate the declared model phase consistently, then restore the training state.
        for attention, _phase in attention_states:
            attention.training_phase = model.training_phase
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            for domain, rows, expected in selected_batches():
                item = move(
                    collate_records(rows, model.config.pad_token_id) if len(rows) > 1 else rows[0],
                    device,
                )
                count = int(item["labels"][:, 1:].ne(-100).sum())
                if expected is not None and count != expected:
                    raise ValueError("native validation CE differs from its stored loss mask")
                if not count:
                    continue
                batches += 1
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    output = _forward(model, item)
                    loss = float(output.lm_loss)
                    if not math.isfinite(loss):
                        raise ValueError("nonfinite native validation loss")
                    losses[domain] += loss * count
                    counts[domain] += count
                    if item.get("media"):
                        # A mixed-domain source can contain both text-only and media rows.
                        # Black-media denominators cover only records with actual media.
                        media_rows = [r for r in rows if r.get("media")]
                        media_item = (
                            item
                            if len(media_rows) == len(rows)
                            else move(
                                collate_records(media_rows, model.config.pad_token_id), device
                            )
                        )
                        altered = dict(
                            media_item,
                            media=[
                                dict(s, patches=torch.zeros_like(s["patches"]))
                                for s in media_item["media"]
                            ],
                        )
                        changed = _forward(model, altered)
                        black_loss = float(changed.lm_loss)
                        if not math.isfinite(black_loss):
                            raise ValueError("nonfinite black-media validation loss")
                        black_count = int(media_item["labels"][:, 1:].ne(-100).sum())
                        wrong_losses[domain] += black_loss * black_count
                        wrong_counts[domain] += black_count
    finally:
        for attention, phase in attention_states:
            attention.training_phase = phase
        model.train(was_training)
    if selection is not None and (
        sum(counts.values()) != selection.binding[scope]["actual_ce_tokens"]
        or dict(counts) != selection.binding[scope]["domain_ce"]
    ):
        raise ValueError("native validation totals differ from the fixed CE selection")
    result = dict(
        nll=sum(losses.values()) / max(1, sum(counts.values())),
        ce_tokens=sum(counts.values()),
        per_domain={k: losses[k] / counts[k] for k in counts if counts[k]},
        black_media_nll={
            k: wrong_losses[k] / wrong_counts[k] for k in wrong_counts if wrong_counts[k]
        },
        capability_qualified=False,
        examples=min(len(dataset), limit or len(dataset)),
        selection="full_validation" if not limit or limit >= len(dataset) else "explicit_prefix",
        context_length=max_length,
    )
    if selection is not None:
        result.update(
            evaluation_scope=scope,
            requested_ce_tokens=selection.binding[scope]["requested_ce_tokens"],
            selection="fixed_without_replacement",
            selection_sha256=selection.binding[scope]["indices_sha256"],
            examples=selection.binding[scope]["examples"],
            example_unit="native answer or text context window",
            domain_ce=dict(counts),
            black_media_ce=dict(wrong_counts),
            micro_batches=batches,
            duration_seconds=time.monotonic() - started,
            attention_phase=model.training_phase,
        )
    return result


def train(
    *,
    data,
    output,
    phase="pilot",
    config=None,
    device="cpu",
    steps=None,
    token_budget=None,
    input_batch_tokens=16384,
    batch_size=8,
    seed=42,
    init=None,
    resume=None,
    run_kind="acceptance",
    evidence=None,
    stop_after_updates=None,
    optimizer_kind="adamw",
    lr=None,
    vision_lr=None,
    save_every=100,
    eval_every=100,
    pretraining_eval=False,
    weights=None,
    diagnostic_attention=None,
    profile_warmup=50,
    profile_updates=200,
):
    performance_only = run_kind == "performance"
    if performance_only:
        if (
            phase != "p0"
            or init
            or resume
            or stop_after_updates is not None
            or token_budget is not None
            or steps is not None
            or not 1 <= profile_warmup < 250
            or not 1 <= profile_updates <= 250 - profile_warmup
        ):
            raise ValueError(
                "P0 performance requires a fresh bounded warmup/measurement window, without weights or token budgets"
            )
        steps = profile_warmup + profile_updates
    if diagnostic_attention is not None and (
        run_kind != "acceptance" or phase != "sft" or diagnostic_attention != "dense_pretrain"
    ):
        raise ValueError("attention override is restricted to dense acceptance SFT diagnosis")
    if phase not in {"pilot", "p0", "p1", "indexer", "p2", "p3", "sft"}:
        raise ValueError("use the posttrain command for RL/teacher/OPD/DPO/draft/QAT")
    if (
        (init and resume)
        or input_batch_tokens < 1
        or batch_size < 1
        or save_every < 1
        or eval_every < 1
    ):
        raise ValueError("invalid training controls")
    if run_kind not in {"acceptance", "strategy", "performance"}:
        raise ValueError("unknown run kind")
    if pretraining_eval and (phase not in {"p0", "p1", "p2", "p3"} or performance_only):
        raise ValueError("fixed CE validation belongs to main base-training phases")
    pretraining_eval = pretraining_eval or (
        run_kind == "strategy" and phase in {"p0", "p1", "p2", "p3"}
    )
    if steps is None and token_budget is None:
        token_budget = PHASES[phase]["budget"]
    if run_kind == "acceptance" and (
        steps is None
        or not 1 <= steps <= 10000
        or (token_budget is not None and token_budget > 2_000_000)
    ):
        raise ValueError(
            "acceptance requires 1-10000 explicit updates and at most 2M diagnostic tokens"
        )
    if (token_budget is not None and token_budget < 1) or (
        stop_after_updates is not None and stop_after_updates < 1
    ):
        raise ValueError("budgets/stop point must be positive")
    if run_kind == "strategy" and token_budget != PHASES[phase]["budget"]:
        raise ValueError("formal stage budget must match the frozen plan")
    if run_kind == "strategy" and steps is not None:
        raise ValueError("formal stages end at their token budget; use stop_after_updates to pause")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "MF1 defaults to independent single-device jobs; DDP needs separate benchmark admission"
        )
    device = torch.device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    output, data = Path(output).resolve(), Path(data).resolve()
    if performance_only and output.exists():
        raise FileExistsError("performance measurement requires a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and not resume:
        raise FileExistsError(
            "run already has a recoverable checkpoint; choose --resume or a new output"
        )
    saved = (
        torch.load(resume or init, weights_only=True, map_location="cpu")
        if resume or init
        else None
    )
    values = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
    c = (
        MiniFrontier1Config(**(values or cast(dict, saved)["config"]))
        if values or saved
        else MiniFrontier1Config()
    )
    dataset, validation = open_dataset(data, "train", c), open_dataset(data, "val", c)
    if not len(dataset) or not len(validation):
        raise ValueError("training and held-out validation must both be nonempty")
    if dataset.tokenizer.get_vocab_size() > c.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds output head")
    bound = bindings(c, data, init)
    production_path = run_kind in {"strategy", "performance"}
    if production_path and weights is None:
        if phase == "sft":
            weights = SFT_MIX
        else:
            visual = PHASES[phase].get("visual_ce", 0.20)
            vision_mix = dict(VISION_MIX, caption=0.35, video=0.0) if phase == "p0" else VISION_MIX
            weights = {
                **{k: v * (1 - visual) for k, v in TEXT_MIX.items()},
                **{k: v * visual for k, v in vision_mix.items() if v > 0},
            }
    if resume:
        assert saved is not None
        bound["actual_init_checkpoint_sha256"] = saved["run_spec"]["actual_init_checkpoint_sha256"]
    gate = dict(status="diagnostic_only", stage=phase, formal_admission=False)
    if run_kind == "strategy":
        if evidence is None:
            raise ValueError("formal MF1 training requires a checkpoint-bound stage gate")
        gate = validate_gate(
            phase,
            json.loads(Path(evidence).read_text()),
            bound,
            dataset.manifest,
            saved,
            resume=bool(resume),
        )
    evaluation_length = (
        max(
            length
            for length in PHASES[phase].get("lengths", {c.max_position_embeddings: 1})
            if length <= c.max_position_embeddings
        )
        if production_path
        else c.max_position_embeddings
    )
    ce_validation = (
        NativeCEValidation(validation, max_length=evaluation_length, seed=seed)
        if pretraining_eval
        else None
    )
    attention = diagnostic_attention or PHASES[phase]["attention"]
    model = MiniFrontier1ForCausalLM(c, attention).to(device)
    if saved:
        if (
            saved["model_name"] != "minifrontier1"
            or cast(dict, saved)["config"] != asdict(c)
            or saved["tokenizer_sha256"] != bound["tokenizer_sha256"]
        ):
            raise ValueError("checkpoint model/config/tokenizer differs from actual inputs")
        model.load_state_dict(saved["model"], strict=True)
    peak = (
        lr if lr is not None else 2e-5 if phase == "sft" else 1e-4 if phase == "indexer" else 3e-4
    )
    opt = make_optimizer(
        model,
        lr=peak,
        vision_lr=vision_lr or (5e-6 if phase == "sft" else 1e-4),
        scalar_lr=min(peak, 1e-4),
        kind=optimizer_kind,
    )
    group_manifest = [
        dict(
            name=g["name"],
            parameters=[n for n, p in model.named_parameters() if any(p is q for q in g["params"])],
            base_lr=g["base_lr"],
            weight_decay=g["weight_decay"],
        )
        for g in opt.param_groups
    ]
    run = dict(
        format="mf1-run-v1",
        model_name="minifrontier1",
        kind=run_kind,
        mf1_phase=phase,
        **bound,
        seed=seed,
        steps=steps,
        token_budget=token_budget,
        unit=PHASES[phase]["unit"],
        input_batch_tokens=input_batch_tokens,
        batch_size=batch_size,
        optimizer_kind=optimizer_kind,
        optimizer_groups_sha256=digest(group_manifest),
        mixture=weights,
        chat_template=CONTROL_VERSION,
        diagnostic_attention=diagnostic_attention,
    )
    if performance_only:
        largest_context = max(n for n in PHASES[phase]["lengths"] if n <= c.max_position_embeddings)
        run.update(
            main_budget_eligible=False,
            formal_admission=False,
            exports_model_weights=False,
            performance_profile=dict(warmup=profile_warmup, measured_updates=profile_updates),
            input_token_upper_bound=steps * (input_batch_tokens + largest_context - 1),
            model_config=asdict(c),
            optimizer_groups=group_manifest,
            schedule="shared production P0 main-CE warmup; measurement warmup is separate",
        )
    sampler, balance = (
        Sampler(dataset, seed, weights, length_filter=production_path),
        QuantileBalance(model),
    )
    if ce_validation is not None:
        ce_validation.binding["attention_phase"] = attention
        run["validation"] = ce_validation.binding
    ledger = dict(
        input_tokens=0,
        ce_tokens=0,
        generated_tokens=0,
        vision_tokens=0,
        mtp_target_tokens=0,
        index_query_tokens=0,
        replay_ce_tokens=0,
        media_exposures=0,
        main_ce_tokens=0,
        phase_tokens=0,
        optimizer_updates=0,
    )
    inherited_states = {}
    if saved:
        # P0→P1 and P1→I→P2→P3 preserve the main moments; SFT starts a new optimization task.
        continuous = phase in {"p1", "indexer", "p2", "p3"} and saved.get("mf1_phase") in {
            "p0",
            "p1",
            "indexer",
            "p2",
        }
        if resume:
            from .execution_upgrade import resume_matches

            previous_run = dict(saved["run_spec"])
            # Resume binds the original initialization artifact, not the resume file as a new init.
            run["actual_init_checkpoint_sha256"] = previous_run["actual_init_checkpoint_sha256"]
            if not resume_matches(previous_run, run, resume) or saved["mf1_phase"] != phase:
                raise ValueError(
                    "exact resume requires the same data/source/config/optimizer/budget/phase"
                )
            opt.load_state_dict(saved["optimizer"])
            sampler.load_state_dict(saved["sampler"])
            balance.load_state_dict(saved["router_balance"])
            ledger = saved["ledger"]
            inherited_states = saved.get("continuous_optimizer_states", {})
            restore_rng(saved["rng"], device)
        elif continuous:
            if saved["run_spec"]["optimizer_kind"] != optimizer_kind:
                raise ValueError("continuous pretraining cannot silently replace its optimizer")
            inherited_states = saved["continuous_optimizer_states"]
            _load_named(model, opt, inherited_states, device)
            balance.load_state_dict(saved["router_balance"])
            ledger["main_ce_tokens"] = saved["ledger"]["main_ce_tokens"]
            # Keep lifetime exposures and domain CE counters visible across phase changes.
            if (
                saved["sampler"]["buckets"] == sampler.buckets
                and saved["sampler"]["weights"] == sampler.weights
            ):
                sampler.load_state_dict(saved["sampler"])
    if not resume:
        shutil.copyfile(data / "tokenizer.json", output / "tokenizer.json")
        write_json(output / "run.json", run)
        write_json(output / "resolved_config.json", asdict(c))
        write_json(output / "source_manifest.json", bound["source"])
        write_json(output / "data_manifest.json", dataset.manifest)
        write_json(output / "param_report.json", parameter_report(model, opt))
        write_json(output / "optimizer_groups.json", group_manifest)
        write_json(output / "stage_gate.json", gate)
        write_json(
            output / "environment.json",
            dict(
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                device=str(device),
                sharing="not measured; do not infer isolated throughput",
            ),
        )
    writer = None
    with contextlib.suppress(ImportError):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output / "tensorboard"))
    if performance_only and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, step_started = time.monotonic(), time.monotonic()
    profile = []
    from .prefetch import OrderedPrefetch
    from .prefetch import enabled as prefetch_enabled

    producer_tokens = ledger["phase_tokens"]

    def produce_window():
        nonlocal producer_tokens
        window, inputs = [], 0
        capacity = (
            context_length(phase, sampler.rng, producer_tokens, c.max_position_embeddings)
            if production_path
            else None
        )
        packed: list[dict[str, Any]] = []
        packed_length = 0
        while inputs < input_batch_tokens:
            item, domain = sampler.next_item(dataset, capacity)
            item["bucket"] = domain
            if capacity is not None and item["input_ids"].shape[1] > capacity:
                raise ValueError(
                    "sample exceeds selected curriculum bucket; prepare length-specific native shards instead of silently truncating"
                )
            count = int(item["labels"][:, 1:].ne(-100).sum())
            sampler.ce[domain] += count
            if capacity is None:
                window.append(item)
            else:
                if packed_length + item["input_ids"].shape[1] > capacity:
                    window.append(pack_records(packed, capacity))
                    packed, packed_length = [], 0
                packed.append(item)
                packed_length += item["input_ids"].shape[1]
            inputs += item["input_ids"].numel()
        if packed:
            window.append(pack_records(packed, capacity))
        ce_count = sum(int(item["labels"][:, 1:].ne(-100).sum()) for item in window)
        producer_tokens += inputs if phase == "indexer" else ce_count
        return (window, inputs, capacity, ce_count), sampler.state_dict()

    prefetch = None
    if prefetch_enabled():
        if phase not in {"pilot", "p0", "p1", "p3"}:
            raise ValueError("MF1 lookahead is limited to phases with producer-owned sampling RNG")
        prefetch = OrderedPrefetch(produce_window, sampler.state_dict())

    unique_media = set(cast(dict, saved).get("seen_media", [])) if resume else set()
    step = ledger["optimizer_updates"]

    def record(values):
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")
        if writer:
            for tag, value in mf1_scalars(values).items():
                writer.add_scalar(tag, value, step)
            writer.flush()

    def save(state):
        if performance_only:
            write_json(
                output / "status.json",
                dict(
                    kind="performance",
                    mf1_phase=phase,
                    state=state,
                    step=step,
                    ledger=ledger,
                    main_budget_eligible=False,
                    formal_admission=False,
                    exported_model_weights=False,
                    resumable=False,
                ),
            )
            return
        continuous_states = dict(inherited_states)
        continuous_states.update(_optimizer_named(model, opt))
        artifact = dict(
            format="mf1-checkpoint-v1",
            model_name="minifrontier1",
            config=asdict(c),
            phase=attention,
            stage="sft"
            if phase == "sft"
            else "dense_distill"
            if phase == "indexer"
            else "sparse_cpt"
            if phase in {"p2", "p3"}
            else "pretrain",
            mf1_phase=phase,
            step=step,
            model=model.state_dict(),
            tokenizer_sha256=bound["tokenizer_sha256"],
            chat_template=CONTROL_VERSION,
            run_spec=run,
            optimizer=opt.state_dict(),
            continuous_optimizer_states=continuous_states,
            sampler=prefetch.committed if prefetch is not None else sampler.state_dict(),
            router_balance=balance.state_dict(),
            ledger=ledger,
            rng=rng_state(device),
            seen_media=sorted(unique_media),
            capability_qualified=False,
        )
        atomic_save(artifact, output / "checkpoint.pt")
        write_json(
            output / "checkpoint_manifest.json",
            dict(
                sha256=sha256(output / "checkpoint.pt"),
                state=state,
                step=step,
                ledger=ledger,
                resumable=True,
                capability_qualified=False,
            ),
        )
        write_json(
            output / "status.json",
            dict(
                stage=artifact["stage"],
                mf1_phase=phase,
                state=state,
                step=step,
                ledger=ledger,
                unique_media=len(unique_media),
            ),
        )

    def finished():
        return (steps is not None and step >= steps) or (
            token_budget is not None and ledger["phase_tokens"] >= token_budget
        )

    last_evaluation = None
    pause = False

    def validate(final=False):
        nonlocal last_evaluation
        if last_evaluation is not None and last_evaluation[:2] == (step, final):
            return last_evaluation[2]
        metrics = evaluate(
            model,
            validation,
            device,
            max_length=evaluation_length,
            selection=ce_validation,
            final=final,
            batch_size=batch_size,
        )
        record(
            dict(event="validation", step=step, main_ce_tokens=ledger["main_ce_tokens"], **metrics)
        )
        last_evaluation = (step, final, metrics)
        return metrics

    try:
        model.train()
        if ce_validation is not None and step == 0 and not finished():
            validate()
            step_started = time.monotonic()
        while not finished():
            previous_main_ce = ledger["main_ce_tokens"]
            data_started = time.monotonic()
            window, inputs = [], 0
            if prefetch is not None:
                window, inputs, capacity, ce_count = prefetch.next()
            else:
                (window, inputs, capacity, ce_count), _ = produce_window()
            if (
                run_kind == "acceptance"
                and ledger["phase_tokens"] + (inputs if phase == "indexer" else ce_count)
                > 2_000_000
            ):
                # The current window has advanced the sampler but is not trained.
                # Keep any earlier checkpoint intact instead of saving this cursor.
                write_json(
                    output / "status.json",
                    dict(
                        kind=run_kind,
                        state="acceptance_token_limit",
                        step=step,
                        ledger=ledger,
                        rejected_window_tokens=inputs if phase == "indexer" else ce_count,
                    ),
                )
                raise ValueError("acceptance would exceed its 2M actual-token limit")
            mtp_count = 0
            for item in window:
                metadata = token_metadata(
                    item["input_ids"], c, item.get("media"), segment_ids=item.get("segment_ids")
                )
                mtp_count += int(
                    mtp_targets(item["input_ids"], item["labels"], metadata, c)[1].ne(-100).sum()
                )
            if not ce_count and phase != "indexer":
                raise ValueError("all-mask window has no training signal; fix the dataset")
            data_seconds = time.monotonic() - data_started
            factor = (
                1.0
                if run_kind == "acceptance"
                else scheduler_factor(
                    phase,
                    ledger["phase_tokens"] + inputs
                    if phase == "indexer"
                    else ledger["phase_tokens"] + ce_count,
                    ledger["main_ce_tokens"] + ce_count,
                )
            )
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * factor
                if "base_adam_lr" in group:
                    group["adam_lr"] = group["base_adam_lr"] * factor
            opt.zero_grad(set_to_none=True)
            totals: dict[str, float] = defaultdict(float)
            if phase == "p2":
                for raw in window:
                    sparse = sampler.rng.random() < min(1.0, ledger["phase_tokens"] / 20_000_000)
                    raw["attention_phase"] = "sparse_cpt" if sparse else "dense_distill"
            actual_batches = []
            padded_inputs = 0
            for raw in microbatches(
                window,
                batch_size=batch_size,
                max_padded_tokens=input_batch_tokens,
                pad_token_id=c.pad_token_id,
            ):
                # The labels already exist on CPU; avoid a CUDA scalar read per microbatch.
                local_ce = int(raw["labels"][:, 1:].ne(-100).sum())
                item = move(raw, device)
                actual_batches.append(item["input_ids"].shape[0])
                padded_inputs += item["input_ids"].numel()
                if phase == "p2":
                    for layer in model.layers:
                        if layer.kind != "kda":
                            cast(Any, layer).attention.training_phase = raw["attention_phase"]
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    with balance.capture(item["input_ids"].ne(0)):
                        result = _forward(model, item)
                    local_queries = result.index_query_tokens
                    objective = (
                        result.indexer_loss * local_queries
                        if phase == "indexer"
                        else result.lm_loss * local_ce / ce_count
                    )
                    if phase != "indexer":
                        if result.mtp_loss is not None:
                            objective = (
                                objective
                                + c.mtp_loss_coef
                                * result.mtp_loss
                                * result.mtp_tokens
                                / max(1, mtp_count)
                            )
                        objective = (
                            objective + c.indexer_loss_coef * result.indexer_loss * local_queries
                        )
                if phase != "indexer" or local_queries:
                    objective.backward()
                totals["lm_sum"] += float(result.lm_loss.detach()) * local_ce
                totals["index_query_tokens"] += local_queries
                totals["mtp_target_tokens"] += result.mtp_tokens
                totals["vision_tokens"] += sum(s["feature_count"] for s in item.get("media", []))
                totals["media_exposures"] += item.get("media_exposures", len(item.get("media", [])))
                unique_media.update(item["media_hashes"])
            # Only indexer parameters receive this independent KL gradient/denominator.
            for name, p in model.named_parameters():
                if ".indexer." in name and p.grad is not None:
                    p.grad.div_(max(1, totals["index_query_tokens"]))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if phase == "indexer" and not totals["index_query_tokens"]:
                raise ValueError(
                    "indexer batch has no eligible historical KL queries; increase history or inspect masks"
                )
            opt.step()
            router_metrics = balance.update() if phase != "indexer" else {}
            step += 1
            ledger["optimizer_updates"] = step
            ledger["input_tokens"] += inputs
            if phase != "indexer":
                ledger["ce_tokens"] += ce_count
            if phase in {"p0", "p1", "p2", "p3"}:
                ledger["main_ce_tokens"] += ce_count
            ledger["phase_tokens"] += inputs if phase == "indexer" else ce_count
            for name in (
                "index_query_tokens",
                "mtp_target_tokens",
                "vision_tokens",
                "media_exposures",
            ):
                ledger[name] += int(totals[name])
            if performance_only and device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - step_started
            metrics = dict(
                step=step,
                train_lm_loss=totals["lm_sum"] / max(1, ce_count),
                grad_norm=float(norm),
                step_seconds=elapsed,
                ce_per_second=ce_count / elapsed if phase != "indexer" else 0,
                input_per_second=inputs / elapsed,
                data_preparation_seconds=data_seconds,
                ce_fraction=ce_count / inputs if phase != "indexer" else 0,
                padding_fraction=(padded_inputs - inputs) / padded_inputs,
                input_batch_actual=inputs,
                micro_batches=len(actual_batches),
                microbatch_max_samples=max(actual_batches),
                microbatch_samples=actual_batches,
                **ledger,
            )
            if device.type == "cuda":
                metrics.update(
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                )
                if torch.cuda.max_memory_reserved(device) > 22 * 1024**3:
                    save("memory_limit")
                    raise RuntimeError("MF1 exceeded the 22 GiB single-device admission limit")
            if performance_only:
                if device.type == "cuda":
                    metrics["device_free_gib"] = torch.cuda.mem_get_info(device)[0] / 1024**3
                    if metrics["device_free_gib"] < 2:
                        save("memory_limit")
                        raise RuntimeError("MF1 performance device has less than 2 GiB free")
                if profile_warmup < step <= profile_warmup + profile_updates:
                    profile.append(
                        dict(
                            step=step,
                            seconds=elapsed,
                            data_preparation_seconds=data_seconds,
                            ce_tokens=ce_count,
                            input_tokens=inputs,
                            context_length=capacity,
                            vision_tokens=int(totals["vision_tokens"]),
                            media_exposures=int(totals["media_exposures"]),
                            microbatch_samples=actual_batches,
                            padding_fraction=metrics["padding_fraction"],
                            lm_loss=metrics["train_lm_loss"],
                            grad_norm=metrics["grad_norm"],
                            learning_rates={g["name"]: g["lr"] for g in opt.param_groups},
                            peak_allocated_gib=metrics.get("peak_allocated_gib"),
                            peak_reserved_gib=metrics.get("peak_reserved_gib"),
                            device_free_gib=metrics.get("device_free_gib"),
                        )
                    )
                if step == profile_warmup and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            record(metrics)
            if router_metrics:
                with (output / "router_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(step=step, routers=router_metrics)) + "\n")
            due = (
                not finished()
                and validation_due(previous_main_ce, ledger["main_ce_tokens"], ce_validation.policy)
                if ce_validation is not None
                else step % eval_every == 0
            )
            if not performance_only and due:
                validate()
            pause_request = output / "pause.request"
            pause = (
                stop_after_updates is not None and step >= stop_after_updates
            ) or pause_request.exists()
            if step % save_every == 0 or pause:
                save("paused" if pause else "running")
            if pause:
                pause_request.unlink(missing_ok=True)
                break
            step_started = time.monotonic()
        paused = pause and not finished()
        if performance_only:
            if (
                len(profile) != profile_updates
                or ledger["input_tokens"] > run["input_token_upper_bound"]
            ):
                raise ValueError("performance measurement window or input budget differs")
            seconds = sum(row["seconds"] for row in profile)
            result = dict(
                recipe=run,
                updates=profile,
                measured_updates=len(profile),
                ce_tokens_per_second=sum(row["ce_tokens"] for row in profile) / seconds,
                input_tokens_per_second=sum(row["input_tokens"] for row in profile) / seconds,
                measurement_seconds=seconds,
                total_elapsed_seconds=time.monotonic() - started,
                all_updates_ledger=ledger,
                domain_ce=dict(sampler.ce),
                unique_media=len(unique_media),
                context_updates=dict(Counter(row["context_length"] for row in profile)),
                device=str(device),
                formal_admission=False,
                main_budget_eligible=False,
                capability_qualified=False,
                exports_model_weights=False,
                state="measurement_complete_unqualified",
                scope="P0 production batch/loss/optimizer path; performance evidence only, not data, resume or capability admission",
                includes=[
                    "data",
                    "media_processing",
                    "packing",
                    "forward",
                    "backward",
                    "mtp",
                    "optimizer",
                    "router_balance",
                ],
            )
            write_json(output / "performance.json", result)
            save(result["state"])
            return result
        save("paused" if paused else "budget_complete_unqualified")
        evaluation = (
            validate(final=not paused)
            if ce_validation is not None
            else evaluate(model, validation, device, max_length=evaluation_length)
        )
        write_json(
            output / "evaluation.json",
            dict(
                checkpoint_sha256=sha256(output / "checkpoint.pt"),
                metrics=evaluation,
                split="val",
                diagnostic_only=run_kind == "acceptance",
            ),
        )
    except BaseException as error:
        if performance_only:
            save("measurement_failed")
            write_json(
                output / "failure.json",
                dict(error=type(error).__name__ + ": " + str(error), step=step, ledger=ledger),
            )
        raise
    finally:
        if prefetch is not None:
            prefetch.close()
        if writer:
            writer.close()
    return dict(
        output=str(output),
        step=step,
        state="paused" if paused else "budget_complete_unqualified",
        ledger=ledger,
        evaluation=evaluation,
        elapsed_seconds=time.monotonic() - started,
    )
