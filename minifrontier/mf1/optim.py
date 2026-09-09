"""Complete optimizer ownership, bounded accumulation-level quantile balancing and reports."""

import contextlib
from collections import defaultdict
from dataclasses import asdict

import torch

from minifrontier.models.minifrontier1.moe import Router
from minifrontier.training.kimi_quantile_balance import QuantileHistogram


def parameter_group(name, p):
    if ".indexer." in name:
        return "indexer"
    if name.startswith("vision.") and not name.startswith("vision.merger."):
        return "vision" if p.ndim >= 2 else "scalar"
    if name.startswith(("embed_tokens.", "lm_head.", "lookup.tables.")):
        return "embedding"
    return "matrix" if p.ndim >= 2 else "scalar"


def make_optimizer(
    model, *, lr=3e-4, vision_lr=1e-4, scalar_lr=1e-4, indexer_lr=1e-4, kind="adamw", muon_lr=0.003
):
    opt: torch.optim.Optimizer
    rates = dict(matrix=lr, embedding=lr, vision=vision_lr, scalar=scalar_lr, indexer=indexer_lr)
    decay = dict(matrix=0.1, embedding=0.01, vision=0.1, scalar=0.0, indexer=0.01)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if kind == "muon":
        from minifrontier.training.semantic_optim import SemanticOptimizer

        specs = {}
        heads = model.config.num_attention_heads
        for name, p in named:
            group = parameter_group(name, p)
            blocks = None
            if p.ndim == 2 and group not in {"embedding", "indexer"} and ".router." not in name:
                parts = (
                    heads
                    if any(
                        s in name
                        for s in (
                            "core.q_proj",
                            "core.k_proj",
                            "core.v_proj",
                            "attention.q_up",
                            "attention.kv_up",
                        )
                    )
                    else 1
                )
                if "vision." in name and ".qkv." in name:
                    parts = 3 * model.config.vision_config.num_heads
                if p.shape[0] % parts:
                    raise ValueError(f"semantic head partition does not divide {name}")
                width = p.shape[0] // parts
                blocks = [(i * width, (i + 1) * width) for i in range(parts)]
            specs[id(p)] = (
                blocks,
                f"mf1/{group}/"
                + (
                    "per-head"
                    if blocks and len(blocks) > 1
                    else "whole-matrix"
                    if blocks
                    else "adamw"
                ),
            )
        opt = SemanticOptimizer(
            model,
            specs,
            lr=muon_lr,
            adam_lr=lr,
            coefficients=((3.4445, -4.775, 2.0315),) * 5,
            scaling=0.2,
            recipe="mf1-semantic-muon-v1",
        )
        for g in opt.param_groups:
            category = parameter_group(g["name"], g["params"][0])
            g.update(base_lr=muon_lr, base_adam_lr=rates[category], weight_decay=decay[category])
    elif kind == "adamw":
        buckets = defaultdict(list)
        for name, p in named:
            buckets[parameter_group(name, p)].append(p)
        opt = torch.optim.AdamW(
            [
                dict(
                    params=ps,
                    name=category,
                    lr=rates[category],
                    base_lr=rates[category],
                    weight_decay=decay[category],
                )
                for category, ps in buckets.items()
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise ValueError("MF1 optimizer must be adamw or muon")
    owned = [id(p) for g in opt.param_groups for p in g["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for _, p in named}:
        raise ValueError("every trainable parameter must occur exactly once in the optimizer")
    return opt


class QuantileBalance:
    def __init__(self, model, *, bins=256, warmup_updates=200, max_delta=1e-3):
        self.routers = [
            (name, module, QuantileHistogram(module.correction_bias, module.top_k, bins))
            for name, module in model.named_modules()
            if isinstance(module, Router)
        ]
        self.warmup_updates, self.max_delta = warmup_updates, max_delta
        self.updates = 0
        self.ema = {
            name: torch.zeros_like(module.correction_bias) for name, module, _ in self.routers
        }

    @contextlib.contextmanager
    def capture(self, valid, modality=None):
        handles = []
        for _, router, histogram in self.routers:
            if not router.weight.requires_grad:
                continue

            def collect(module, args, output, histogram=histogram):
                ids, _, scores = output
                histogram.add(scores.detach()[valid.flatten()], ids.detach()[valid.flatten()])

            handles.append(router.register_forward_hook(collect))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def update(self):
        self.updates += 1
        records = {}
        for name, router, histogram in self.routers:
            old = router.correction_bias.clone()
            load = histogram.load.float().clone()
            result = histogram.update()
            if result is None:
                continue
            scale = min(1.0, self.updates / self.warmup_updates)
            correction = (router.correction_bias - old).clamp(
                -self.max_delta, self.max_delta
            ) * scale
            router.correction_bias.copy_(old + correction)
            histogram._range()
            self.ema[name].lerp_(load / load.sum().clamp_min(1), 0.05)
            records[name] = dict(
                result,
                min_mean_load=float(load.min() / load.mean().clamp_min(1)),
                max_bias_delta=float(correction.abs().max()),
            )
        return records

    def state_dict(self):
        return dict(
            updates=self.updates,
            ema=self.ema,
            histograms={n: h.state_dict() for n, _, h in self.routers},
            warmup_updates=self.warmup_updates,
            max_delta=self.max_delta,
        )

    def load_state_dict(self, saved):
        if (
            saved["warmup_updates"] != self.warmup_updates
            or saved["max_delta"] != self.max_delta
            or set(saved["histograms"]) != {n for n, _, _ in self.routers}
        ):
            raise ValueError("router balancing configuration/topology differs")
        self.updates = saved["updates"]
        for name, _, hist in self.routers:
            hist.load_state_dict(saved["histograms"][name])
            self.ema[name].copy_(saved["ema"][name])


def parameter_report(model, optimizer=None):
    categories: dict[str, int] = defaultdict(int)
    seen = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        # Parameter identity handles aliases on meta too; real storage identity catches views.
        key = (
            id(p)
            if p.device.type == "meta"
            else (str(p.device), p.untyped_storage().data_ptr(), p.storage_offset(), tuple(p.shape))
        )
        if key in seen:
            continue
        seen.add(key)
        category = (
            "mtp"
            if name.startswith("mtp.")
            else "vision"
            if name.startswith("vision.")
            else "lookup"
            if name.startswith("lookup.")
            else "embedding_and_head"
            if name.startswith(("embed_tokens.", "lm_head."))
            else "gr"
            if "_gr." in name
            else "moe"
            if ".moe." in name
            else "kda"
            if ".core." in name
            else "qsa_mla"
            if any(
                name.startswith(f"layers.{i}.")
                for i, k in enumerate(model.config.attention_schedule)
                if k == "qsa_mla"
            )
            else "csa"
        )
        categories[category] += p.numel()
    total = sum(categories.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    details = dict(
        routed=sum(p.numel() for n, p in model.named_parameters() if ".experts." in n),
        shared=sum(p.numel() for n, p in model.named_parameters() if ".shared." in n),
    )
    main = total - categories["vision"] - categories["mtp"]
    main_routed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith("layers.") and ".experts." in n
    )
    lookup_tables = model.lookup.tables if model.lookup is not None else []
    lookup_rows = sum(table.num_embeddings * table.embedding_dim for table in lookup_tables)
    active = (
        main
        - main_routed
        + main_routed * model.config.num_experts_per_token // model.config.num_experts
        - model.embed_tokens.weight.numel()
        + model.config.hidden_size
        - lookup_rows
        + sum(table.embedding_dim for table in lookup_tables)
    )
    return dict(
        model="MiniFrontier1.0",
        config=asdict(model.config),
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        categories=dict(categories),
        expert_details=details,
        main_without_vision_and_mtp=main,
        embedding=model.embed_tokens.weight.numel(),
        output_head=model.lm_head.weight.numel(),
        active_text_parameter_estimate=active,
        active_count_definition="all main projections/head, top-k routed experts, one embedding and four lookup rows; excludes vision/MTP; not FLOPs or state memory",
        mtp_output_head="shared by passing main lm_head.weight; no duplicate module",
        bf16_weight_bytes=total * 2,
        fp32_weight_bytes=total * 4,
        conservative_state_bytes=total * 16,
        optimizer_parameters=sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        if optimizer
        else None,
        measured_gpu_peak=None,
    )
