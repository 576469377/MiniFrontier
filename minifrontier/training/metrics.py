"""Shared TensorBoard presentation; JSON logs retain all original run events."""

import math

PERFORMANCE_METRICS = {
    "step_seconds",
    "ce_per_second",
    "input_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "data_preparation_seconds",
    "optimizer_seconds",
    "tokens_per_second",
    "device_free_gib",
    "ce_fraction",
    "padding_fraction",
}


def training_scalars(values, *, stage=None):
    """Group learning/validation/performance scalars without plotting operational events.

    MF1 train rows have no event field. Native trainers flatten per-domain
    validation values as well as retaining a nested copy; emit each only once.
    Pass the recorded run stage to normalize native CE throughput. An unknown
    stage retains its original throughput name; ledger totals cannot establish
    the denominator of an individual update.
    """
    if values.get("event", "train") not in {"train", "validation", "eval"}:
        return {}
    validation = values.get("event") in {"validation", "eval"}
    scope = "phase_end_" if values.get("evaluation_scope") == "phase_end" else ""
    if not validation:
        # Native counters are nested; explicit top-level values take precedence.
        values = {**values.get("token_ledger", {}), **values}
        inputs, seconds = values.get("input_batch_actual"), values.get("step_seconds")
        if "input_per_second" not in values and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in (inputs, seconds)
        ):
            values["input_per_second"] = inputs / seconds
    result = {}
    for key, value in values.items():
        if key in {"step", "optimizer_updates", "event", "data_offset"}:
            continue
        if validation and key == "reward" and values.get("loss_denominator") == "ce_tokens":
            continue  # The generic evaluator writes a placeholder reward outside RL/DPO.
        if validation and key in {"per_domain", "black_media_nll", "domain_ce", "black_media_ce"}:
            prefix = {
                "per_domain": "lm_loss",
                "black_media_nll": "black_media_lm_loss",
                "domain_ce": "ce_tokens",
                "black_media_ce": "black_media_ce_tokens",
            }[key]
            for domain, loss in value.items():
                if isinstance(loss, dict):
                    for metric in ("lm_loss", "ce_tokens"):
                        if metric in loss:
                            result[f"eval/{scope}{metric}_{domain}"] = loss[metric]
                else:
                    result[f"eval/{scope}{prefix}_{domain}"] = loss
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if validation:
                aliases = {"nll": "lm_loss"}
                if values.get("loss_denominator") == "ce_tokens":
                    aliases.update(supervised_tokens="ce_tokens", seconds="duration_seconds")
                normalized = aliases.get(key, key)
                if normalized != key and normalized in values:
                    continue
                tag = f"eval/{scope}{normalized}"
            elif key == "peak_allocated_mib":
                tag, value = "perf/peak_allocated_gib", value / 1024
            elif key == "tokens_per_second" and stage in {"pretrain", "sft", "sparse_cpt"}:
                # train.py counts shifted CE labels for these stages. Indexer,
                # preference, and rollout updates retain their original name.
                if "ce_per_second" in values:
                    continue
                tag = "perf/ce_per_second"
            elif key in PERFORMANCE_METRICS:
                tag = f"perf/{key}"
            else:
                tag = f"train/{'lm_loss' if key == 'train_lm_loss' else key}"
            result[tag] = value
    return result


def mf1_scalars(values):
    """Compatibility entry point for existing MF1 scripts and notebooks."""
    return training_scalars(values)
