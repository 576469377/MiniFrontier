"""Shared TensorBoard presentation; JSON logs retain all original run events."""

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


def training_scalars(values):
    """Group learning/validation/performance scalars without plotting operational events.

    MF1 train rows have no event field. Native trainers flatten per-domain
    validation values as well as retaining a nested copy; emit each only once.
    Throughput names retain their original denominators across training stages.
    """
    if values.get("event", "train") not in {"train", "validation", "eval"}:
        return {}
    validation = values.get("event") in {"validation", "eval"}
    scope = "phase_end_" if values.get("evaluation_scope") == "phase_end" else ""
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
                tag = f"eval/{scope}{'lm_loss' if key == 'nll' else key}"
            elif key == "peak_allocated_mib":
                tag, value = "perf/peak_allocated_gib", value / 1024
            elif key in PERFORMANCE_METRICS:
                tag = f"perf/{key}"
            else:
                tag = f"train/{'lm_loss' if key == 'train_lm_loss' else key}"
            result[tag] = value
    return result


def mf1_scalars(values):
    """Compatibility entry point for existing MF1 scripts and notebooks."""
    return training_scalars(values)
