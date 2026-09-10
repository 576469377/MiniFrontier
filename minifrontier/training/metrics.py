"""TensorBoard presentation of MF1's immutable JSON metric records."""

PERFORMANCE_METRICS = {
    "step_seconds",
    "ce_per_second",
    "input_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "data_preparation_seconds",
    "ce_fraction",
    "padding_fraction",
}


def mf1_scalars(values):
    """Use three chart groups and keep validation denominators separate from training."""
    validation = values.get("event") in {"validation", "eval"}
    scope = "phase_end_" if values.get("evaluation_scope") == "phase_end" else ""
    result = {}
    for key, value in values.items():
        if key in {"step", "optimizer_updates", "event"}:
            continue
        if validation and key in {"per_domain", "black_media_nll", "domain_ce", "black_media_ce"}:
            prefix = {
                "per_domain": "lm_loss",
                "black_media_nll": "black_media_lm_loss",
                "domain_ce": "ce_tokens",
                "black_media_ce": "black_media_ce_tokens",
            }[key]
            for domain, loss in value.items():
                result[f"eval/{scope}{prefix}_{domain}"] = loss
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if validation:
                tag = f"eval/{scope}{'lm_loss' if key == 'nll' else key}"
            elif key in PERFORMANCE_METRICS:
                tag = f"perf/{key}"
            else:
                tag = f"train/{'lm_loss' if key == 'train_lm_loss' else key}"
            result[tag] = value
    return result
