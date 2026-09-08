"""Explicit, audited changes between stages; exact resume never permits migration."""

from dataclasses import asdict

import torch


def load_previous(model, saved, *, transition="exact", stage="pretrain", resume=False):
    source = asdict(type(model.config)(**saved["config"]))
    target = asdict(model.config)
    changed = {key for key in target if source[key] != target[key]}
    report = dict(transition=transition, changed=sorted(changed))
    if resume or transition == "exact":
        if changed:
            raise ValueError("checkpoint capacity differs from requested configuration")
    elif transition == "qat":
        if (
            changed != {"qat_scheme"}
            or source["qat_scheme"] != "bf16"
            or target["qat_scheme"] == "bf16"
            or stage != "sft"
        ):
            raise ValueError("QAT transition only enables the model's QAT recipe entering SFT")
    elif transition == "mtp-weight":
        if changed != {"mtp_loss_coef"} or not 0 <= target["mtp_loss_coef"] <= 1:
            raise ValueError("MTP-weight transition may only change its loss coefficient")
    elif transition == "text-to-vision":
        if saved["model_name"] != "minideepseekv4" or stage != "pretrain":
            raise ValueError("native Text-to-Vision is a DeepSeek continued-pretraining transition")
        from minifrontier.models.minideepseekv4.migration import migrate_text_state

        report["migration"] = migrate_text_state(model, saved["config"], saved["model"])
        return report
    else:
        raise ValueError("unknown initialization transition")
    model.load_state_dict(saved["model"], strict=True)
    return report


def set_visual_rates(optimizer, model, *, vision_lr=None, projector_lr=None):
    """Preserve existing semantic groups and algorithm; split Adam groups by module."""
    if vision_lr is None and projector_lr is None:
        return
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    matched = set()
    for group in optimizer.param_groups:
        divided: dict[float | None, list[torch.Tensor]] = {}
        for parameter in group["params"]:
            name = names[id(parameter)]
            is_projector = name.startswith("image_") or name.startswith(
                ("vision.aligner.", "vision.merger.")
            )
            rate = (
                projector_lr if is_projector else vision_lr if name.startswith("vision.") else None
            )
            if rate is not None:
                matched.add("projector" if is_projector else "vision")
            divided.setdefault(rate, []).append(parameter)
        for rate, parameters in divided.items():
            item = dict(group, params=parameters)
            if rate is not None:
                item["visual_base_lr"] = rate
            groups.append(item)
    for kind, rate in (("vision", vision_lr), ("projector", projector_lr)):
        if rate is not None and kind not in matched:
            raise ValueError(f"no {kind} trainable parameters for its requested learning rate")
    optimizer.param_groups = groups
