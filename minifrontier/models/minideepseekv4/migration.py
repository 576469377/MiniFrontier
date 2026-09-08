"""Explicit Text-v2 -> native Vision-v1 migration; every inherited key is checked."""

from dataclasses import asdict


def migrate_text_state(model, text_config, state):
    if model.vision is None:
        raise ValueError("destination must have the native visual tower")
    source = asdict(type(model.config)(**text_config))
    destination = asdict(model.config)
    source.pop("vision_config", None)
    destination.pop("vision_config", None)
    if source != destination:
        raise ValueError(
            "migration cannot silently change text dimensions, theta, scale or epsilon"
        )
    current = model.state_dict()
    added = set(current) - set(state)
    if set(state) - set(current):
        raise ValueError("unexpected source text keys")
    for key in added:
        if not key.startswith(("vision.", "image_")) and not key.endswith(".gate.bias_vl"):
            raise ValueError(f"unapproved new migration key: {key}")
    rows = []
    for key, value in current.items():
        inherited = key in state
        if inherited:
            if state[key].shape != value.shape or state[key].dtype != value.dtype:
                raise ValueError(f"migration shape/dtype mismatch: {key}")
            current[key] = state[key]
        rows.append(
            dict(
                key=key,
                shape=list(value.shape),
                dtype=str(value.dtype),
                action="inherited" if inherited else "initialized",
            )
        )
    model.load_state_dict(current, strict=True)
    return dict(
        schema_version=1,
        text_equivalence="same config and inherited tensors; validate forward separately",
        keys=rows,
    )


def configure_visual_warmup(model):
    if model.vision is None:
        raise ValueError("visual warmup needs a visual tower")
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(name.startswith(("vision.", "image_")))
    model.zero_grad(set_to_none=True)
