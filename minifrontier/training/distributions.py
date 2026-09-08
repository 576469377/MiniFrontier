"""One legal action support for rollout, old/reference and policy recomputation."""


def forbidden_actions(model):
    config = model.config
    ids = set(getattr(config, "forbidden_action_ids", (0, 1)))
    if getattr(config, "vision_config", None) is not None and hasattr(config, "image_token_id"):
        ids.add(config.image_token_id)
    return tuple(sorted(ids))


def action_logits(logits, vocab_size=None, forbidden_ids=(0, 1)):
    width = logits.shape[-1] if vocab_size is None else vocab_size
    if not 2 < width <= logits.shape[-1]:
        raise ValueError("invalid policy vocabulary")
    result = logits.float().clone()
    result[..., :2] = float("-inf")  # PAD and BOS cannot be actions; EOS can.
    if 2 in forbidden_ids or any(i < 0 or i >= logits.shape[-1] for i in forbidden_ids):
        raise ValueError("invalid forbidden action IDs; real EOS must remain legal")
    result[..., list(forbidden_ids)] = float("-inf")
    result[..., width:] = float("-inf")
    return result
