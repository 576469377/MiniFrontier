"""Small shared LM interface and strict batch validation."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CausalLMOutput:
    logits: Tensor | None
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    aux_loss: Tensor | None = None
    indexer_loss: Tensor | None = None
    hidden_states: Tensor | None = None
    multistream_hidden: Tensor | None = None
    mtp_loss: Tensor | None = None
    mtp_tokens: int = 0
    mtp_aux_loss: Tensor | None = None
    mtp_aux_count: int = 0


def insert_media_embeddings(input_ids, embeddings, media, encoder, placeholder_id):
    """Strict one-to-one feature insertion; raw pixels never enter the word table."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    result = embeddings.clone()
    for span in media or []:
        batch, start = span["batch_index"], span["start"]
        features = encoder(span["patches"], span["grid_thw"])
        features = torch.cat(tuple(features), dim=0)
        end = start + features.shape[0]
        if not 0 <= batch < input_ids.shape[0] or start < 0 or end > input_ids.shape[1]:
            raise ValueError("media span exceeds the language sequence")
        if mask[batch, start:end].any() or not input_ids[batch, start:end].eq(placeholder_id).all():
            raise ValueError("overlapping media or feature/placeholder mismatch")
        if span.get("feature_count", features.shape[0]) != features.shape[0]:
            raise ValueError("processor and vision encoder feature counts disagree")
        result[batch, start:end] = features.to(result.dtype)
        mask[batch, start:end] = True
    if not torch.equal(input_ids.eq(placeholder_id), mask):
        raise ValueError("image placeholder has missing or extra media features")
    return result, mask


def validate_batch(input_ids, config, attention_mask=None, labels=None):
    if input_ids.ndim != 2 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be nonempty int64 [batch, sequence]")
    limit = getattr(config, "max_position_embeddings", getattr(config, "max_seq_len", 0))
    if input_ids.shape[1] > limit:
        raise ValueError("sequence exceeds configured context")
    if ((input_ids < 0) | (input_ids >= config.vocab_size)).any():
        raise ValueError("token IDs outside vocabulary")
    if attention_mask is not None:
        if (
            attention_mask.shape != input_ids.shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("attention mask must be binary and match inputs")
        if (attention_mask[:, 1:] > attention_mask[:, :-1]).any():
            raise ValueError("only right padding is supported")
    if labels is not None:
        if labels.shape != input_ids.shape or labels.dtype != torch.long:
            raise ValueError("labels must be int64 and match inputs")
        if ((labels != -100) & ((labels < 0) | (labels >= config.vocab_size))).any():
            raise ValueError("labels outside vocabulary")
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(), -100)
    return labels
