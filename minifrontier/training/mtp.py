"""One additional future-token depth, with explicit boundary and CE denominators."""

import torch


def shifted_batch(
    input_ids, labels, *, vocab_size, image_token_id=7, eos_token_id=2, attention_mask=None
):
    """h_i + Emb(t_i+1) predicts t_i+2; real EOS can be a target, never a bridge.

    Sequences contain one contiguous document. Image and padding positions may
    not be crossed. Return full-length tensors to keep recurrent/HC batch axes.
    """
    valid = input_ids.ne(0) if attention_mask is None else attention_mask.bool()
    textual = valid & input_ids.lt(vocab_size) & input_ids.ne(image_token_id)
    context = torch.zeros_like(valid)
    context[:, :-2] = (
        textual[:, :-2]
        & textual[:, 1:-1]
        & textual[:, 2:]
        & input_ids[:, :-2].ne(eos_token_id)
        & input_ids[:, 1:-1].ne(eos_token_id)
    )
    shifted = torch.full_like(input_ids, eos_token_id)
    shifted[:, :-1] = input_ids[:, 1:]
    shifted = shifted.masked_fill(~context, eos_token_id)
    targets = torch.full_like(input_ids, -100)
    targets[:, :-2] = labels[:, 2:]
    targets = targets.masked_fill(~context, -100)
    return shifted, targets, context


def window_mtp_counts(batches, config, device):
    total = torch.zeros(3, device=device, dtype=torch.int64)
    for x, y in batches:
        _, targets, valid = shifted_batch(
            x, y, vocab_size=config.vocab_size, image_token_id=getattr(config, "image_token_id", 7)
        )
        total[0] += targets.ne(-100).sum().to(device)
        total[1] += valid.sum().to(device)
        total[2] += valid.any(-1).sum().to(device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return total
