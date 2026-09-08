"""Actual supervision counters and accumulation-window normalization."""

from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass
class TokenLedger:
    ce_tokens: int = 0
    input_tokens: int = 0
    response_tokens: int = 0
    optimizer_updates: int = 0
    skipped_windows: int = 0
    image_occurrences: int = 0
    video_examples: int = 0
    video_frames: int = 0
    image_features: int = 0

    def state_dict(self):
        return asdict(self)


def window_counts(batches, device, *, pairwise=False):
    counts = torch.zeros(3, dtype=torch.int64, device=device)
    for x, y in batches:
        counts[0] += y[..., 1:].ne(-100).sum().to(device)
        counts[1] += x.ne(0).sum().to(device)
        counts[2] += x.shape[0] if pairwise else x.ne(0).any(-1).sum().to(device)
    if dist.is_initialized():
        dist.all_reduce(counts)
    return counts


def scaled_ce(mean_loss, local_targets, global_targets, world_size=1):
    """DDP averages gradients: each local CE SUM gets world/global_targets."""
    return mean_loss * local_targets * world_size / max(1, int(global_targets))
