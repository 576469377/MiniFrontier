"""Three distinct target-frozen objectives and portable, target-bound draft exports."""

import torch
from torch import nn

from minifrontier.data import sha256
from minifrontier.inference import generate_ids, load_checkpoint
from minifrontier.models.minideepseekv4.dspark import DSparkDraft
from minifrontier.models.minikimik3.draft import KimiDraft
from minifrontier.models.miniqwen4.draft import QwenDraft


def build_draft(family, target):
    return {
        "minikimik3": KimiDraft,
        "miniqwen4": QwenDraft,
        "minideepseekv4": DSparkDraft,
    }[family](target)


class DraftObjective(nn.Module):
    def __init__(self, family, draft):
        super().__init__()
        self.family, self.draft = family, draft

    def forward(self, trajectory, prompt_length, media=None):
        """Ground-truth corpus answers are never supplied; trajectory comes from target."""
        extra = dict(media=media) if media else {}
        generated = trajectory.shape[1] - prompt_length
        if generated < 2:
            zero = sum(p.sum() * 0 for p in self.draft.parameters() if p.requires_grad)
            return zero, 0, 0.0
        # Uniform target-generated anchor locations, including early response positions.
        offset = int(torch.randint(generated - 1, ()))
        prefix_length = prompt_length + offset
        prefix = trajectory[:, :prefix_length]
        if self.family == "minikimik3":
            # Always seven self-fed LK steps, unless an actual EOS terminates an item.
            loss, report = self.draft.unroll(prefix, steps=7, **extra)
        elif self.family == "minideepseekv4":
            anchor = trajectory[:, prefix_length : prefix_length + 1]
            future = trajectory[:, prefix_length + 1 : prefix_length + 6]
            if future.shape[1] < 5:
                # An unfinished response has no fabricated EOS or future labels.
                if int(future[0, -1]) != self.draft.target().config.eos_token_id:
                    prefix_length = max(prompt_length, trajectory.shape[1] - 6)
                    prefix = trajectory[:, :prefix_length]
                    anchor = trajectory[:, prefix_length : prefix_length + 1]
                    future = trajectory[:, prefix_length + 1 : prefix_length + 6]
                    if future.shape[1] < 5:
                        return (
                            sum(p.sum() * 0 for p in self.draft.parameters() if p.requires_grad),
                            0,
                            0.0,
                        )
                else:
                    future = torch.nn.functional.pad(future, (0, 5 - future.shape[1]), value=2)
                    # These storage-only EOS values are masked after the real first EOS.
            loss, report = self.draft.training_block(prefix, anchor, future, **extra)
        else:
            labels = trajectory.clone()
            labels[:, :prompt_length] = -100
            ce, full = self.draft(trajectory, labels, **extra)
            unrolled, short = self.draft.unroll(prefix, steps=3, **extra)
            count = full["positions"] + short["positions"]
            loss = (ce * full["positions"] + unrolled * short["positions"]) / max(1, count)
            report = dict(positions=count)
        return loss, report["positions"], report.get("normalizer", float(report["positions"]))


@torch.no_grad()
def target_trajectory(target, sample, *, rollout_tokens=64):
    """Use only the first user prompt and media wholly before its first answer."""
    from minifrontier.multimodal import TrainingBatch

    if not isinstance(sample, TrainingBatch):
        ids, labels = sample
        sample = TrainingBatch(ids[None], labels[None])
    sample = sample.to(next(target.parameters()).device)
    supervised = sample.labels[0].ne(-100).nonzero().flatten()
    if not supervised.numel() or int(supervised[0]) < 2:
        raise ValueError("draft data requires genuine SFT prompts with masked context")
    length = int(supervised[0])
    media = [s for s in sample.extras.get("media", []) if s["start"] < length]
    # A later turn's image must not enter the first prompt or its frozen target features.
    if any(s["start"] + s.get("sequence_length", s["feature_count"]) > length for s in media):
        raise ValueError("draft prompt cuts through a native image span")
    ids = sample.input_ids[:, :length]
    limit = getattr(
        target.config, "max_position_embeddings", getattr(target.config, "max_seq_len", 0)
    )
    # Reserve seven extra positions for the Kimi unroll at a late target prefix.
    if length + rollout_tokens + 8 > limit:
        raise ValueError("draft prompt, rollout and unroll exceed context; no truncation applied")
    result = generate_ids(
        target, ids, max_new_tokens=rollout_tokens, temperature=1, top_p=1, media=media
    )
    return result, length, media


def load_draft(path, target_checkpoint, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != "minifrontier-draft-v1":
        raise ValueError("unknown draft artifact format")
    if artifact["target_sha256"] != sha256(target_checkpoint):
        raise ValueError("draft belongs to a different frozen target checkpoint")
    target, tokenizer, meta = load_checkpoint(target_checkpoint, device)
    if artifact["model_name"] != meta["model_name"]:
        raise ValueError("draft family differs from target")
    draft = build_draft(meta["model_name"], target)
    draft.load_state_dict(artifact["draft"], strict=True)
    return target, draft.eval(), tokenizer, meta
