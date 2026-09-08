"""Educational post-training objectives, independent of model backbone.

DPO: Rafailov et al. 2023 eq. 7; completion sums, frozen reference policy.
GRPO: grouped verifiable rewards and clipped policy ratios; no value network.
MOPD: Kimi K3 report eq. 15 sampled-token clipped log-ratio reward.
These are local small-scale recipes, not undisclosed official data/hyperparameters.
"""

import torch
import torch.nn.functional as F


def token_log_probs(logits, labels, *, actions=False, vocab_size=None, forbidden_ids=(0, 1)):
    if actions:
        from minifrontier.training.distributions import action_logits

        logits = action_logits(logits, vocab_size, forbidden_ids)
    targets = labels[:, 1:]
    mask = targets != -100
    selected = (
        logits[:, :-1]
        .float()
        .log_softmax(-1)
        .gather(-1, targets.masked_fill(~mask, 2).unsqueeze(-1))
        .squeeze(-1)
    )
    return selected.masked_fill(~mask, 0), mask


def dpo_loss(policy_logits, reference_logits, labels, beta=0.1):
    policy, mask = token_log_probs(policy_logits, labels)
    reference, _ = token_log_probs(reference_logits, labels)
    if not mask.any(-1).all() or policy.shape[0] % 2:
        raise ValueError("DPO needs paired chosen/rejected completions with supervised tokens")
    log_ratio = (policy.sum(-1) - reference.detach().sum(-1)).view(-1, 2)
    margin = beta * (log_ratio[:, 0] - log_ratio[:, 1])
    return -F.logsigmoid(margin).mean(), (margin.detach() > 0).float().mean()


def grouped_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GRPO requires complete groups of at least two responses")
    groups = rewards.reshape(-1, group_size)
    return (
        (groups - groups.mean(-1, keepdim=True))
        / groups.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    ).flatten()


def policy_loss(logp, old_logp, advantages, mask, *, clip=0.2, reference_logp=None, kl_coef=0.01):
    ratio = (logp - old_logp.detach()).exp()
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    advantage = advantages.detach()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if reference_logp is not None:
        delta = reference_logp.detach() - logp
        loss = loss + kl_coef * (delta.exp() - delta - 1)
    per_response = (loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_response.sum() / mask.any(-1).sum().clamp_min(1)


def mopd_advantages(teacher_logp, student_logp, reward_clip=5.0):
    return (teacher_logp - student_logp).detach().clamp(-reward_clip, reward_clip)
