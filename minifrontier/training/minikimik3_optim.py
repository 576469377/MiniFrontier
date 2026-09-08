"""Kimi-specific Per-Head Muon groups; special matrix choices are documented.

K3 specifies per-head NS but does not publish a complete mini optimizer. Five
quintic NS iterations and 0.2*sqrt(max shape) scaling are explicit local choices,
requiring the strategy's equal-token AdamW comparison; no Qwen Polar Express or
DeepSeek 8+2 iteration recipe is claimed here.
"""

from torch import nn

from minifrontier.models.minikimik3.kernels import FusedRMSNormGated
from minifrontier.models.minikimik3.upstream_layers import (
    KimiDeltaAttention,
    KimiMLAAttention,
    KimiMoEGate,
    KimiRMSNorm,
)
from minifrontier.training.semantic_optim import SemanticOptimizer, head_rows, whole


def kimi_parameter_specs(model):
    specs = {}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            specs[id(module.weight)] = (
                whole(module.weight),
                "independent dense/latent/expert matrix",
            )
            if module.bias is not None:
                specs[id(module.bias)] = (None, "bias AdamW")
        if isinstance(module, (nn.Embedding, KimiRMSNorm, KimiMoEGate, FusedRMSNormGated)):
            specs[id(module.weight)] = (None, "embedding/norm/router AdamW")
        if isinstance(module, nn.Conv1d):
            specs[id(module.weight)] = (None, "local short-convolution AdamW choice")
            if module.bias is not None:
                specs[id(module.bias)] = (None, "convolution bias AdamW")
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            for key in ("q_proj", "k_proj"):
                p = getattr(module, key).weight
                specs[id(p)] = (
                    head_rows(p, [module.head_k_dim] * module.num_k_heads),
                    "KDA Q/K per head",
                )
            p = module.v_proj.weight
            specs[id(p)] = (head_rows(p, [module.head_dim] * module.num_heads), "KDA V per head")
        if isinstance(module, KimiMLAAttention):
            q = module.q_b_proj.weight if module.q_lora_rank is not None else module.q_proj.weight
            specs[id(q)] = (
                head_rows(q, [module.q_head_dim] * module.num_heads),
                "MLA Q up-projection per head",
            )
            kv = module.kv_b_proj.weight
            widths = [module.qk_nope_head_dim, module.v_head_dim] * module.num_heads
            specs[id(kv)] = (head_rows(kv, widths), "MLA separate K/V head rows")
            p = module.kv_a_proj_with_mqa.weight
            specs[id(p)] = (
                head_rows(p, [module.kv_lora_rank, module.qk_rope_head_dim]),
                "independent latent down projection and shared key component",
            )
    for name, p in model.named_parameters():
        if name.startswith("vision."):
            specs[id(p)] = (None, "native vision and merger AdamW pilot choice")
            continue
        if name == "lm_head.weight" or "res_proj.weight" in name:
            specs[id(p)] = (None, "LM head / local AttnRes query AdamW choice")
        if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            specs[id(p)] = (None, "KDA scalar/decay or frozen QB bias")
        if ".self_attn." in name and any(
            f".{key}." in name
            for key in ("b_proj", "f_a_proj", "f_b_proj", "g_proj", "g_a_proj", "g_b_proj")
        ):
            specs[id(p)] = (None, "local KDA/MLA gate AdamW choice")
    return specs


class MiniKimiK3Optimizer(SemanticOptimizer):
    def __init__(self, model, *, lr, adam_lr, weight_decay=0.1, eps=1e-8):
        super().__init__(
            model,
            kimi_parameter_specs(model),
            lr=lr,
            adam_lr=adam_lr,
            coefficients=((3.4445, -4.7750, 2.0315),) * 5,
            scaling=0.2,
            recipe="kimi-per-head-local-ns5-v1",
            weight_decay=weight_decay,
            eps=eps,
        )
