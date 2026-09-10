"""V4 Alg. 1: 8+2 hybrid NS, Nesterov momentum, 0.18 RMS rescaling."""

from minifrontier.models.minideepseekv4.upstream_layers import Gate, Linear, RMSNorm
from minifrontier.training.semantic_optim import SemanticOptimizer, whole

HYBRID_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2


def deepseek_parameter_specs(model, *, indexer_adamw=False):
    specs = {}
    for module in model.modules():
        if isinstance(module, (Linear, Gate)):
            specs[id(module.weight)] = (
                whole(module.weight),
                "independent projection/router matrix",
            )
        if isinstance(module, RMSNorm):
            specs[id(module.weight)] = (None, "RMSNorm AdamW")
    for name, p in model.named_parameters():
        if indexer_adamw and ".indexer." in name:
            specs[id(p)] = (None, "explicit pretraining indexer AdamW")
        elif name.startswith(("vision.", "image_")):
            specs[id(p)] = (None, "native vision/aligner/sentinels AdamW pilot")
        elif name in {"embed.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}:
            specs[id(p)] = (None, "embedding or prediction head module AdamW")
        elif name.startswith("mtp.hc_head"):
            specs[id(p)] = (None, "MTP prediction head AdamW")
        elif name.endswith(("hc_attn_fn", "hc_ffn_fn")):
            specs[id(p)] = (whole(p), "dynamic mHC independent matrix")
        elif name.endswith(("hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale")):
            specs[id(p)] = (None, "static mHC bias/gate AdamW")
        elif name.endswith("attn.attn_sink"):
            specs[id(p)] = (None, "local scalar attention-sink AdamW decision")
        elif name.endswith(".ape"):
            specs[id(p)] = (whole(p), "compressor learned position matrix")
    return specs


class MiniDeepSeekV4Optimizer(SemanticOptimizer):
    def __init__(
        self, model, *, lr, adam_lr=None, weight_decay=0.1, eps=1e-20, indexer_adamw=False
    ):
        super().__init__(
            model,
            deepseek_parameter_specs(model, indexer_adamw=indexer_adamw),
            lr=lr,
            adam_lr=lr if adam_lr is None else adam_lr,
            coefficients=HYBRID_COEFFICIENTS,
            scaling=0.18,
            recipe="deepseek-v4-hybrid-ns-v1",
            weight_decay=weight_decay,
            eps=eps,
        )
