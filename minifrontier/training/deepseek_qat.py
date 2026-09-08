"""DeepSeek MXFP4 experts plus rotated FP4 indexer Q/K and BF16 scores."""

import torch

from .mx_quant import configure_experts, fake_mx, hadamard


def configure(model):
    from minifrontier.models.minideepseekv4.attention import Indexer

    experts = configure_experts(model)
    indexers = []
    for name, module in model.named_modules():
        if isinstance(module, Indexer):
            if module.head_dim % 32:
                raise ValueError("indexer FP4 group needs head width divisible by 32")
            module.qat_enabled = True
            indexers.append(name)
    return dict(
        experts=experts,
        indexers=indexers,
        block_size=32,
        scale="E8M0-ceil-amax",
        index_qk="Hadamard-MXFP4",
        index_scores="BF16",
        master="FP32",
        compute="BF16",
        gradient="STE",
        native_low_precision_acceleration=False,
    )


def index_scores(q, k, weights):
    q = fake_mx(hadamard(q), bits=4).to(torch.bfloat16)
    k = fake_mx(hadamard(k), bits=4).to(torch.bfloat16)
    # Explicit BF16 boundaries agree in training, teacher scoring and cache decode.
    dot = torch.einsum("bthd,bcd->bthc", q.float(), k.float()).to(torch.bfloat16)
    weighted = dot.relu() * weights.to(torch.bfloat16).unsqueeze(-1)
    return weighted.sum(2).to(torch.bfloat16).float()
