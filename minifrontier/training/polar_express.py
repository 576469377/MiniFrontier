"""Polar Express numerical primitive shared by the MiniQwen4 optimizer."""

from __future__ import annotations

from torch import Tensor

POLAR_EXPRESS_8_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (8.237312490495555, -23.157747414558187, 16.6805684114459),
    (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
    (3.926347992254648, -2.854746803476524, 0.531802242289498),
    (3.298218713308519, -2.42454198102671, 0.48632008358844137),
    (2.297036943455259, -1.636625581259032, 0.4002628455953629),
    (1.8763805351440375, -1.2347896577722182, 0.35891887501668146),
    (1.8564423485504604, -1.2132449880713188, 0.35680034877168715),
    (1.8564366929791287, -1.2132397423614751, 0.35680064129754985),
)


def polar_express_orthogonalize(matrix: Tensor, steps: int = 8, eps: float = 1e-14) -> Tensor:
    """Approximate a matrix polar factor with Polar Express' step schedule.

    Qwen3.8-Flash-Next specifies eight iterations and ``1e-14`` as the
    numerical-stability constant in the preceding Frobenius normalization.
    The reference path stays in float32 so it is deterministic on CPU as well
    as on the four-3090 training target.
    """

    if matrix.ndim not in (2, 3):
        raise ValueError("Polar Express expects a matrix or a batch of independent matrices")
    if not 1 <= steps <= len(POLAR_EXPRESS_8_COEFFICIENTS):
        raise ValueError(
            f"Polar Express steps must be between 1 and {len(POLAR_EXPRESS_8_COEFFICIENTS)}"
        )
    if eps <= 0:
        raise ValueError("Polar Express normalization epsilon must be positive")
    transposed = matrix.size(-2) > matrix.size(-1)
    update = matrix.float().transpose(-2, -1) if transposed else matrix.float()
    # Normalize each semantic matrix independently, never across the batch.
    norm = update.norm() if matrix.ndim == 2 else update.norm(dim=(-2, -1), keepdim=True)
    update = update / (norm + eps)
    for a, b, c in POLAR_EXPRESS_8_COEFFICIENTS[:steps]:
        gram = update @ update.transpose(-2, -1)
        update = a * update + (b * gram + c * (gram @ gram)) @ update
    return update.transpose(-2, -1) if transposed else update
