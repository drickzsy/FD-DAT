"""Query-level consistency across decoder layers (Eqs. 24-25)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _jensen_shannon(probabilities: Tensor, reference: Tensor, eps: float) -> Tensor:
    probabilities = probabilities.clamp_min(eps)
    reference = reference.clamp_min(eps)
    mixture = 0.5 * (probabilities + reference)
    first = F.kl_div(mixture.log(), probabilities, reduction="none").sum(dim=-1)
    second = F.kl_div(mixture.log(), reference, reduction="none").sum(dim=-1)
    return 0.5 * (first + second)


def decoder_consistency_loss(
    class_logits: Tensor,
    boxes: Tensor,
    localization_weight: float = 0.3,
    eps: float = 1e-7,
) -> Tensor:
    """Compare every decoder layer with the mean decoder prediction.

    Args:
        class_logits: ``[L, B, Q, K+1]`` pre-softmax logits.
        boxes: ``[L, B, Q, 4]`` normalized ``cx, cy, w, h`` boxes.
    """
    if class_logits.ndim != 4 or boxes.ndim != 4:
        raise ValueError("expected [layers, batch, queries, channels]")
    if class_logits.shape[:3] != boxes.shape[:3] or boxes.shape[-1] != 4:
        raise ValueError("class and box predictions are not aligned")

    probabilities = class_logits.softmax(dim=-1)
    reference_probabilities = probabilities.mean(dim=0)
    reference_boxes = boxes.mean(dim=0)

    classification = torch.stack(
        [_jensen_shannon(layer, reference_probabilities, eps).mean() for layer in probabilities]
    ).mean()
    localization = (boxes - reference_boxes.unsqueeze(0)).abs().sum(dim=-1).mean()
    return classification + localization_weight * localization

