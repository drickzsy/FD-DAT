"""Epoch-level task-oriented gradient mask from Eqs. (8)-(13)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class GradientMaskBank(nn.Module):
    """Collect detection gradients and update one global channel mask.

    The bank intentionally updates only at epoch boundaries. During warm-up,
    the all-one mask is retained exactly as described by the paper.
    """

    def __init__(self, channels: int = 256, warmup_epochs: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.warmup_epochs = warmup_epochs
        self.register_buffer("mask", torch.ones(channels), persistent=True)
        self.register_buffer("gradient_sum", torch.zeros(channels), persistent=False)
        self.register_buffer("observation_count", torch.zeros((), dtype=torch.long), persistent=False)

    def broadcast(self, reference: Tensor) -> Tensor:
        return self.mask.to(dtype=reference.dtype).view(1, -1, 1, 1)

    def split(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        mask = self.broadcast(latent)
        return latent * mask, latent * (1.0 - mask)

    def collect_from_score(
        self,
        detection_score: Tensor,
        latent_levels: Sequence[Tensor],
        *,
        retain_graph: bool = True,
    ) -> Tensor:
        """Accumulate per-channel |d score / d latent| statistics.

        ``detection_score`` should be the mean ground-truth-class logit from
        foreground queries selected by Hungarian matching. Unmatched
        no-object queries must be excluded by the caller.
        """
        if detection_score.ndim != 0:
            raise ValueError("detection_score must be a scalar")
        if not latent_levels:
            raise ValueError("at least one latent feature level is required")

        gradients = torch.autograd.grad(
            detection_score,
            tuple(latent_levels),
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )
        per_level = []
        for gradient in gradients:
            if gradient.shape[1] != self.channels:
                raise ValueError("all latent levels must use the configured channel count")
            # Eq. (10), additionally averaged over the mini-batch.
            per_level.append(gradient.detach().abs().mean(dim=(0, 2, 3)))
        importance = torch.stack(per_level).mean(dim=0)
        self.gradient_sum.add_(importance)
        self.observation_count.add_(1)
        return importance

    @torch.no_grad()
    def finish_epoch(self, epoch_index: int) -> Tensor:
        """Update the mask for the next epoch and reset accumulators."""
        if epoch_index < self.warmup_epochs:
            self.mask.fill_(1.0)
        elif self.observation_count.item() > 0:
            mean_gradient = self.gradient_sum / self.observation_count
            # ``torch.median`` selects the lower middle value for an even
            # channel count. Linear 0.5 quantile is the conventional median
            # and gives the intended approximately balanced 256-channel split.
            threshold = torch.quantile(mean_gradient, 0.5)
            self.mask.copy_((mean_gradient >= threshold).to(self.mask.dtype))

        updated = self.mask.clone()
        self.gradient_sum.zero_()
        self.observation_count.zero_()
        return updated

    @property
    def di_fraction(self) -> float:
        return float(self.mask.float().mean().item())
