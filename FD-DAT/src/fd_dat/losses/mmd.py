"""Squared empirical multi-kernel maximum mean discrepancy."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def _as_vectors(features: Tensor) -> Tensor:
    if features.ndim == 4:
        return features.mean(dim=(2, 3))
    if features.ndim == 2:
        return features
    raise ValueError("features must be [B, C] or [B, C, H, W]")


class MultiKernelMMD(nn.Module):
    """Unbiased MK-MMD with the five bandwidths reported in the paper."""

    def __init__(self, sigma_squared: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        bandwidths = torch.as_tensor(tuple(sigma_squared), dtype=torch.float32)
        if bandwidths.numel() == 0 or torch.any(bandwidths <= 0):
            raise ValueError("sigma_squared must contain positive values")
        self.register_buffer("sigma_squared", bandwidths, persistent=True)

    def _kernel(self, left: Tensor, right: Tensor) -> Tensor:
        squared_distance = torch.cdist(left, right, p=2).square().unsqueeze(-1)
        bandwidths = self.sigma_squared.to(dtype=left.dtype, device=left.device)
        kernels = torch.exp(-squared_distance / (2.0 * bandwidths))
        return kernels.mean(dim=-1)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        first = _as_vectors(first)
        second = _as_vectors(second)
        if first.shape[1] != second.shape[1]:
            raise ValueError("feature dimensions must match")
        m, n = first.shape[0], second.shape[0]
        if m < 2 or n < 2:
            raise ValueError("unbiased MK-MMD requires at least two samples per set")

        kernel_xx = self._kernel(first, first)
        kernel_yy = self._kernel(second, second)
        kernel_xy = self._kernel(first, second)
        within_x = (kernel_xx.sum() - kernel_xx.diagonal().sum()) / (m * (m - 1))
        within_y = (kernel_yy.sum() - kernel_yy.diagonal().sum()) / (n * (n - 1))
        cross = kernel_xy.mean()
        return within_x + within_y - 2.0 * cross

