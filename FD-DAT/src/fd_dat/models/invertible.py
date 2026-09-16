"""Invertible 1x1-convolutional coupling blocks used by FD-DAT.

The implementation follows Eqs. (1)-(7) of the paper: every block splits
the channels in half, applies an affine coupling transform, and then uses a
fixed channel permutation. The same mapper can be reused for all four DETR
feature levels because every projected level has 256 channels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _pointwise_subnet(in_channels: int, hidden_channels: int) -> nn.Sequential:
    network = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
    )
    # Starting near the identity is materially more stable for a detector
    # initialized from a pretrained backbone.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class AffineCouplingBlock(nn.Module):
    """A reversible affine coupling block with a fixed permutation."""

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 256,
        gamma: float = 1.0,
        permutation_seed: int = 0,
    ) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even for an equal coupling split")
        half = channels // 2
        self.channels = channels
        self.gamma = float(gamma)
        self.scale_net = _pointwise_subnet(half, hidden_channels)
        self.translation_net = _pointwise_subnet(half, hidden_channels)

        generator = torch.Generator().manual_seed(permutation_seed)
        permutation = torch.randperm(channels, generator=generator)
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(channels)
        self.register_buffer("permutation", permutation, persistent=True)
        self.register_buffer("inverse_permutation", inverse_permutation, persistent=True)

    def _affine_parameters(self, first_half: Tensor) -> tuple[Tensor, Tensor]:
        log_scale = self.gamma * torch.tanh(self.scale_net(first_half))
        translation = self.translation_net(first_half)
        return log_scale, translation

    def forward(self, inputs: Tensor, reverse: bool = False) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, H, W], got {tuple(inputs.shape)}"
            )
        if reverse:
            unpermuted = inputs[:, self.inverse_permutation]
            first, transformed = unpermuted.chunk(2, dim=1)
            log_scale, translation = self._affine_parameters(first)
            second = (transformed - translation) * torch.exp(-log_scale)
            return torch.cat((first, second), dim=1)

        first, second = inputs.chunk(2, dim=1)
        log_scale, translation = self._affine_parameters(first)
        transformed = second * torch.exp(log_scale) + translation
        coupled = torch.cat((first, transformed), dim=1)
        return coupled[:, self.permutation]


class InvertibleFeatureMapper(nn.Module):
    """A stack of affine coupling blocks shared across feature levels."""

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 256,
        num_blocks: int = 4,
        gamma: float = 1.0,
        permutation_seed: int = 3407,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.channels = channels
        self.blocks = nn.ModuleList(
            AffineCouplingBlock(
                channels=channels,
                hidden_channels=hidden_channels,
                gamma=gamma,
                permutation_seed=permutation_seed + block_index,
            )
            for block_index in range(num_blocks)
        )

    def forward(self, inputs: Tensor, reverse: bool = False) -> Tensor:
        blocks = reversed(self.blocks) if reverse else self.blocks
        outputs = inputs
        for block in blocks:
            outputs = block(outputs, reverse=reverse)
        return outputs

    def inverse(self, latent: Tensor) -> Tensor:
        return self.forward(latent, reverse=True)

