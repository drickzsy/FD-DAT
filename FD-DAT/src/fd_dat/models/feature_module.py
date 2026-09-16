"""Feature disentanglement and distribution-alignment core of FD-DAT."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from fd_dat.losses.mmd import MultiKernelMMD
from .gradient_mask import GradientMaskBank
from .invertible import InvertibleFeatureMapper


@dataclass
class FeatureAlignmentOutput:
    source_latent: list[Tensor]
    target_latent: list[Tensor]
    source_di: list[Tensor]
    source_ds: list[Tensor]
    target_di: list[Tensor]
    target_ds: list[Tensor]
    cdd_loss: Tensor
    regularization_loss: Tensor


class FDDATFeatureModule(nn.Module):
    """Apply shared INN mapping, gradient-mask splitting, and MK-MMD losses."""

    def __init__(
        self,
        channels: int = 256,
        inn_hidden_channels: int = 256,
        inn_blocks: int = 4,
        gamma: float = 1.0,
        permutation_seed: int = 3407,
        warmup_epochs: int = 5,
        mmd_sigma_squared: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    ) -> None:
        super().__init__()
        self.mapper = InvertibleFeatureMapper(
            channels=channels,
            hidden_channels=inn_hidden_channels,
            num_blocks=inn_blocks,
            gamma=gamma,
            permutation_seed=permutation_seed,
        )
        self.mask_bank = GradientMaskBank(channels=channels, warmup_epochs=warmup_epochs)
        self.mmd = MultiKernelMMD(mmd_sigma_squared)

    @staticmethod
    def _validate_levels(source: Sequence[Tensor], target: Sequence[Tensor]) -> None:
        if len(source) == 0 or len(source) != len(target):
            raise ValueError("source and target must contain the same non-zero feature levels")
        for source_level, target_level in zip(source, target):
            if source_level.shape != target_level.shape:
                raise ValueError(
                    "the scaffold pairs source and target levels one-to-one; "
                    "use equal batch and spatial shapes"
                )

    def forward(
        self,
        source_features: Sequence[Tensor],
        target_features: Sequence[Tensor],
    ) -> FeatureAlignmentOutput:
        self._validate_levels(source_features, target_features)
        source_latent = [self.mapper(level) for level in source_features]
        target_latent = [self.mapper(level) for level in target_features]

        source_split = [self.mask_bank.split(level) for level in source_latent]
        target_split = [self.mask_bank.split(level) for level in target_latent]
        source_di, source_ds = map(list, zip(*source_split))
        target_di, target_ds = map(list, zip(*target_split))

        cdd_terms = []
        regularization_terms = []
        for source, target, source_h, target_h, source_h_di, target_h_di in zip(
            source_features,
            target_features,
            source_latent,
            target_latent,
            source_di,
            target_di,
        ):
            mask = self.mask_bank.broadcast(source_h).bool()
            # Eq. (14), implemented as coordinate-preserving replacement.
            # Concatenating active channels would destroy the INN channel order.
            swapped_source_h = torch.where(mask, target_h, source_h)
            swapped_target_h = torch.where(mask, source_h, target_h)
            reconstructed_source = self.mapper.inverse(swapped_source_h)
            reconstructed_target = self.mapper.inverse(swapped_target_h)
            cdd_terms.extend(
                (
                    self.mmd(source, reconstructed_source),
                    self.mmd(target, reconstructed_target),
                )
            )
            regularization_terms.append(self.mmd(source_h_di, target_h_di))

        return FeatureAlignmentOutput(
            source_latent=source_latent,
            target_latent=target_latent,
            source_di=source_di,
            source_ds=source_ds,
            target_di=target_di,
            target_ds=target_ds,
            cdd_loss=torch.stack(cdd_terms).mean(),
            regularization_loss=torch.stack(regularization_terms).mean(),
        )

