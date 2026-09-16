"""Detector integration contract and one FD-DAT optimization step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fd_dat.losses.consistency import decoder_consistency_loss
from fd_dat.models.feature_module import FDDATFeatureModule


@dataclass
class DetectorPredictions:
    """Normalized outputs expected from a Deformable DETR adapter."""

    class_logits: Tensor  # [decoder_layers, batch, queries, classes + 1]
    boxes: Tensor  # [decoder_layers, batch, queries, 4]
    domain_logits: Tensor | None = None  # [batch]
    raw: Any = None


class DetectorAdapter(Protocol):
    """The only interface required from an upstream detector."""

    def extract_projected_features(self, images: Tensor) -> list[Tensor]: ...

    def detect_disentangled(
        self,
        di_features: list[Tensor],
        ds_features: list[Tensor],
        *,
        domain: str,
    ) -> DetectorPredictions: ...

    def detection_loss(
        self,
        predictions: DetectorPredictions,
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]: ...

    def match(
        self,
        predictions: DetectorPredictions,
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]: ...


@dataclass(frozen=True)
class ObjectiveWeights:
    cdd: float = 0.8
    regularization: float = 0.6
    consistency: float = 0.2
    adversarial: float = 0.1
    localization_consistency: float = 0.3


class FDDATTrainingCore(nn.Module):
    """Compose the paper-specific modules around a detector adapter."""

    def __init__(
        self,
        detector: DetectorAdapter,
        feature_module: FDDATFeatureModule,
        weights: ObjectiveWeights = ObjectiveWeights(),
    ) -> None:
        super().__init__()
        self.detector = detector  # type: ignore[assignment]
        self.feature_module = feature_module
        self.weights = weights

    def forward(
        self,
        source_images: Tensor,
        source_targets: list[dict[str, Tensor]],
        target_images: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
        source_features = self.detector.extract_projected_features(source_images)
        target_features = self.detector.extract_projected_features(target_images)
        aligned = self.feature_module(source_features, target_features)

        source_predictions = self.detector.detect_disentangled(
            aligned.source_di, aligned.source_ds, domain="source"
        )
        target_predictions = self.detector.detect_disentangled(
            aligned.target_di, aligned.target_ds, domain="target"
        )
        detection_terms = self.detector.detection_loss(source_predictions, source_targets)
        detection_loss = torch.stack(tuple(detection_terms.values())).sum()

        source_consistency = decoder_consistency_loss(
            source_predictions.class_logits,
            source_predictions.boxes,
            localization_weight=self.weights.localization_consistency,
        )
        target_consistency = decoder_consistency_loss(
            target_predictions.class_logits,
            target_predictions.boxes,
            localization_weight=self.weights.localization_consistency,
        )
        consistency = 0.5 * (source_consistency + target_consistency)

        adversarial = detection_loss.new_zeros(())
        if (
            source_predictions.domain_logits is not None
            and target_predictions.domain_logits is not None
        ):
            source_domain = torch.zeros_like(source_predictions.domain_logits)
            target_domain = torch.ones_like(target_predictions.domain_logits)
            adversarial = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    source_predictions.domain_logits, source_domain
                )
                + F.binary_cross_entropy_with_logits(
                    target_predictions.domain_logits, target_domain
                )
            )

        losses = {
            **detection_terms,
            "loss_cdd": aligned.cdd_loss,
            "loss_regularization": aligned.regularization_loss,
            "loss_consistency": consistency,
            "loss_adversarial": adversarial,
        }
        total = (
            detection_loss
            + self.weights.cdd * aligned.cdd_loss
            + self.weights.regularization * aligned.regularization_loss
            + self.weights.consistency * consistency
            + self.weights.adversarial * adversarial
        )
        context = {
            "aligned": aligned,
            "source_predictions": source_predictions,
            "target_predictions": target_predictions,
        }
        return total, losses, context

    def collect_mask_statistics(
        self,
        source_targets: list[dict[str, Tensor]],
        context: dict[str, Any],
        epoch_index: int,
    ) -> Tensor | None:
        """Collect Eq. (10) after matching, before ``total_loss.backward()``."""
        bank = self.feature_module.mask_bank
        if epoch_index < bank.warmup_epochs:
            return None
        predictions: DetectorPredictions = context["source_predictions"]
        matches = self.detector.match(predictions, source_targets)
        final_logits = predictions.class_logits[-1]
        selected = []
        for batch_index, (query_indices, target_indices) in enumerate(matches):
            if query_indices.numel() == 0:
                continue
            labels = source_targets[batch_index]["labels"][target_indices]
            selected.append(final_logits[batch_index, query_indices, labels])
        if not selected:
            return None
        detection_score = torch.cat(selected).mean()
        return bank.collect_from_score(
            detection_score,
            context["aligned"].source_latent,
            retain_graph=True,
        )

