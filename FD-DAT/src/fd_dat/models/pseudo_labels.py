"""Source-centroid initialized spherical K-means pseudo labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PseudoLabelResult:
    labels: Tensor
    accepted: Tensor
    cosine_distance: Tensor
    centroids: Tensor


class SphericalKMeansPseudoLabeler:
    """Generate conservative target pseudo labels in normalized feature space."""

    def __init__(
        self,
        num_classes: int,
        distance_threshold: float = 0.05,
        minimum_per_class: int = 2,
        iterations: int = 10,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if not 0.0 < distance_threshold < 2.0:
            raise ValueError("distance_threshold must be in (0, 2)")
        self.num_classes = num_classes
        self.distance_threshold = distance_threshold
        self.minimum_per_class = minimum_per_class
        self.iterations = iterations

    def _source_centroids(self, features: Tensor, labels: Tensor) -> Tensor:
        vectors = F.normalize(features, dim=-1)
        centroids = []
        for class_index in range(self.num_classes):
            members = vectors[labels == class_index]
            if members.shape[0] == 0:
                raise ValueError(f"source batch has no samples for class {class_index}")
            centroids.append(F.normalize(members.mean(dim=0), dim=0))
        return torch.stack(centroids)

    @torch.no_grad()
    def __call__(
        self,
        source_features: Tensor,
        source_labels: Tensor,
        target_features: Tensor,
    ) -> PseudoLabelResult:
        if source_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError("features must be [N, C]")
        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError("source and target dimensions must match")

        target = F.normalize(target_features, dim=-1)
        centroids = self._source_centroids(source_features, source_labels)
        labels = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)

        for _ in range(self.iterations):
            similarity = target @ centroids.transpose(0, 1)
            next_labels = similarity.argmax(dim=1)
            updated = []
            for class_index in range(self.num_classes):
                members = target[next_labels == class_index]
                if members.shape[0] == 0:
                    updated.append(centroids[class_index])
                else:
                    updated.append(F.normalize(members.mean(dim=0), dim=0))
            new_centroids = torch.stack(updated)
            labels = next_labels
            if torch.allclose(new_centroids, centroids, atol=1e-6, rtol=0.0):
                centroids = new_centroids
                break
            centroids = new_centroids

        assigned_similarity = (target * centroids[labels]).sum(dim=1)
        cosine_distance = 1.0 - assigned_similarity
        accepted = cosine_distance < self.distance_threshold
        for class_index in range(self.num_classes):
            class_acceptance = accepted & (labels == class_index)
            if int(class_acceptance.sum()) < self.minimum_per_class:
                accepted[class_acceptance] = False

        output_labels = labels.clone()
        output_labels[~accepted] = -1
        return PseudoLabelResult(
            labels=output_labels,
            accepted=accepted,
            cosine_distance=cosine_distance,
            centroids=centroids,
        )

