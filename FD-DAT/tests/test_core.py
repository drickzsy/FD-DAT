import torch

from fd_dat.losses.consistency import decoder_consistency_loss
from fd_dat.losses.mmd import MultiKernelMMD
from fd_dat.models.gradient_mask import GradientMaskBank
from fd_dat.models.invertible import InvertibleFeatureMapper
from fd_dat.models.pseudo_labels import SphericalKMeansPseudoLabeler


def test_invertible_mapper_round_trip() -> None:
    mapper = InvertibleFeatureMapper(
        channels=8,
        hidden_channels=12,
        num_blocks=3,
        permutation_seed=7,
    )
    inputs = torch.randn(2, 8, 5, 7)
    recovered = mapper.inverse(mapper(inputs))
    torch.testing.assert_close(recovered, inputs, atol=1e-6, rtol=1e-6)


def test_gradient_mask_uses_epoch_median() -> None:
    bank = GradientMaskBank(channels=6, warmup_epochs=1)
    bank.gradient_sum.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    bank.observation_count.fill_(1)
    mask = bank.finish_epoch(epoch_index=1)
    torch.testing.assert_close(mask, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_mkmmd_is_symmetric() -> None:
    first = torch.randn(5, 9)
    second = torch.randn(4, 9)
    loss = MultiKernelMMD()
    torch.testing.assert_close(loss(first, second), loss(second, first))


def test_decoder_consistency_is_zero_for_identical_layers() -> None:
    logits = torch.randn(1, 2, 10, 3).repeat(4, 1, 1, 1)
    boxes = torch.rand(1, 2, 10, 4).repeat(4, 1, 1, 1)
    loss = decoder_consistency_loss(logits, boxes)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_spherical_kmeans_rejects_ambiguous_targets() -> None:
    source = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.7, 0.7]])
    labeler = SphericalKMeansPseudoLabeler(
        num_classes=2,
        distance_threshold=0.05,
        minimum_per_class=2,
        iterations=1,
    )
    result = labeler(source, labels, target)
    assert result.accepted.tolist() == [True, True, False]
    assert result.labels.tolist() == [0, 0, -1]
