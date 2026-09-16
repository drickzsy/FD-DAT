"""Configuration check and core-module smoke test.

A complete training run additionally needs a Deformable DETR adapter satisfying
``fd_dat.engine.DetectorAdapter``. Keeping that boundary explicit prevents this
research scaffold from silently pretending to reproduce unpublished details.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from fd_dat.models import FDDATFeatureModule


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def smoke_test(config: dict) -> None:
    model_config = config["model"]
    inn = model_config["inn"]
    module = FDDATFeatureModule(
        channels=model_config["hidden_dim"],
        inn_hidden_channels=inn["subnet_hidden_dim"],
        inn_blocks=inn["blocks"],
        gamma=inn["gamma"],
        permutation_seed=inn["permutation_seed"],
        warmup_epochs=model_config["gradient_mask"]["warmup_epochs"],
        mmd_sigma_squared=config["loss"]["mmd_sigma_squared"],
    )
    channels = model_config["hidden_dim"]
    shapes = ((16, 16), (8, 8), (4, 4), (2, 2))
    source = [torch.randn(2, channels, h, w) for h, w in shapes]
    target = [torch.randn(2, channels, h, w) for h, w in shapes]
    output = module(source, target)
    round_trip = module.mapper.inverse(module.mapper(source[0]))
    error = (round_trip - source[0]).abs().max().item()
    print(f"feature levels: {len(output.source_latent)}")
    print(f"INN maximum round-trip error: {error:.3e}")
    print(f"CDD loss: {output.cdd_loss.item():.6f}")
    print(f"DI regularization loss: {output.regularization_loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/fd_dat_r101_voc.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"configuration OK: {config['experiment']['name']}")
    if args.smoke_test:
        smoke_test(config)
    else:
        print("Add a DetectorAdapter implementation, then connect it to FDDATTrainingCore.")


if __name__ == "__main__":
    main()

