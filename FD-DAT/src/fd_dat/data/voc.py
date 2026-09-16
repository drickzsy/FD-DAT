"""Minimal PASCAL VOC dataset used by the four transfer tasks in the paper."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class VOCDomainDataset(Dataset):
    """Read a VOC-style labeled source or unlabeled target domain."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str],
        *,
        unlabeled: bool = False,
    ) -> None:
        self.root = Path(root)
        self.unlabeled = unlabeled
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        split_path = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = [
            line.strip().split()[0]
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def _image_path(self, image_id: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = self.root / "JPEGImages" / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no image found for VOC id {image_id}")

    def _annotation(self, image_id: str) -> tuple[Tensor, Tensor]:
        root = ElementTree.parse(self.root / "Annotations" / f"{image_id}.xml").getroot()
        boxes: list[list[float]] = []
        labels: list[int] = []
        for instance in root.findall("object"):
            name = instance.findtext("name")
            if name not in self.class_to_index:
                continue
            box = instance.find("bndbox")
            if box is None:
                continue
            boxes.append(
                [
                    float(box.findtext("xmin", "0")),
                    float(box.findtext("ymin", "0")),
                    float(box.findtext("xmax", "0")),
                    float(box.findtext("ymax", "0")),
                ]
            )
            labels.append(self.class_to_index[name])
        return (
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = self.ids[index]
        with Image.open(self._image_path(image_id)) as pil_image:
            image = pil_to_tensor(pil_image.convert("RGB")).float().div_(255.0)
        if self.unlabeled:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes, labels = self._annotation(image_id)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.long),
        }
        return image, target
