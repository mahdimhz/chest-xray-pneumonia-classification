from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.transforms import build_transforms


class ChestXrayDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = pd.read_csv(self.manifest_path)
        self.transform = transform

        required_columns = {"path", "label", "class_name", "split"}
        missing = required_columns.difference(self.records.columns)
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        image_path = config.PROJECT_ROOT / row["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


def seed_worker(worker_id: int) -> None:
    worker_seed = config.SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader(
    split: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    manifest_path = config.SPLIT_DIR / f"{split}.csv"
    dataset = ChestXrayDataset(manifest_path, transform=build_transforms(split))
    generator = torch.Generator().manual_seed(config.SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:
    return {
        split: build_dataloader(split, batch_size=batch_size, num_workers=num_workers)
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    loaders = build_dataloaders(num_workers=0)
    for split_name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(
            f"{split_name}: batch_images={tuple(images.shape)}, "
            f"batch_labels={tuple(labels.shape)}, labels={sorted(labels.unique().tolist())}"
        )
