from __future__ import annotations

from torchvision import transforms

from src import config


def build_transforms(split: str, image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ]
        )

    if split in {"val", "test"}:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unknown split: {split}")
