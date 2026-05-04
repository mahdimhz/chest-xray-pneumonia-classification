from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from src import config
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetRoot:
    path: Path
    duplicate_candidates: tuple[Path, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_valid_dataset_root(path: Path) -> bool:
    return all(
        (path / split / class_name).is_dir()
        for split in EXPECTED_SPLITS
        for class_name in config.CLASS_NAMES
    )


def resolve_dataset_root(raw_chest_xray_dir: Path) -> DatasetRoot:
    candidates = [
        path
        for path in [raw_chest_xray_dir, *raw_chest_xray_dir.glob("**/chest_xray")]
        if is_valid_dataset_root(path)
    ]
    unique_candidates = tuple(dict.fromkeys(candidates))

    if not unique_candidates:
        raise FileNotFoundError(
            f"No valid chest_xray root found under {raw_chest_xray_dir}. "
            "Expected train/val/test folders with NORMAL and PNEUMONIA subfolders."
        )

    preferred = raw_chest_xray_dir if is_valid_dataset_root(raw_chest_xray_dir) else unique_candidates[0]
    duplicates = tuple(path for path in unique_candidates if path != preferred)
    return DatasetRoot(path=preferred, duplicate_candidates=duplicates)


def list_images(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for split in EXPECTED_SPLITS:
        for class_name in config.CLASS_NAMES:
            class_dir = dataset_root / split / class_name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "label": config.LABEL_MAP[class_name],
                            "path": str(image_path),
                            "file_name": image_path.name,
                        }
                    )
    return pd.DataFrame(rows)


def count_images(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        manifest.groupby(["split", "class_name"], observed=True)
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "class_name"])
    )
    return counts


def sample_manifest(manifest: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    sampled = []
    for _, group in manifest.groupby(["split", "class_name"], observed=True):
        n = min(len(group), max_per_group)
        sampled.append(group.sample(n=n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def inspect_image(path: Path) -> dict[str, float | int | str]:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        stat = ImageStat.Stat(grayscale)
        arr = np.asarray(grayscale)
        return {
            "width": int(grayscale.width),
            "height": int(grayscale.height),
            "aspect_ratio": float(grayscale.width / grayscale.height),
            "mean_intensity": float(stat.mean[0]),
            "std_intensity": float(stat.stddev[0]),
            "min_intensity": int(arr.min()),
            "max_intensity": int(arr.max()),
            "mode": image.mode,
        }


def build_image_stats(sampled_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in sampled_manifest.itertuples(index=False):
        image_path = Path(row.path)
        stats = inspect_image(image_path)
        rows.append({**row._asdict(), **stats})
    return pd.DataFrame(rows)


def save_tables(manifest: pd.DataFrame, counts: pd.DataFrame, image_stats: pd.DataFrame) -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    public_manifest = manifest.copy()
    public_image_stats = image_stats.copy()
    for frame in (public_manifest, public_image_stats):
        if "path" in frame.columns:
            frame["path"] = frame["path"].map(
                lambda path: str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
            )

    public_manifest.to_csv(config.TABLES_DIR / "image_manifest.csv", index=False)
    counts.to_csv(config.TABLES_DIR / "class_counts_by_split.csv", index=False)
    public_image_stats.to_csv(config.TABLES_DIR / "sampled_image_stats.csv", index=False)


def run_inspection(max_stats_per_group: int) -> None:
    from src.visualization.eda_plots import (
        plot_class_counts,
        plot_example_grid,
        plot_image_size_scatter,
        plot_intensity_summary,
    )

    set_seed(config.SEED)
    dataset_root = resolve_dataset_root(config.RAW_CHEST_XRAY_DIR)
    manifest = list_images(dataset_root.path)
    counts = count_images(manifest)
    sampled = sample_manifest(manifest, max_stats_per_group, config.SEED)
    image_stats = build_image_stats(sampled)

    save_tables(manifest, counts, image_stats)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_counts(counts, config.FIGURES_DIR / "class_counts_by_split.png")
    plot_example_grid(manifest, config.FIGURES_DIR / "example_images_grid.png", seed=config.SEED)
    plot_image_size_scatter(image_stats, config.FIGURES_DIR / "sampled_image_sizes.png")
    plot_intensity_summary(image_stats, config.FIGURES_DIR / "sampled_intensity_summary.png")

    print(f"Dataset root used: {dataset_root.path}")
    if dataset_root.duplicate_candidates:
        print("Duplicate valid dataset roots detected and ignored:")
        for duplicate in dataset_root.duplicate_candidates:
            print(f"  - {duplicate}")
    print("\nClass counts:")
    print(counts.to_string(index=False))
    print("\nSampled image size summary:")
    print(
        image_stats[["width", "height", "aspect_ratio", "mean_intensity", "std_intensity"]]
        .describe()
        .round(3)
        .to_string()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the raw chest X-ray dataset.")
    parser.add_argument(
        "--max-stats-per-group",
        type=int,
        default=150,
        help="Maximum images sampled per split/class group for image-size and intensity statistics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inspection(max_stats_per_group=args.max_stats_per_group)
