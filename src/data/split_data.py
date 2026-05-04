from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data.dataset_info import list_images, resolve_dataset_root, set_seed


SPLIT_FILENAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}

PERSON_PATTERN = re.compile(r"(person\d+)", flags=re.IGNORECASE)
NORMAL_PATTERN = re.compile(r"(IM-\d+)", flags=re.IGNORECASE)


def _to_relative_path(path: str) -> str:
    return str(Path(path).resolve().relative_to(config.PROJECT_ROOT))


def infer_split_group(file_name: str, class_name: str) -> str:
    person_match = PERSON_PATTERN.search(file_name)
    if person_match:
        return f"{class_name}:{person_match.group(1).lower()}"

    normal_match = NORMAL_PATTERN.search(file_name)
    if normal_match:
        return f"{class_name}:{normal_match.group(1).upper()}"

    return f"{class_name}:{Path(file_name).stem}"


def _prepare_manifest() -> pd.DataFrame:
    dataset_root = resolve_dataset_root(config.RAW_CHEST_XRAY_DIR)
    manifest = list_images(dataset_root.path).rename(columns={"split": "original_split"})
    manifest["path"] = manifest["path"].map(_to_relative_path)
    manifest["split_group"] = [
        infer_split_group(file_name=file_name, class_name=class_name)
        for file_name, class_name in zip(manifest["file_name"], manifest["class_name"])
    ]
    return manifest


def _split_train_val(manifest: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_val = manifest[manifest["original_split"].isin(["train", "val"])].copy()
    val_groups = []

    for class_name, class_frame in train_val.groupby("class_name", observed=True):
        groups = class_frame[["split_group", "class_name"]].drop_duplicates().reset_index(drop=True)
        _, class_val_groups = train_test_split(
            groups,
            test_size=val_fraction,
            random_state=config.SEED,
            shuffle=True,
        )
        val_groups.extend(class_val_groups["split_group"].tolist())

    val_group_set = set(val_groups)
    val_df = train_val[train_val["split_group"].isin(val_group_set)].copy()
    train_df = train_val[~train_val["split_group"].isin(val_group_set)].copy()

    train_df = train_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    return train_df, val_df


def _build_internal_splits(manifest: pd.DataFrame, val_fraction: float) -> dict[str, pd.DataFrame]:
    train_df, val_df = _split_train_val(manifest, val_fraction)
    test_df = (
        manifest[manifest["original_split"] == "test"]
        .copy()
        .sample(frac=1, random_state=config.SEED)
        .reset_index(drop=True)
    )

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for split_name, frame in split_frames.items():
        frame.insert(0, "split", split_name)
    return split_frames


def _check_split_leakage(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    split_names = list(split_frames)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            left_paths = set(split_frames[left_name]["path"])
            right_paths = set(split_frames[right_name]["path"])
            overlap = left_paths.intersection(right_paths)
            rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "n_overlapping_paths": len(overlap),
                }
            )
    return pd.DataFrame(rows)


def _summarize_splits(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = pd.concat(split_frames.values(), ignore_index=True)
    return (
        summary.groupby(["split", "class_name"], observed=True)
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "class_name"])
    )


def save_splits(split_frames: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["split", "original_split", "class_name", "label", "split_group", "path", "file_name"]
    for split_name, frame in split_frames.items():
        frame[columns].to_csv(output_dir / SPLIT_FILENAMES[split_name], index=False)

    summary = _summarize_splits(split_frames)
    leakage = _check_split_leakage(split_frames)

    summary.to_csv(output_dir / "split_summary.csv", index=False)
    leakage.to_csv(output_dir / "leakage_report.csv", index=False)
    return summary, leakage


def create_internal_splits(val_fraction: float) -> None:
    set_seed(config.SEED)
    manifest = _prepare_manifest()
    split_frames = _build_internal_splits(manifest, val_fraction=val_fraction)
    summary, leakage = save_splits(split_frames, config.SPLIT_DIR)

    print(f"Saved split manifests to: {config.SPLIT_DIR}")
    print("\nInternal split counts:")
    print(summary.to_string(index=False))
    print("\nPath overlap check:")
    print(leakage.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test CSV manifests.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=config.VAL_FRACTION,
        help="Validation fraction drawn from original train + val images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_internal_splits(val_fraction=args.val_fraction)
