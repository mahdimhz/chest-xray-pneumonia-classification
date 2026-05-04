from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from src import config


def _read_split(split: str) -> pd.DataFrame:
    path = config.SPLIT_DIR / f"{split}.csv"
    frame = pd.read_csv(path)
    frame["split"] = split
    return frame


def load_internal_splits() -> pd.DataFrame:
    return pd.concat([_read_split(split) for split in ("train", "val", "test")], ignore_index=True)


def path_overlap_report(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        left_paths = set(manifest.loc[manifest["split"] == left, "path"])
        right_paths = set(manifest.loc[manifest["split"] == right, "path"])
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "n_overlapping_paths": len(left_paths.intersection(right_paths)),
            }
        )
    return pd.DataFrame(rows)


def image_content_hash(image_path: Path, size: int = 64) -> str:
    with Image.open(image_path) as image:
        normalized = image.convert("L").resize((size, size))
        digest = hashlib.sha256(normalized.tobytes()).hexdigest()
    return digest


def content_overlap_report(manifest: pd.DataFrame) -> pd.DataFrame:
    hashed = manifest.copy()
    hashed["content_hash"] = [
        image_content_hash(config.PROJECT_ROOT / path) for path in hashed["path"]
    ]

    rows = []
    for content_hash, group in hashed.groupby("content_hash", observed=True):
        splits = sorted(group["split"].unique())
        if len(splits) > 1:
            rows.append(
                {
                    "content_hash": content_hash,
                    "splits": ",".join(splits),
                    "n_images": len(group),
                    "paths": "|".join(group["path"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def run_checks() -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_internal_splits()

    path_report = path_overlap_report(manifest)
    content_report = content_overlap_report(manifest)

    path_report.to_csv(config.TABLES_DIR / "split_path_overlap_report.csv", index=False)
    content_report.to_csv(config.TABLES_DIR / "split_content_overlap_report.csv", index=False)

    print("Path overlap report:")
    print(path_report.to_string(index=False))
    print("\nContent overlap report:")
    if content_report.empty:
        print("No duplicate resized grayscale content hashes across splits.")
    else:
        print(content_report[["splits", "n_images", "paths"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check internal splits for path and image-content leakage.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run_checks()
