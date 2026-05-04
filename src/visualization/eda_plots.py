from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from src import config


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_class_counts(counts: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="split", y="n_images", hue="class_name", ax=ax)
    ax.set_title("Chest X-ray Counts by Split and Class")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of images")
    ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_example_grid(manifest: pd.DataFrame, output_path: Path, seed: int) -> None:
    _set_style()
    examples = (
        manifest.groupby(["split", "class_name"], observed=True)
        .sample(n=1, random_state=seed)
        .sort_values(["split", "class_name"])
    )

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    for ax, row in zip(axes.ravel(), examples.itertuples(index=False)):
        with Image.open(row.path) as image:
            ax.imshow(image.convert("L"), cmap="gray")
        ax.set_title(f"{row.split} / {row.class_name}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_size_scatter(image_stats: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=image_stats,
        x="width",
        y="height",
        hue="class_name",
        style="split",
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(config.IMAGE_SIZE, color="black", linewidth=1, linestyle="--")
    ax.axhline(config.IMAGE_SIZE, color="black", linewidth=1, linestyle="--")
    ax.set_title("Sampled Original Image Dimensions")
    ax.set_xlabel("Width, pixels")
    ax.set_ylabel("Height, pixels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_intensity_summary(image_stats: pd.DataFrame, output_path: Path) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.boxplot(data=image_stats, x="split", y="mean_intensity", hue="class_name", ax=axes[0])
    axes[0].set_title("Mean Grayscale Intensity")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Mean intensity")
    sns.boxplot(data=image_stats, x="split", y="std_intensity", hue="class_name", ax=axes[1])
    axes[1].set_title("Grayscale Contrast")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Intensity std.")

    for ax in axes:
        ax.legend(title="Class")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
