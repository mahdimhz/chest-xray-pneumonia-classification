from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src import config
from src.data.dataloaders import build_dataloader
from src.models.architectures import build_model
from src.visualization.result_plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def set_reproducible(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str = config.DEVICE) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "sensitivity_pneumonia": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity_normal": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["average_precision"] = np.nan

    return metrics


def suspicious_metric_warnings(metrics: dict[str, float | int], split: str) -> list[str]:
    watched = ["accuracy", "f1", "macro_f1", "roc_auc", "sensitivity_pneumonia", "specificity_normal"]
    high_metrics = [
        metric_name
        for metric_name in watched
        if metric_name in metrics and metrics[metric_name] == metrics[metric_name] and metrics[metric_name] >= 0.99
    ]
    if not high_metrics:
        return []

    return [
        (
            f"Suspiciously high {split} metrics ({', '.join(high_metrics)} >= 0.99). "
            "Re-check split manifests, duplicate hashes, preprocessing consistency, and confusion matrix before trusting this run."
        )
    ]


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_prob = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="predict", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    y_true, y_prob = predict(model, loader, device, max_batches=max_batches)
    return compute_binary_metrics(y_true, y_prob)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def save_evaluation_outputs(
    metrics: dict[str, float | int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split: str,
    run_name: str,
) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob >= 0.5).astype(int)
    pd.DataFrame([{"split": split, **metrics}]).to_csv(
        config.METRICS_DIR / f"{run_name}_{split}_metrics.csv",
        index=False,
    )
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}).to_csv(
        config.METRICS_DIR / f"{run_name}_{split}_predictions.csv",
        index=False,
    )
    with (config.METRICS_DIR / f"{run_name}_{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=config.FIGURES_DIR / f"{run_name}_{split}_confusion_matrix.png",
    )
    if len(np.unique(y_true)) == 2:
        plot_roc_curve(
            y_true=y_true,
            y_prob=y_prob,
            output_path=config.FIGURES_DIR / f"{run_name}_{split}_roc_curve.png",
        )
        plot_precision_recall_curve(
            y_true=y_true,
            y_prob=y_prob,
            output_path=config.FIGURES_DIR / f"{run_name}_{split}_precision_recall_curve.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained chest X-ray classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible(config.SEED)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    loader = build_dataloader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    y_true, y_prob = predict(model, loader, device, max_batches=args.max_batches)
    metrics = compute_binary_metrics(y_true, y_prob)

    run_name = args.run_name or checkpoint.get("run_name", args.checkpoint.stem)
    save_evaluation_outputs(metrics, y_true, y_prob, split=args.split, run_name=run_name)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    for warning in suspicious_metric_warnings(metrics, args.split):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
