from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.data.dataloaders import build_dataloader
from src.models.evaluate import (
    compute_binary_metrics,
    load_checkpoint,
    predict,
    resolve_device,
    save_evaluation_outputs,
    set_reproducible,
    suspicious_metric_warnings,
)


def evaluate_checkpoint(
    checkpoint_path: Path,
    splits: list[str],
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    device_name: str,
) -> list[dict]:
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    run_name = checkpoint.get("run_name", checkpoint_path.stem)
    rows = []

    for split in splits:
        loader = build_dataloader(split, batch_size=batch_size, num_workers=num_workers)
        y_true, y_prob = predict(model, loader, device, max_batches=max_batches)
        metrics = compute_binary_metrics(y_true, y_prob)
        save_evaluation_outputs(metrics, y_true, y_prob, split=split, run_name=run_name)

        row = {
            "checkpoint": str(checkpoint_path),
            "run_name": run_name,
            "model_name": checkpoint["model_name"],
            "checkpoint_epoch": checkpoint["epoch"],
            "split": split,
            **metrics,
        }
        rows.append(row)

        for warning in suspicious_metric_warnings(metrics, split):
            print(f"WARNING [{run_name} / {split}]: {warning}")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one or more trained checkpoints.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-name", default="checkpoint_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible(config.SEED)
    rows = []

    for checkpoint_path in args.checkpoints:
        rows.extend(
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                splits=args.splits,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
                device_name=args.device,
            )
        )

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison_path = config.METRICS_DIR / args.output_name
    comparison.to_csv(comparison_path, index=False)

    display_columns = [
        "run_name",
        "model_name",
        "split",
        "accuracy",
        "f1",
        "macro_f1",
        "roc_auc",
        "sensitivity_pneumonia",
        "specificity_normal",
    ]
    print(f"Saved comparison: {comparison_path}")
    print(comparison[display_columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
