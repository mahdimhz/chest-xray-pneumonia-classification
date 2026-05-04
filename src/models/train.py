from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src import config
from src.data.dataloaders import build_dataloaders
from src.models.architectures import build_model, configure_trainable_layers, count_trainable_parameters
from src.models.evaluate import evaluate_model, resolve_device, set_reproducible


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_name: str,
    model_name: str,
    stage: str,
    val_metrics: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "run_name": run_name,
            "model_name": model_name,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "class_names": config.CLASS_NAMES,
            "image_size": config.IMAGE_SIZE,
            "seed": config.SEED,
        },
        output_path,
    )


def metric_for_selection(metrics: dict[str, float | int]) -> float:
    roc_auc = metrics.get("roc_auc")
    if roc_auc == roc_auc:
        return float(roc_auc)
    return float(metrics["f1"])


def fit(args: argparse.Namespace) -> Path:
    set_reproducible(config.SEED)
    device = resolve_device(args.device)
    loaders = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args.model, pretrained=args.pretrained and args.checkpoint is None).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"Checkpoint model '{checkpoint['model_name']}' does not match requested model '{args.model}'."
            )
        model.load_state_dict(checkpoint["model_state_dict"])

    configure_trainable_layers(model, args.model, args.stage)
    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{args.model}_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_path = config.METRICS_DIR / f"{run_name}_training_metrics.csv"
    checkpoint_path = config.CHECKPOINT_DIR / f"{run_name}_best.pt"

    history = []
    best_score = -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate_model(
            model,
            loaders["val"],
            device,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        score = metric_for_selection(val_metrics)
        if score > best_score:
            best_score = score
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                run_name=run_name,
                model_name=args.model,
                stage=args.stage,
                val_metrics=val_metrics,
                output_path=checkpoint_path,
            )

        print(
            f"epoch={epoch:03d} stage={args.stage} train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training metrics: {metrics_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chest X-ray classifier.")
    parser.add_argument("--model", default="baseline_cnn", choices=["baseline_cnn", "resnet18"])
    parser.add_argument("--stage", default="full", choices=["full", "head", "finetune"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=config.BASELINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=config.BASELINE_LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    fit(parse_args())
