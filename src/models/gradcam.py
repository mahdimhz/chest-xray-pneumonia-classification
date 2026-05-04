from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src import config
from src.data.transforms import build_transforms
from src.models.evaluate import load_checkpoint, resolve_device, set_reproducible


@dataclass(frozen=True)
class PredictionRecord:
    index: int
    path: Path
    true_label: int
    predicted_label: int
    pneumonia_probability: float


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor).view(-1)
        score = logits[0] if target_class == 1 else -logits[0]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return np.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Grad-CAM is implemented for the transfer model only, not '{model_name}'.")


def load_manifest(split: str) -> pd.DataFrame:
    manifest_path = config.SPLIT_DIR / f"{split}.csv"
    return pd.read_csv(manifest_path)


def load_image_tensor(path: Path, split: str, device: torch.device) -> torch.Tensor:
    transform = build_transforms("val" if split != "train" else "train")
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


def load_display_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    return np.asarray(image, dtype=float) / 255.0


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    split: str,
    device: torch.device,
    max_images: int,
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    correct = []
    misclassified = []

    for row in tqdm(manifest.head(max_images).itertuples(index=True), total=min(len(manifest), max_images), desc="scan"):
        image_path = config.PROJECT_ROOT / row.path
        image_tensor = load_image_tensor(image_path, split=split, device=device)
        logit = model(image_tensor).view(-1)[0]
        probability = float(torch.sigmoid(logit).cpu())
        predicted_label = int(probability >= 0.5)
        record = PredictionRecord(
            index=row.Index,
            path=image_path,
            true_label=int(row.label),
            predicted_label=predicted_label,
            pneumonia_probability=probability,
        )

        if predicted_label == int(row.label):
            correct.append(record)
        else:
            misclassified.append(record)

    return correct, misclassified


def select_records(
    correct: list[PredictionRecord],
    misclassified: list[PredictionRecord],
    num_correct: int,
    num_misclassified: int,
) -> list[tuple[str, PredictionRecord]]:
    selected = [("correct", record) for record in correct[:num_correct]]
    selected.extend(("misclassified", record) for record in misclassified[:num_misclassified])
    return selected


def save_overlay(
    base_image: np.ndarray,
    cam: np.ndarray,
    record: PredictionRecord,
    group: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base_image, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.38)
    true_name = config.CLASS_NAMES[record.true_label]
    pred_name = config.CLASS_NAMES[record.predicted_label]
    ax.set_title(f"{group}: true={true_name}, pred={pred_name}, p={record.pneumonia_probability:.3f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_gradcam_outputs(args: argparse.Namespace) -> None:
    set_reproducible(config.SEED)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()

    model_name = checkpoint["model_name"]
    run_name = args.run_name or checkpoint.get("run_name", args.checkpoint.stem)
    target_layer = get_target_layer(model, model_name)

    manifest = load_manifest(args.split)
    correct, misclassified = collect_predictions(
        model=model,
        manifest=manifest,
        split=args.split,
        device=device,
        max_images=args.max_images,
    )

    output_dir = config.FIGURES_DIR / "gradcam" / f"{run_name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_records(correct, misclassified, args.num_correct, args.num_misclassified)
    gradcam = GradCAM(model, target_layer)

    rows = []
    try:
        for group, record in selected:
            image_tensor = load_image_tensor(record.path, split=args.split, device=device)
            target_class = record.predicted_label if args.target == "predicted" else record.true_label
            cam = gradcam(image_tensor, target_class=target_class)
            base_image = load_display_image(record.path)
            output_path = output_dir / f"{group}_{record.index:05d}_true{record.true_label}_pred{record.predicted_label}.png"
            save_overlay(base_image, cam, record, group, output_path)
            rows.append(
                {
                    "group": group,
                    "path": str(record.path.relative_to(config.PROJECT_ROOT)),
                    "true_label": record.true_label,
                    "predicted_label": record.predicted_label,
                    "pneumonia_probability": record.pneumonia_probability,
                    "target": args.target,
                    "output_path": str(output_path.relative_to(config.PROJECT_ROOT)),
                }
            )
    finally:
        gradcam.close()

    pd.DataFrame(rows).to_csv(output_dir / "gradcam_index.csv", index=False)

    print(f"Scanned images: {min(len(manifest), args.max_images)}")
    print(f"Correct found: {len(correct)}")
    print(f"Misclassified found: {len(misclassified)}")
    print(f"Saved Grad-CAM outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a ResNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--num-correct", type=int, default=4)
    parser.add_argument("--num-misclassified", type=int, default=4)
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    generate_gradcam_outputs(parse_args())
