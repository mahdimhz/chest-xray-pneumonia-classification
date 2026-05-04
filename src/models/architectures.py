from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def _build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNN()
    if model_name == "resnet18":
        return _build_resnet18(pretrained=pretrained)
    raise ValueError(f"Unknown model name: {model_name}")


def configure_trainable_layers(model: nn.Module, model_name: str, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if model_name == "baseline_cnn":
        return

    if model_name == "resnet18" and stage == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if model_name == "resnet18" and stage == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported stage '{stage}' for model '{model_name}'")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
