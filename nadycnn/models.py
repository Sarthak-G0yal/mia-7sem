"""Architectures compared in the paper (Sec. 2.3, Fig. 2).

Every model takes [0, 1] B-scans and applies per-image Z-normalization (Eq. 2) as its first step,
so the noise-aware branch can corrupt images in the post-export intensity domain before normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, resnet18

from . import CLASSES


class PerImageZNorm(nn.Module):
    """Eq. (2): Z(x) = (x - mu(x)) / (sigma(x) + eps), computed per image."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)


class DynamicConv2d(nn.Module):
    """Dynamic convolution (Chen et al., CVPR 2020) with K expert kernels, Eq. (3).

    The router is GAP -> 1x1 conv -> ReLU -> 1x1 conv -> softmax and yields per-image mixture weights
    alpha; the effective kernel W_eff = sum_k alpha_k W_k is applied with a grouped convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, num_experts: int = 4, reduction: int = 4):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride, self.padding = stride, padding
        hidden = max(in_channels // reduction, 4)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )
        self.weight = nn.Parameter(torch.empty(num_experts, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(num_experts, out_channels))
        for expert in self.weight:
            nn.init.kaiming_normal_(expert, mode="fan_out", nonlinearity="relu")
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        alpha = torch.softmax(self.router(x).flatten(1).float(), dim=1)
        self.last_alpha = alpha.detach()
        alpha = alpha.to(x.dtype)
        weight = torch.einsum("bk,koihw->boihw", alpha, self.weight.to(x.dtype))
        bias = (alpha @ self.bias.to(x.dtype)).reshape(-1)
        out = F.conv2d(x.reshape(1, batch * self.in_channels, height, width),
                       weight.reshape(batch * self.out_channels, self.in_channels, *weight.shape[-2:]),
                       bias, stride=self.stride, padding=self.padding, groups=batch)
        return out.reshape(batch, self.out_channels, *out.shape[-2:])


class OCTConvNet(nn.Module):
    """Basic CNN (Fig. 2a) or Dynamic CNN (Fig. 2b): four conv blocks (16-32-64-128), GAP, two-layer MLP.

    With dynamic=True every 3x3 convolution becomes a DynamicConv2d with K experts, keeping the
    macro-architecture identical (0.11 M vs 0.40 M parameters, Table 2).
    """

    def __init__(self, num_classes: int = len(CLASSES), widths: tuple[int, ...] = (16, 32, 64, 128),
                 dynamic: bool = False, num_experts: int = 4, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.znorm = PerImageZNorm()
        blocks, in_channels = [], 1
        for width in widths:
            conv = (DynamicConv2d(in_channels, width, num_experts=num_experts) if dynamic
                    else nn.Conv2d(in_channels, width, 3, padding=1))
            blocks.append(nn.Sequential(conv, nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
            in_channels = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.znorm(x))
        return self.classifier(F.adaptive_avg_pool2d(features, 1).flatten(1))

    def routing_weights(self) -> list[torch.Tensor]:
        """Mixture weights alpha of each dynamic layer from the most recent forward pass."""
        return [m.last_alpha for m in self.modules() if isinstance(m, DynamicConv2d)]


class TorchvisionClassifier(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.znorm = PerImageZNorm()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.znorm(x))


def build_mobilenet_v3_small(num_classes: int = len(CLASSES)) -> nn.Module:
    model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
    return TorchvisionClassifier(model)


def build_resnet18(num_classes: int = len(CLASSES)) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    return TorchvisionClassifier(model)


ARCHITECTURES = {
    "basic": lambda: OCTConvNet(dynamic=False),
    "dynamic": lambda: OCTConvNet(dynamic=True),
    "mobilenetv3": build_mobilenet_v3_small,
    "resnet18": build_resnet18,
}


def build_model(architecture: str) -> nn.Module:
    return ARCHITECTURES[architecture]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
