"""Smoother (learned denoiser), U-Net segmenter with hyperspherical head, and the lambda feedback controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    """DnCNN-style residual denoiser D(x). Trained self-supervised (Noise2Void blind-spot loss), so it needs
    no clean targets; the task loss does not change it, which keeps "smoothing" meaning denoising."""

    def __init__(self, channels: int = 32, depth: int = 6):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.net(x)).clamp(0.0, 1.0)


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small U-Net returning full-resolution features (per-image Z-normalized input)."""

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.down = nn.ModuleList()
        in_channels = 1
        for width in widths:
            self.down.append(_block(in_channels, width))
            in_channels = width
        self.bottleneck = _block(widths[-1], widths[-1] * 2)
        self.up, self.fuse = nn.ModuleList(), nn.ModuleList()
        in_channels = widths[-1] * 2
        for width in reversed(widths):
            self.up.append(nn.ConvTranspose2d(in_channels, width, 2, stride=2))
            self.fuse.append(_block(width * 2, width))
            in_channels = width
        self.out_channels = widths[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, fuse, skip in zip(self.up, self.fuse, reversed(skips)):
            x = fuse(torch.cat([up(x), skip], dim=1))
        return x


class HypersphericalHead(nn.Module):
    """Per-pixel embeddings on the unit sphere scored against learnable unit prototypes.

    Class logits are s * cos(e, p_c); a separate boundary prototype gives the boundary score. Confidence is
    the cosine to the nearest class prototype mapped to [0, 1]: low when a pixel sits between classes.
    """

    def __init__(self, in_channels: int, num_classes: int = 2, dim: int = 16, scale: float = 10.0):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, 1)
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_classes + 1, dim), dim=1))
        self.scale = scale
        self.boundary_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = F.normalize(self.embed(features), dim=1)
        cosine = torch.einsum("bdhw,kd->bkhw", embedding, F.normalize(self.prototypes, dim=1))
        class_cosine = cosine[:, :-1]
        return {"logits": self.scale * class_cosine,
                "boundary_logits": self.scale * cosine[:, -1:] + self.boundary_bias,
                "confidence": (class_cosine.max(dim=1, keepdim=True).values + 1.0) / 2.0,
                "embedding": embedding}


class LinearHead(nn.Module):
    """Standard 1x1-conv softmax head (ablation of the hyperspherical head); confidence = max softmax."""

    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.classify = nn.Conv2d(in_channels, num_classes, 1)
        self.boundary = nn.Conv2d(in_channels, 1, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classify(features)
        return {"logits": logits, "boundary_logits": self.boundary(features),
                "confidence": logits.softmax(1).max(dim=1, keepdim=True).values}


LAMBDA_MIN, LAMBDA_MAX = 0.02, 0.98


def bounded_lambda(z: torch.Tensor) -> torch.Tensor:
    """lambda = sigmoid(z) squeezed into [LAMBDA_MIN, LAMBDA_MAX] so its gradient never vanishes."""
    return LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * torch.sigmoid(z)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(3 * sigma)
    taps = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-taps ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))


class ScaleSpaceSmoother(nn.Module):
    """S(x, lambda): per-pixel interpolation along [x, D(x), G_1(D(x)), G_2(D(x))].

    lambda = 0 keeps the raw scan, 1/3 is the learned denoiser, 1 is the denoised scan blurred with sigma = 2 px,
    so lambda reads directly as "how much smoothing" and stays differentiable."""

    def __init__(self, blur_sigmas: tuple[float, ...] = (1.0, 2.0)):
        super().__init__()
        self.denoiser = Denoiser()
        self.blur_sigmas = blur_sigmas

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        denoised = self.denoiser(x).detach()
        return torch.stack([x, denoised] + [_gaussian_blur(denoised, s) for s in self.blur_sigmas], dim=0)

    @staticmethod
    def apply(levels: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        position = lam * (levels.shape[0] - 1)
        lower = position.floor().clamp(max=levels.shape[0] - 2)
        weight = position - lower
        index = lower.long()[None]
        low = torch.gather(levels, 0, index)[0]
        high = torch.gather(levels, 0, index + 1)[0]
        return low + weight * (high - low)


class LambdaController(nn.Module):
    """Feedback update: z_{t+1} = z_t + g(confidence, boundary, |grad x_s|, lambda_t), lambda = bounded(z).

    The last layer starts at zero, so the loop begins as "keep the current smoothing" and learns from the
    segmentation loss when to smooth more (unsure + no boundary) or less (boundary)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, confidence: torch.Tensor, boundary: torch.Tensor,
                smoothed: torch.Tensor) -> torch.Tensor:
        features = torch.cat([confidence, boundary, image_gradient(smoothed), bounded_lambda(z)], dim=1)
        return z + self.net(features)


class InitialLambda(nn.Module):
    """Open-loop z_0 (lambda_0 = bounded(z_0)) predicted from the raw image alone."""

    def __init__(self, hidden: int = 16, start: float = 0.33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        unit = (start - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
        nn.init.constant_(self.net[-1].bias, float(torch.logit(torch.tensor(unit))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    kernels = torch.stack([sobel, sobel.t()])[:, None]
    grad = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernels)
    return grad.pow(2).sum(1, keepdim=True).add(1e-8).sqrt()


@dataclass(frozen=True)
class MethodSpec:
    smoothing: str  # "none" | "fixed" | "open_loop" | "feedback"
    head: str = "hyperspherical"
    fixed_lambda: float = 0.0
    iterations: int = 1


METHODS = {
    "unet_linear": MethodSpec("none", head="linear"),             # plain U-Net baseline
    "unet": MethodSpec("none"),                                     # U-Net + hyperspherical head, no smoothing
    "fixed_0.33": MethodSpec("fixed", fixed_lambda=1 / 3),         # constant: learned denoiser only
    "fixed_0.67": MethodSpec("fixed", fixed_lambda=2 / 3),         # constant: denoiser + Gaussian sigma 1
    "fixed_1.0": MethodSpec("fixed", fixed_lambda=1.0),            # constant: denoiser + Gaussian sigma 2
    "open_loop": MethodSpec("open_loop"),                           # learned lambda map from the image only
    "feedback": MethodSpec("feedback", iterations=3),               # proposed: closed-loop, hyperspherical
    "feedback_linear": MethodSpec("feedback", head="linear", iterations=3),  # ablation: softmax confidence
}


class FeedbackSegmenter(nn.Module):
    """x_s^t = S(x, lambda_t);  out_t = Head(UNet(x_s^t));  lambda_{t+1} = Controller(out_t, x_s^t, lambda_t)."""

    def __init__(self, spec: MethodSpec, num_classes: int = 2):
        super().__init__()
        self.spec = spec
        self.smoother = ScaleSpaceSmoother() if spec.smoothing != "none" else None
        self.unet = UNet()
        self.head = (HypersphericalHead(self.unet.out_channels, num_classes) if spec.head == "hyperspherical"
                     else LinearHead(self.unet.out_channels, num_classes))
        self.initial = InitialLambda() if spec.smoothing in ("open_loop", "feedback") else None
        self.controller = LambdaController() if spec.smoothing == "feedback" else None

    @property
    def denoiser(self) -> Denoiser | None:
        return self.smoother.denoiser if self.smoother is not None else None

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        outputs = {"logits": [], "boundary_logits": [], "confidence": [], "lambda": [], "smoothed": []}
        levels = self.smoother.stack(x) if self.smoother is not None else None
        z = self.initial(x) if self.initial is not None else None
        for step in range(self.spec.iterations):
            if self.spec.smoothing == "none":
                lam, smoothed = torch.zeros_like(x), x
            else:
                lam = bounded_lambda(z) if z is not None else torch.full_like(x, self.spec.fixed_lambda)
                smoothed = self.smoother.apply(levels, lam)
            out = self.head(self.unet(smoothed))
            for key in ("logits", "boundary_logits", "confidence"):
                outputs[key].append(out[key])
            outputs["lambda"].append(lam)
            outputs["smoothed"].append(smoothed)
            if self.controller is not None and step + 1 < self.spec.iterations:
                z = self.controller(z, out["confidence"], torch.sigmoid(out["boundary_logits"]), smoothed)
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
