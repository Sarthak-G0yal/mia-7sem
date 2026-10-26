"""Segmentation datasets in a simple folder layout, subject-level splits, paired augmentation, synthetic data.

Expected layout (written by scripts/download_data.py):

    data/seg/<dataset>/images/<subject>_<frame>.png      grayscale B-scan
    data/seg/<dataset>/masks/<subject>_<frame>.png       defect mask (0 / 255), reference annotator
    data/seg/<dataset>/masks_expert2/...                 optional second annotator (inter-observer reference)

The subject prefix (text before the first "_") keeps all frames of one subject in the same split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = (256, 384)  # (H, W); divisible by 16 for the U-Net, close to the 496x768 OCT aspect ratio


def subject_of(path: Path) -> str:
    return path.stem.split("_")[0]


def _read(path: Path, size: tuple[int, int], is_mask: bool) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    array = cv2.resize(array, (size[1], size[0]), interpolation=interpolation)
    return (array > 0.5).astype(np.float32) if is_mask else np.clip(array, 0.0, 1.0)


@dataclass
class SegData:
    """All images of one dataset held in memory as [N, 1, H, W] float tensors (datasets here are small)."""

    name: str
    images: torch.Tensor
    masks: torch.Tensor
    subjects: np.ndarray
    names: list[str]
    masks_expert2: torch.Tensor | None = None

    @classmethod
    def load(cls, name: str, root: Path, size: tuple[int, int] = IMAGE_SIZE) -> "SegData":
        folder = root / name
        paths = sorted((folder / "images").glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No images in {folder / 'images'}; run scripts/download_data.py first")
        images = np.stack([_read(p, size, False) for p in paths])
        masks = np.stack([_read(folder / "masks" / p.name, size, True) for p in paths])
        expert2 = None
        if all((folder / "masks_expert2" / p.name).exists() for p in paths):
            expert2 = torch.from_numpy(np.stack([_read(folder / "masks_expert2" / p.name, size, True)
                                                 for p in paths]))[:, None]
        return cls(name, torch.from_numpy(images)[:, None], torch.from_numpy(masks)[:, None],
                   np.array([subject_of(p) for p in paths]), [p.stem for p in paths], expert2)

    def split(self, fold: int = 0, folds: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
        """Subject-level k-fold split: chunk `fold` of the shuffled subjects is test, the next chunk is
        validation (early stopping / model selection), the rest is training. With 10 Duke subjects and
        5 folds that is 6 / 2 / 2 subjects, and every subject is tested exactly once across folds."""
        subjects = np.random.default_rng(seed).permutation(np.unique(self.subjects))
        chunks = np.array_split(subjects, folds)
        test, val = chunks[fold % folds], chunks[(fold + 1) % folds]
        train = np.setdiff1d(subjects, np.concatenate([test, val]))
        return {split: np.flatnonzero(np.isin(self.subjects, members))
                for split, members in (("train", train), ("val", val), ("test", test))}


class SegSubset(Dataset):
    def __init__(self, data: SegData, indices: np.ndarray):
        self.data, self.indices = data, np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        return self.data.images[index], self.data.masks[index]


def augment_pair(images: torch.Tensor, masks: torch.Tensor,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Same random flip / rotation / shift / scale for image and mask, then brightness/contrast on the image."""
    batch, device = images.shape[0], images.device

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.rand(batch, device=device, generator=generator) * (high - low) + low

    angle = uniform(-8.0, 8.0) * math.pi / 180.0
    scale = uniform(0.9, 1.1)
    flip = torch.where(torch.rand(batch, device=device, generator=generator) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, uniform(-0.05, 0.05)], dim=1),
                         torch.stack([sin * flip, cos, uniform(-0.05, 0.05)], dim=1)], dim=1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    contrast = uniform(0.85, 1.15).view(-1, 1, 1, 1)
    brightness = uniform(0.9, 1.1).view(-1, 1, 1, 1)
    return (((images - mean) * contrast + mean) * brightness).clamp_(0.0, 1.0), masks


def speckle_noise(images: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Multiplicative post-export speckle perturbation clip(x * (1 + eta), 0, 1), eta ~ N(0, sigma^2)."""
    if sigma <= 0:
        return images
    eta = torch.randn(images.shape, device=images.device, generator=generator) * sigma
    return (images * (1.0 + eta)).clamp(0.0, 1.0)


def boundary_band(masks: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Pixels within `radius` of a mask edge (morphological gradient), used for boundary supervision/analysis."""
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(masks, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-masks, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0, 1)


def make_synthetic(out_dir: Path, subjects: int = 12, frames: int = 10, size: tuple[int, int] = IMAGE_SIZE,
                   seed: int = 0) -> None:
    """OCT-like B-scans: curved hyper/hypo-reflective layers, log-compressed speckle, and dark fluid pockets
    (large and small) inside the retina as the defects to segment. Lets the pipeline run without downloads."""
    rng = np.random.default_rng(seed)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for subject in range(subjects):
        speckle_level = rng.uniform(0.6, 1.2)  # per-"scanner" speckle strength
        for frame in range(frames):
            top = height * rng.uniform(0.25, 0.35) + 12 * np.sin(2 * np.pi * xx / width + rng.uniform(0, 6.28))
            thickness = height * rng.uniform(0.3, 0.38)
            depth = (yy - top) / thickness  # 0 at inner retina, 1 at RPE
            tissue = np.zeros_like(yy)
            for position, amplitude, sharpness in ((0.0, 0.9, 0.03), (0.25, 0.45, 0.05), (0.45, 0.6, 0.04),
                                                   (0.7, 0.35, 0.05), (0.95, 1.0, 0.025)):
                tissue += amplitude * np.exp(-((depth - position) ** 2) / (2 * sharpness ** 2))
            tissue += 0.25 * ((depth > 0) & (depth < 1))
            tissue += 0.15 * np.exp(-np.clip(depth - 1, 0, None) * 4) * (depth > 1)
            mask = np.zeros_like(yy, dtype=bool)
            for _ in range(rng.integers(0, 4)):
                small = rng.random() < 0.4
                cx, cd = rng.uniform(0.15, 0.85) * width, rng.uniform(0.3, 0.7)
                rx = rng.uniform(4, 9) if small else rng.uniform(15, 45)
                ry = rng.uniform(3, 7) if small else rng.uniform(8, 22)
                cy = np.interp(cx, xx[0], top[0]) + cd * thickness
                pocket = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
                mask |= pocket
            tissue = np.where(mask, 0.04, tissue)
            intensity = np.clip(tissue, 0, None) * rng.exponential(1.0, size=yy.shape) ** speckle_level
            image = np.log1p(12 * intensity) / np.log1p(12 * 2.5)
            image = cv2.GaussianBlur(np.clip(image, 0, 1).astype(np.float32), (3, 3), 0.6)
            name = f"syn{subject:02d}_{frame:03d}.png"
            Image.fromarray((image * 255).astype(np.uint8)).save(out_dir / "images" / name)
            Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "masks" / name)
