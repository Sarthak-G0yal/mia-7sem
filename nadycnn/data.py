"""Dataset indexing, quality control, preprocessing cache, and GPU augmentation (paper Sec. 2.1-2.2).

Every B-scan is square-padded to max(W, H), Lanczos-resized to 256x256 and stored once as uint8 in
a memory-mapped cache, so training reads [0, 1] intensities without re-decoding JPEG/TIFF files.
Per-image Z-normalization happens inside the model so that speckle noise can be injected before it.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from . import CLASSES

IMAGE_SIZE = 256
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}
# Resolutions retained after QC (Table 1); rare or highly anisotropic sizes such as 1536x496 are dropped.
ALLOWED_RESOLUTIONS = {(512, 496), (768, 496), (512, 512)}
DATA_ROOTS = {
    "d1": Path("data/ucsd/OCT2017 "),
    "d2": Path("data/oct_c8/RetinalOCT_Dataset/RetinalOCT_Dataset"),
    "d3": Path("data/neh/NEH_UT_2021RetinalOCTDataset/NEH_UT_2021RetinalOCTDataset"),
}
DATASET_NAMES = {"d1": "UCSD", "d2": "OCT-C8", "d3": "NEH"}
# Marker sub-folders that identify each dataset root, so other unzip layouts (Kaggle, Mendeley) also work.
ROOT_MARKERS = {"d1": ("train/CNV", "train/DME"), "d2": ("train/DME", "train/DRUSEN"), "d3": ("CNV", "DRUSEN", "NORMAL")}


def resolve_root(dataset: str) -> Path:
    """The configured path if it exists, else the shallowest folder under data/<dataset dir>/ holding the
    dataset's marker sub-folders (e.g. data/ucsd/OCT2017/train/CNV)."""
    configured = DATA_ROOTS[dataset]
    if all((configured / marker).is_dir() for marker in ROOT_MARKERS[dataset]):
        return configured
    base = Path("data") / configured.parts[1]
    candidates = sorted((p for p in [base, *base.rglob("*")] if p.is_dir() and "__MACOSX" not in p.parts
                         and all((p / marker).is_dir() for marker in ROOT_MARKERS[dataset])),
                        key=lambda p: len(p.parts))
    if not candidates:
        raise FileNotFoundError(f"{DATASET_NAMES[dataset]} not found under {base}/ "
                                f"(need sub-folders {ROOT_MARKERS[dataset]}); see scripts/download_data.py")
    return candidates[0]
UCSD_SPLIT_SEED = 2026


@dataclass
class Record:
    path: str
    label: int
    patient: str
    split: str


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _passes_qc(paths: list[Path], workers: int) -> list[bool]:
    with Pool(workers) as pool:
        sizes = pool.map(_image_size, paths, chunksize=256)
    return [size in ALLOWED_RESOLUTIONS for size in sizes]


def index_ucsd(root: Path, workers: int) -> list[Record]:
    """Pool all UCSD folders, de-duplicate by filename, and split 70/15/15 at the patient level."""
    files: dict[str, tuple[Path, int]] = {}
    for split_dir in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASSES):
            for path in sorted((root / split_dir / class_name).glob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    files.setdefault(path.name, (path, class_index))
    entries = sorted(files.values())
    keep = _passes_qc([path for path, _ in entries], workers)
    entries = [entry for entry, ok in zip(entries, keep) if ok]
    patients = sorted({path.name.split("-")[1] for path, _ in entries})
    order = np.random.default_rng(UCSD_SPLIT_SEED).permutation(len(patients))
    n_train, n_val = round(0.70 * len(patients)), round(0.15 * len(patients))
    split_of = {}
    for rank, index in enumerate(order):
        split_of[patients[index]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return [Record(str(path), label, path.name.split("-")[1], split_of[path.name.split("-")[1]])
            for path, label in entries]


def index_oct_c8(root: Path, workers: int) -> list[Record]:
    """Keep the four overlapping classes and the original splits; images carry no patient IDs."""
    entries = []
    for split_dir in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASSES):
            for path in sorted((root / split_dir / class_name).glob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    entries.append((path, class_index, split_dir))
    keep = _passes_qc([path for path, _, _ in entries], workers)
    return [Record(str(path), label, f"{split}/{path.stem}", split)
            for (path, label, split), ok in zip(entries, keep) if ok]


def index_neh(root: Path, workers: int) -> list[Record]:
    """Keep only frames whose frame-level label (filename suffix) matches the volume label (folder)."""
    records = []
    for class_index, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem.split("_")[-1].upper() != class_name:
                continue
            patient = f"{class_name}/{path.relative_to(class_dir).parts[0]}"
            records.append(Record(str(path), class_index, patient, "test"))
    return records


INDEXERS = {"d1": index_ucsd, "d2": index_oct_c8, "d3": index_neh}


def load_scan(path: str) -> np.ndarray:
    """Read a B-scan as 8-bit-range grayscale in [0, 1]; 16-bit inputs are scaled linearly."""
    with Image.open(path) as image:
        if image.mode.startswith("I"):
            array = np.asarray(image, dtype=np.float32)
            return np.clip(array / 65535.0, 0.0, 1.0)
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def square_pad_resize(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape
    side = max(height, width)
    padded = np.zeros((side, side), dtype=np.float32)
    top, left = (side - height) // 2, (side - width) // 2
    padded[top:top + height, left:left + width] = image
    return np.clip(cv2.resize(padded, (size, size), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)


def _preprocess(path: str) -> np.ndarray:
    return np.round(square_pad_resize(load_scan(path)) * 255.0).astype(np.uint8)


def build_cache(dataset: str, cache_root: Path, workers: int = 12) -> None:
    out_dir = cache_root / dataset
    if (out_dir / "images.npy").exists() and (out_dir / "manifest.csv").exists():
        print(f"[{dataset}] cache exists at {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    records = INDEXERS[dataset](resolve_root(dataset), workers)
    print(f"[{dataset}] {len(records)} B-scans after QC; preprocessing to {IMAGE_SIZE}x{IMAGE_SIZE}")
    images = np.lib.format.open_memmap(out_dir / "images.tmp.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(records), IMAGE_SIZE, IMAGE_SIZE))
    with Pool(workers) as pool:
        for index, array in enumerate(pool.imap(_preprocess, [r.path for r in records], chunksize=64)):
            images[index] = array
    images.flush()
    del images
    (out_dir / "images.tmp.npy").rename(out_dir / "images.npy")
    with open(out_dir / "manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "patient", "split"])
        writer.writerows([r.path, r.label, r.patient, r.split] for r in records)


@dataclass
class CachedDataset:
    """Memory-mapped uint8 images plus per-image label, patient ID and split."""

    name: str
    image_path: Path
    labels: np.ndarray
    patients: np.ndarray
    splits: np.ndarray

    @classmethod
    def load(cls, dataset: str, cache_root: Path) -> "CachedDataset":
        out_dir = cache_root / dataset
        with open(out_dir / "manifest.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return cls(dataset, out_dir / "images.npy",
                   np.array([int(r["label"]) for r in rows], dtype=np.int64),
                   np.array([r["patient"] for r in rows]),
                   np.array([r["split"] for r in rows]))

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    def subset(self, split: str) -> "ScanDataset":
        return ScanDataset(self.image_path, self.indices(split), self.labels)


class ScanDataset(Dataset):
    """Returns (uint8 [1, H, W] tensor, label); the memmap is opened lazily in each worker."""

    def __init__(self, image_path: Path, indices: np.ndarray, labels: np.ndarray):
        self.image_path = image_path
        self.indices = np.asarray(indices)
        self.labels = labels[self.indices]
        self._images = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        if self._images is None:
            self._images = np.load(self.image_path, mmap_mode="r")
        image = torch.from_numpy(np.array(self._images[self.indices[item]]))[None]
        return image, int(self.labels[item])


def to_unit_range(images: torch.Tensor) -> torch.Tensor:
    return images.float().div_(255.0)


def augment(images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Mild spatial (flip, rotation, translation, scaling) and photometric (brightness, contrast) jitter.

    Operates per sample on a GPU batch of [0, 1] images and returns [0, 1] images.
    """
    batch, device = images.shape[0], images.device

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.rand(batch, device=device, generator=generator) * (high - low) + low

    angle = uniform(-10.0, 10.0) * math.pi / 180.0
    scale = uniform(0.9, 1.1)
    flip = torch.where(torch.rand(batch, device=device, generator=generator) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, uniform(-0.05, 0.05)], dim=1),
                         torch.stack([sin * flip, cos, uniform(-0.05, 0.05)], dim=1)], dim=1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    contrast = uniform(0.9, 1.1).view(-1, 1, 1, 1)
    brightness = uniform(0.9, 1.1).view(-1, 1, 1, 1)
    return (((images - mean) * contrast + mean) * brightness).clamp_(0.0, 1.0)


def speckle_noise(images: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Eq. (1): x^(n) = clip(x * (1 + eta), 0, 1) with pixel-wise eta ~ N(0, sigma^2), in the [0, 1] domain."""
    eta = torch.randn(images.shape, device=images.device, generator=generator) * sigma
    return (images * (1.0 + eta)).clamp_(0.0, 1.0)
