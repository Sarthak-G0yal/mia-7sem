"""Training: Noise2Void pre-training of the denoiser, then segmentation + boundary + lambda-prior losses."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SegData, SegSubset, augment_pair, boundary_band
from .metrics import SegmentationMeter
from .models import METHODS, FeedbackSegmenter, count_parameters


@dataclass
class SegConfig:
    method: str
    dataset: str
    seed: int = 0
    fold: int = 0
    folds: int = 5
    split_seed: int = 0
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    denoiser_pretrain_steps: int = 300
    n2v_fraction: float = 0.02
    boundary_weight: float = 0.5
    smooth_prior: float = 0.05  # mild push to at least denoiser-level smoothing (lambda >= 1/3) against collapse
    edge_prior: float = 0.05  # discourage smoothing on annotated defect boundaries
    train_noise_max: float = 0.2  # extra speckle sigma ~ U(0, max) on half of the training batches' images
    patience: int = 15
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def noise2void_loss(denoiser: torch.nn.Module, images: torch.Tensor, fraction: float,
                    generator: torch.Generator) -> torch.Tensor:
    """Blind-spot loss: hide a random pixel subset behind a neighbouring value and predict the original."""
    blind = torch.rand(images.shape, device=images.device, generator=generator) < fraction
    dy, dx = random.choice([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)])
    corrupted = torch.where(blind, torch.roll(images, shifts=(dy, dx), dims=(2, 3)), images)
    return ((denoiser(corrupted) - images) ** 2)[blind].mean()


def soft_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    probability = logits.softmax(1)[:, 1:2]
    intersection = (probability * masks).sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (probability.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1)).mean()


def training_noise(images: torch.Tensor, max_sigma: float, generator: torch.Generator) -> torch.Tensor:
    """Per-image random speckle level (half of the images stay clean) so the controller sees varying noise."""
    batch = images.shape[0]
    sigma = torch.rand(batch, 1, 1, 1, device=images.device, generator=generator) * max_sigma
    sigma = torch.where(torch.rand(batch, 1, 1, 1, device=images.device, generator=generator) < 0.5, 0.0, sigma)
    eta = torch.randn(images.shape, device=images.device, generator=generator) * sigma
    return (images * (1.0 + eta)).clamp(0.0, 1.0)


@torch.no_grad()
def validation_dice(model: FeedbackSegmenter, subset: SegSubset, device: torch.device) -> float:
    model.eval()
    meter = SegmentationMeter()
    for images, masks in DataLoader(subset, batch_size=8):
        prediction = model(images.to(device))["logits"][-1].argmax(1).cpu().numpy()
        for p, m in zip(prediction, masks[:, 0].numpy()):
            meter.update(p, m)
    return meter.compute()["dice"]


def train_segmenter(config: SegConfig, data: SegData, out_dir: Path) -> FeedbackSegmenter:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = get_device()
    spec = METHODS[config.method]
    splits = data.split(config.fold, config.folds, config.split_seed)
    train_set, val_set = SegSubset(data, splits["train"]), SegSubset(data, splits["val"])
    model = FeedbackSegmenter(spec).to(device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(config.seed), num_workers=config.num_workers)
    positive = float(data.masks[splits["train"]].mean())
    class_weights = torch.tensor([1.0, min(20.0, 0.5 / max(positive, 1e-4))], device=device)
    print(f"[{config.method}/{config.dataset}/fold{config.fold}_seed{config.seed}] smoothing={spec.smoothing}, head={spec.head}, "
          f"T={spec.iterations}, params={count_parameters(model):,}, train={len(train_set)}, val={len(val_set)}, "
          f"test={len(splits['test'])}, defect pixels={100 * positive:.2f}%, device={device}")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    if model.denoiser is not None:
        optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=config.lr)
        model.train()
        step = 0
        while step < config.denoiser_pretrain_steps:
            for images, _ in loader:
                images = training_noise(images.to(device), config.train_noise_max, generator)
                optimizer.zero_grad(set_to_none=True)
                noise2void_loss(model.denoiser, images, config.n2v_fraction, generator).backward()
                optimizer.step()
                step += 1
                if step >= config.denoiser_pretrain_steps:
                    break

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, pct_start=0.1,
                                                    total_steps=config.epochs * len(loader))
    weights = torch.tensor([0.5 ** (spec.iterations - 1 - t) for t in range(spec.iterations)], device=device)
    weights = weights / weights.sum()
    history, best_dice, best_epoch = [], -1.0, 0
    for epoch in range(config.epochs):
        model.train()
        started, totals = time.time(), np.zeros(4)
        for images, masks in loader:
            images, masks = augment_pair(images.to(device), masks.to(device), generator)
            images = training_noise(images, config.train_noise_max, generator)
            band = boundary_band(masks)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            seg_loss = sum(w * (F.cross_entropy(logits, masks[:, 0].long(), weight=class_weights)
                                + soft_dice_loss(logits, masks))
                           for w, logits in zip(weights, outputs["logits"]))
            boundary_loss = sum(w * F.binary_cross_entropy_with_logits(
                b, band, pos_weight=torch.tensor(10.0, device=device)) for w, b in zip(weights, outputs["boundary_logits"]))
            loss = seg_loss + config.boundary_weight * boundary_loss
            prior = torch.zeros((), device=device)
            if spec.smoothing in ("open_loop", "feedback"):
                lam = torch.stack(outputs["lambda"])
                prior = (config.smooth_prior * F.relu(1 / 3 - lam).mean()
                         + config.edge_prior * (lam * band).sum() / band.sum().clamp_min(1))
                loss = loss + prior
            n2v = torch.zeros((), device=device)
            if model.denoiser is not None:
                n2v = noise2void_loss(model.denoiser, images, config.n2v_fraction, generator)
                loss = loss + n2v
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [seg_loss.item(), boundary_loss.item(), prior.item(), n2v.item()]
        totals /= len(loader)
        dice = validation_dice(model, val_set, device)
        improved = dice > best_dice
        if improved:
            best_dice, best_epoch = dice, epoch + 1
            torch.save(model.state_dict(), out_dir / "best.pt")
        history.append({"epoch": epoch + 1, "seg_loss": totals[0], "boundary_loss": totals[1],
                        "lambda_prior": totals[2], "n2v_loss": totals[3], "val_dice": dice,
                        "seconds": time.time() - started})
        print(f"  epoch {epoch + 1:3d} seg={totals[0]:.3f} bnd={totals[1]:.3f} n2v={totals[3]:.4f} "
              f"val_dice={dice:.2f} {history[-1]['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if epoch + 1 - best_epoch >= config.patience:
            print(f"  early stop; best epoch {best_epoch} (val dice {best_dice:.2f})")
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    return model
