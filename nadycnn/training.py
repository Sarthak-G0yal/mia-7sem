"""Training protocol (Sec. 2.4-2.5) for the baselines, the ablation variant, and NA-DyCNN."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import CLASSES
from .data import CachedDataset, ScanDataset, augment, speckle_noise, to_unit_range
from .models import OCTConvNet, build_model, count_parameters

# method -> (architecture, objective). "noise_aug" is ablation variant (3) of Table 8.
METHODS = {
    "basic": ("basic", "standard"),
    "dynamic": ("dynamic", "standard"),
    "dynamic_aug": ("dynamic", "noise_aug"),
    "nadycnn": ("dynamic", "noise_aware"),
    "mobilenetv3": ("mobilenetv3", "standard"),
    "resnet18": ("resnet18", "standard"),
}
# Selected noise-aware hyperparameters (Sec. 2.5.1): (sigma, beta, N) per source dataset.
NOISE_DEFAULTS = {"d1": (0.06, 0.12, 4), "d2": (0.10, 0.10, 4)}


@dataclass
class TrainConfig:
    method: str
    source: str
    seed: int = 0
    epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    lr_start: float = 1e-6
    lr_end: float = 1e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 10
    sigma: float | None = None
    beta: float | None = None
    realizations: int | None = None
    noise_aug_prob: float = 0.5
    max_train_samples: int | None = None
    num_workers: int = 6
    amp: bool = True
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        sigma, beta, realizations = NOISE_DEFAULTS.get(self.source, NOISE_DEFAULTS["d2"])
        self.sigma = sigma if self.sigma is None else self.sigma
        self.beta = beta if self.beta is None else self.beta
        self.realizations = realizations if self.realizations is None else self.realizations


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, lr: float, lr_start: float, lr_end: float) -> float:
    """Linear warm-up lr_start -> lr, then cosine annealing lr -> lr_end."""
    if step < warmup_steps:
        return lr_start + (lr - lr_start) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, dataset: ScanDataset, device: torch.device, batch_size: int = 128,
            num_workers: int = 4, amp: bool = True) -> dict[str, np.ndarray]:
    """Clean single-pass inference (noise branch disabled, Sec. 2.4.6). Returns logits and final-block alpha."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    logits, alphas = [], []
    for images, _ in loader:
        images = to_unit_range(images.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(images)
        logits.append(out.float().cpu())
        if isinstance(model, OCTConvNet) and model.routing_weights():
            alphas.append(model.routing_weights()[-1].cpu())
    result = {"logits": torch.cat(logits).numpy(), "labels": dataset.labels.copy()}
    if alphas:
        result["alpha"] = torch.cat(alphas).numpy()
    return result


def balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = logits.argmax(1)
    recalls = [np.mean(predictions[labels == c] == c) for c in np.unique(labels)]
    return float(np.mean(recalls))


def train(config: TrainConfig, cache_root: Path, out_dir: Path) -> nn.Module:
    """Train one (method, source, seed) run with early stopping on source-validation balanced accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = get_device()
    architecture, objective = METHODS[config.method]
    source = CachedDataset.load(config.source, cache_root)
    train_set, val_set = source.subset("train"), source.subset("val")
    if config.max_train_samples and len(train_set) > config.max_train_samples:
        keep = np.sort(np.random.default_rng(config.seed).choice(len(train_set), config.max_train_samples, replace=False))
        train_set = ScanDataset(train_set.image_path, train_set.indices[keep], source.labels)

    model = build_model(architecture).to(device)
    class_weights = inverse_frequency_weights(train_set.labels, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        num_workers=config.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=config.num_workers > 0, generator=generator)
    noise_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = config.epochs * steps_per_epoch, config.warmup_epochs * steps_per_epoch
    use_amp = config.amp and device.type == "cuda"

    def loss_fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        return F.cross_entropy(logits.float(), labels, weight=class_weights, label_smoothing=config.label_smoothing)

    print(f"[{config.method}/{config.source}/seed{config.seed}] {architecture}, objective={objective}, "
          f"params={count_parameters(model):,}, train={len(train_set)}, val={len(val_set)}, device={device}"
          + (f", sigma={config.sigma}, beta={config.beta}, N={config.realizations}" if objective != "standard" else ""))
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history, best_score, best_epoch, step = [], -1.0, -1, 0
    for epoch in range(config.epochs):
        model.train()
        started, running = time.time(), {"loss": 0.0, "std": 0.0, "aug": 0.0, "correct": 0, "seen": 0}
        for images, labels in loader:
            for group in optimizer.param_groups:
                group["lr"] = warmup_cosine(step, warmup_steps, total_steps, config.lr, config.lr_start, config.lr_end)
            images = augment(to_unit_range(images.to(device, non_blocking=True)), noise_generator)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if objective == "noise_aug":
                # Ablation (3): speckle perturbation used as plain augmentation, no expectation over realizations.
                apply = (torch.rand(images.shape[0], device=device, generator=noise_generator)
                         < config.noise_aug_prob).view(-1, 1, 1, 1)
                images = torch.where(apply, speckle_noise(images, config.sigma, noise_generator), images)
            loss_std = loss_fn(images, labels)
            loss_std.backward()
            loss_aug = torch.zeros((), device=device)
            if objective == "noise_aware":
                # Eqs. (5)-(6): L = L_std + beta * (1/N) sum_n l(f(Z(x^(n))), y). Each realization is a separate
                # forward pass through the shared routers; gradients are accumulated to bound GPU memory.
                for _ in range(config.realizations):
                    term = loss_fn(speckle_noise(images, config.sigma, noise_generator), labels) / config.realizations
                    (config.beta * term).backward()
                    loss_aug += term.detach()
            optimizer.step()
            step += 1
            running["loss"] += (loss_std.detach() + config.beta * loss_aug).item() * len(labels)
            running["std"] += loss_std.item() * len(labels)
            running["aug"] += loss_aug.item() * len(labels)
            running["seen"] += len(labels)

        val = predict(model, val_set, device, num_workers=config.num_workers, amp=config.amp)
        val_bacc = balanced_accuracy(val["logits"], val["labels"])
        seen = running["seen"]
        record = {"epoch": epoch + 1, "loss": running["loss"] / seen, "loss_std": running["std"] / seen,
                  "loss_aug": running["aug"] / seen, "val_balanced_acc": val_bacc,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        history.append(record)
        improved = val_bacc > best_score
        if improved:
            best_score, best_epoch = val_bacc, epoch + 1
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(f"  epoch {epoch + 1:3d} loss={record['loss']:.4f} val_bacc={val_bacc:.4f} "
              f"lr={record['lr']:.2e} {record['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if epoch + 1 - best_epoch >= config.patience:
            print(f"  early stop at epoch {epoch + 1}; best epoch {best_epoch} (val_bacc={best_score:.4f})")
            break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    (out_dir / "summary.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "best_val_balanced_acc": best_score, "epochs_run": len(history),
         "mean_epoch_seconds": float(np.mean([h["seconds"] for h in history]))}, indent=2))
    return model
