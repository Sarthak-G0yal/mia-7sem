"""Zero-shot cross-dataset evaluation (Sec. 2.6) and aggregation of results into the paper's tables."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.flop_counter import FlopCounterMode

from . import CLASSES, NEH_CLASS_INDICES, NEH_CLASSES
from .data import DATASET_NAMES, CachedDataset
from .metrics import (classification_metrics, hierarchical_bootstrap, mondrian_conformal,
                      patient_calibration_split, softmax)
from .models import build_model, count_parameters
from .training import METHODS, get_device, predict

TARGETS = {"d1": ("d2", "d3"), "d2": ("d1", "d3")}
SHIFT_ORDER = (("d1", "d3"), ("d2", "d3"), ("d2", "d1"), ("d1", "d2"))
METHOD_LABELS = {"basic": "Basic CNN", "dynamic": "Dynamic CNN", "dynamic_aug": "Dynamic CNN + Noise Aug",
                 "nadycnn": "**NA-DyCNN**", "mobilenetv3": "MobileNetV3", "resnet18": "ResNet18 (ref)"}
MAIN_METHODS = ("basic", "dynamic", "nadycnn", "mobilenetv3", "resnet18")


def save_predictions(model: torch.nn.Module, source: str, run_dir: Path, cache_root: Path) -> None:
    """Clean forward passes on the source val/test splits and on every zero-shot target."""
    device = get_device()
    jobs = [(source, "val"), (source, "test")] + [(target, "test") for target in TARGETS[source]]
    for dataset_name, split in jobs:
        dataset = CachedDataset.load(dataset_name, cache_root)
        result = predict(model, dataset.subset(split), device)
        result["patients"] = dataset.patients[dataset.indices(split)]
        np.savez_compressed(run_dir / f"pred_{dataset_name}_{split}.npz", **result)


def _probabilities(prediction: dict, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax probabilities; for NEH the DME logit is discarded and the remaining classes renormalized."""
    logits, labels, patients = prediction["logits"], prediction["labels"], prediction["patients"]
    if target != "d3":
        return softmax(logits), labels, patients
    keep = np.isin(labels, NEH_CLASS_INDICES)
    remap = {old: new for new, old in enumerate(NEH_CLASS_INDICES)}
    return (softmax(logits[keep][:, list(NEH_CLASS_INDICES)]),
            np.array([remap[label] for label in labels[keep]]), patients[keep])


def _load(run_dir: Path, dataset: str, split: str) -> dict:
    with np.load(run_dir / f"pred_{dataset}_{split}.npz", allow_pickle=False) as data:
        return dict(data)


def evaluate_run(run_dir: Path, source: str, seed: int, epsilon: float = 0.05) -> dict:
    results = {}
    in_domain = _load(run_dir, source, "test")
    results[f"{source}->{source}"] = {"metrics": classification_metrics(*_probabilities(in_domain, source)[:2])}
    source_val = _load(run_dir, source, "val")
    for target in TARGETS[source]:
        probs, labels, patients = _probabilities(_load(run_dir, target, "test"), target)
        cal_probs, cal_labels, _ = _probabilities(source_val, target)
        cal_mask, eval_mask = patient_calibration_split(patients, labels, seed=seed)
        class_names = NEH_CLASSES if target == "d3" else CLASSES
        results[f"{source}->{target}"] = {
            "metrics": classification_metrics(probs, labels),
            "conformal_zero_shot": mondrian_conformal(cal_probs, cal_labels, probs, labels, epsilon),
            "conformal_site_specific": mondrian_conformal(probs[cal_mask], labels[cal_mask],
                                                          probs[eval_mask], labels[eval_mask], epsilon),
            "confusion_matrix": confusion_matrix(labels, probs.argmax(1), labels=range(len(class_names))).tolist(),
            "class_names": list(class_names),
            "n_test": int(len(labels)), "n_site_calibration": int(cal_mask.sum()),
        }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def profile_models(device: torch.device | None = None, passes: int = 1000) -> dict[str, dict]:
    """Table 2: parameters, MACs and batch-1 latency for a 256x256 grayscale input."""
    device = device or get_device()
    rows = {}
    for architecture in ("basic", "dynamic", "mobilenetv3", "resnet18"):
        model = build_model(architecture).to(device).eval()
        x = torch.rand(1, 1, 256, 256, device=device)
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(x)
        timings = []
        with torch.no_grad():
            for i in range(passes + 50):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 50:
                    timings.append((time.perf_counter() - started) * 1000.0)
        rows[architecture] = {"params_m": count_parameters(model) / 1e6, "macs_m": counter.get_total_flops() / 2e6,
                              "latency_ms": float(np.mean(timings)), "latency_std_ms": float(np.std(timings)),
                              "fps": 1000.0 / float(np.mean(timings))}
    return rows


def routing_dataset_probe(runs_root: Path, method: str, seeds: list[int]) -> tuple[float, float] | None:
    """Sec. 3.7: 5-fold logistic regression predicting dataset of origin (D2 vs D3) from the final-block
    routing coefficients of the D1-trained model. The larger domain is subsampled to balance the classes."""
    scores = []
    for seed in seeds:
        run_dir = runs_root / "d1" / method / f"seed{seed}"
        try:
            alpha_d2, alpha_d3 = _load(run_dir, "d2", "test")["alpha"], _load(run_dir, "d3", "test")["alpha"]
        except (FileNotFoundError, KeyError):
            continue
        rng = np.random.default_rng(seed)
        n = min(len(alpha_d2), len(alpha_d3))
        features = np.concatenate([alpha_d2[rng.choice(len(alpha_d2), n, replace=False)],
                                   alpha_d3[rng.choice(len(alpha_d3), n, replace=False)]])
        origin = np.repeat([0, 1], n)
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        scores.append(cross_val_score(probe, features, origin, cv=folds, scoring="accuracy").mean())
    return (float(np.mean(scores)), float(np.std(scores))) if scores else None


def _fmt(values: list[float], digits: int = 2) -> str:
    if not values:
        return "–"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def build_report(runs_root: Path, bootstrap_iterations: int = 2000) -> str:
    """Aggregate all runs (mean ± std over seeds) into markdown versions of Tables 3-8."""
    runs = defaultdict(dict)  # (source, method) -> {seed: metrics}
    for path in sorted(runs_root.glob("*/*/seed*/metrics.json")):
        source, method, seed = path.parts[-4], path.parts[-3], int(path.parts[-2][4:])
        runs[(source, method)][seed] = json.loads(path.read_text())
    lines = ["# NA-DyCNN reproduction results", "",
             f"Mean ± std over seeds. Runs found: " + ", ".join(
                 f"{s}/{m} ({len(v)} seed{'s' if len(v) > 1 else ''})" for (s, m), v in sorted(runs.items())), ""]

    def collect(source: str, method: str, shift: str, *keys: str) -> list[float]:
        values = []
        for result in runs.get((source, method), {}).values():
            node = result.get(shift)
            for key in keys:
                node = node.get(key) if isinstance(node, dict) else None
            if node is not None:
                values.append(float(node))
        return values

    lines += ["## Table 3 — Cross-dataset diagnostic competence", "",
              "| Train → Test | Model | Weighted F2 (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) |",
              "|---|---|---|---|---|---|"]
    for source, target in SHIFT_ORDER + (("d1", "d1"), ("d2", "d2")):
        shift = f"{source}->{target}"
        for method in MAIN_METHODS:
            if (source, method) in runs:
                cells = [_fmt(collect(source, method, shift, "metrics", k))
                         for k in ("weighted_f2", "accuracy", "sensitivity", "specificity")]
                label = f"{DATASET_NAMES[source]} → {DATASET_NAMES[target]}" + (" (in-domain)" if source == target else "")
                lines.append(f"| {label} | {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")

    lines += ["", "## Table 4 — Mondrian conformal prediction (ε = 0.05)", "",
              "| Shift | Model | Zero-shot coverage (%) | Set size | Singleton (%) "
              "| Site-specific coverage (%) | Set size | Singleton (%) |", "|---|---|---|---|---|---|---|---|"]
    for source, target in SHIFT_ORDER:
        shift = f"{source}->{target}"
        for method in MAIN_METHODS:
            if (source, method) in runs:
                cells = [_fmt(collect(source, method, shift, mode, k), 1 if k != "set_size" else 2)
                         for mode in ("conformal_zero_shot", "conformal_site_specific")
                         for k in ("coverage", "set_size", "singleton")]
                lines.append(f"| {DATASET_NAMES[source]} → {DATASET_NAMES[target]} | {METHOD_LABELS[method]} | "
                             + " | ".join(cells) + " |")

    lines += ["", "## Table 6 — Uncertainty calibration quality", "",
              "| Shift | Model | Cohen's κ | ECE | Err.Unc (%) | Risk@80% (%) |", "|---|---|---|---|---|---|"]
    for source, target in SHIFT_ORDER:
        shift = f"{source}->{target}"
        for method in MAIN_METHODS:
            if (source, method) in runs:
                cells = [_fmt(collect(source, method, shift, "metrics", "kappa"), 3),
                         _fmt(collect(source, method, shift, "metrics", "ece"), 3),
                         _fmt(collect(source, method, shift, "metrics", "err_unc"), 1),
                         _fmt(collect(source, method, shift, "metrics", "risk_at_80"), 2)]
                lines.append(f"| {DATASET_NAMES[source]} → {DATASET_NAMES[target]} | {METHOD_LABELS[method]} | "
                             + " | ".join(cells) + " |")

    lines += ["", "## Table 7 — Hierarchical bootstrap (NA-DyCNN − baseline, 95% CI)", "",
              "| Shift | Versus | ΔRisk@80% | ΔF2 | ΔAccuracy |", "|---|---|---|---|---|"]
    for source, target in SHIFT_ORDER:
        seeds_a = sorted(runs.get((source, "nadycnn"), {}))
        for baseline in ("basic", "dynamic", "mobilenetv3"):
            seeds = [s for s in seeds_a if s in runs.get((source, baseline), {})]
            if not seeds:
                continue
            preds = {m: [_probabilities(_load(runs_root / source / m / f"seed{s}", target, "test"), target)
                         for s in seeds] for m in ("nadycnn", baseline)}
            labels, patients = preds["nadycnn"][0][1], preds["nadycnn"][0][2]
            result = hierarchical_bootstrap([p[0] for p in preds["nadycnn"]], [p[0] for p in preds[baseline]],
                                            labels, patients, iterations=bootstrap_iterations)
            cells = [f"{'**' if lo > 0 or hi < 0 else ''}{mean:+.2f} [{lo:.2f}, {hi:.2f}]{'**' if lo > 0 or hi < 0 else ''}"
                     for mean, lo, hi in (result[m] for m in ("risk_at_80", "weighted_f2", "accuracy"))]
            lines.append(f"| {DATASET_NAMES[source]} → {DATASET_NAMES[target]} | {METHOD_LABELS[baseline]} | "
                         + " | ".join(cells) + " |")

    lines += ["", "## Table 8 — Ablation on OCT-C8 → NEH (site-specific calibration)", "",
              "| Variant | Weighted F2 (%) | Accuracy (%) | Cohen's κ | Set size |", "|---|---|---|---|---|"]
    for method in ("basic", "dynamic", "dynamic_aug", "nadycnn"):
        if ("d2", method) in runs:
            lines.append(f"| {METHOD_LABELS[method]} | {_fmt(collect('d2', method, 'd2->d3', 'metrics', 'weighted_f2'))} | "
                         f"{_fmt(collect('d2', method, 'd2->d3', 'metrics', 'accuracy'))} | "
                         f"{_fmt(collect('d2', method, 'd2->d3', 'metrics', 'kappa'), 3)} | "
                         f"{_fmt(collect('d2', method, 'd2->d3', 'conformal_site_specific', 'set_size'))} |")

    lines += ["", "## Sec. 3.7 — Dataset information in routing coefficients (UCSD-trained, OCT-C8 vs NEH)", "",
              "| Model | Dataset-of-origin probe accuracy (lower = more invariant) |", "|---|---|"]
    for method in ("dynamic", "nadycnn"):
        probe = routing_dataset_probe(runs_root, method, sorted(runs.get(("d1", method), {})))
        if probe:
            lines.append(f"| {METHOD_LABELS[method]} | {probe[0]:.3f} ± {probe[1]:.3f} |")
    return "\n".join(lines) + "\n"


__all__ = ["evaluate_run", "save_predictions", "profile_models", "build_report", "METHODS"]
