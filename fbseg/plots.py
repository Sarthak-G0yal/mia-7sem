"""Notebook figures: robustness curves, lambda by region, training curves."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette, assigned in fixed order per method (color follows the method, not its rank).
METHOD_COLORS = {
    "unet_linear": "#2a78d6", "unet": "#eb6834", "fixed_0.33": "#1baf7a", "fixed_0.67": "#eda100",
    "fixed_1.0": "#e87ba4", "open_loop": "#008300", "feedback": "#6250d6", "feedback_linear": "#e34948",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, color=TEXT, fontsize=11, loc="left")
    axis.set_xlabel(xlabel, color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED)


def load_runs(runs_root: Path, dataset: str) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted((runs_root / dataset).glob("*/fold*_seed*/metrics.json")):
        runs[path.parts[-3]].append(json.loads(path.read_text()))
    return {m: runs[m] for m in METHOD_COLORS if m in runs}


def plot_robustness(runs: dict[str, list[dict]], metrics: tuple[str, ...] = ("dice", "boundary_f1", "small_defect_recall")):
    """One panel per metric (one y-axis each): metric vs extra test-time speckle, mean ± std over runs."""
    labels = {"dice": "Dice (%)", "boundary_f1": "Boundary F1 (%)", "small_defect_recall": "Small-defect recall (%)",
              "hd95": "HD95 (px, lower is better)"}
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        for method, results in runs.items():
            levels = sorted(results[0]["test"])
            sigmas = [float(level.split("_")[1]) for level in levels]
            values = np.array([[r["test"][level][metric] for level in levels] for r in results], dtype=float)
            mean, std = np.nanmean(values, 0), np.nanstd(values, 0)
            axis.plot(sigmas, mean, color=METHOD_COLORS[method], linewidth=2, marker="o", markersize=5,
                      label=method, markeredgecolor="white", markeredgewidth=1)
            if len(results) > 1:
                axis.fill_between(sigmas, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        _style(axis, labels.get(metric, metric), "extra speckle σ at test time", labels.get(metric, metric))
    axes[0][-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=TEXT)
    figure.tight_layout()
    return figure


def plot_lambda_regions(runs: dict[str, list[dict]], level: str = "sigma_0.0"):
    """Mean final-iteration lambda per region for the learned-lambda methods (grouped bars)."""
    methods = [m for m in ("open_loop", "feedback", "feedback_linear") if m in runs]
    regions = ("boundary", "defect_interior", "background")
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        last = max(int(k.split("_")[0][4:]) for k in runs[method][0]["test"][level]["lambda"])
        values = [np.mean([r["test"][level]["lambda"][f"iter{last}_{region}"] for r in runs[method]]) for region in regions]
        positions = np.arange(len(regions)) + (i - (len(methods) - 1) / 2) * width
        axis.bar(positions, values, width=width - 0.03, color=METHOD_COLORS[method], label=method)
        for x, v in zip(positions, values):
            axis.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    axis.set_xticks(np.arange(len(regions)), [r.replace("_", " ") for r in regions])
    axis.set_ylim(0, 1)
    _style(axis, f"Smoothing strength λ by region ({level.replace('sigma_', 'σ = ')})", "", "mean λ")
    axis.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    figure.tight_layout()
    return figure


def plot_samples(data, count: int = 3):
    """The scans with the most defect pixels: image, mask, and the 2 px boundary band used by the metrics."""
    import torch
    from .data import boundary_band
    order = torch.argsort(data.masks.sum(dim=(1, 2, 3)), descending=True)[:count]
    figure, axes = plt.subplots(len(order), 3, figsize=(13, 2.4 * len(order)), squeeze=False)
    for row, index in enumerate(order):
        panels = [("B-scan", data.images[index, 0]), ("defect mask", data.masks[index, 0]),
                  ("boundary band (2 px)", boundary_band(data.masks[index:index + 1])[0, 0])]
        for axis, (title, panel) in zip(axes[row], panels):
            axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(title)
        axes[row, 0].text(4, 16, data.names[index], color="white", fontsize=8)
    figure.tight_layout()
    return figure


def plot_scale_space(runs_root: Path, dataset: str, data, method: str = "feedback", run: str = "fold0_seed0"):
    """The trained smoother's levels for the scan with the most defect pixels: lambda = 0, 1/3, 2/3, 1."""
    import torch
    from .experiment import load_trained
    from .training import get_device
    run_dir = runs_root / dataset / method / run
    if not (run_dir / "best.pt").exists():
        return None
    model, _ = load_trained(run_dir)
    index = int(torch.argmax(data.masks.sum(dim=(1, 2, 3))))
    with torch.no_grad():
        levels = model.smoother.stack(data.images[index:index + 1].to(get_device()))
    figure, axes = plt.subplots(1, 4, figsize=(16, 3))
    for axis, level, title in zip(axes, levels, ["λ = 0 (raw)", "λ = 1/3 (denoiser)", "λ = 2/3 (+ σ1)", "λ = 1 (+ σ2)"]):
        axis.imshow(level[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_axis_off()
    figure.tight_layout()
    return figure


def save_all_figures(runs_root: Path, dataset: str, data=None) -> list[Path]:
    """Write every results figure to runs_seg/<dataset>/figures/ (for headless servers)."""
    out_dir = runs_root / dataset / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_root, dataset)
    figures = {}
    if data is not None:
        figures["samples"] = plot_samples(data)
    if runs:
        figures["robustness"] = plot_robustness(runs, ("dice", "boundary_f1", "small_defect_recall"))
        figures["robustness_hd95"] = plot_robustness(runs, ("hd95",))
        if any(m in runs for m in ("open_loop", "feedback", "feedback_linear")):
            for level in ("sigma_0.0", "sigma_0.3"):
                if level in next(iter(runs.values()))[0]["test"]:
                    figures[f"lambda_regions_{level}"] = plot_lambda_regions(runs, level)
        figures["training_curves_fold0"] = plot_training_curves(runs_root, dataset)
    if data is not None:
        scale_space = plot_scale_space(runs_root, dataset, data)
        if scale_space is not None:
            figures["scale_space"] = scale_space
    paths = []
    for name, figure in figures.items():
        paths.append(out_dir / f"{name}.png")
        figure.savefig(paths[-1], dpi=130, bbox_inches="tight")
        plt.close(figure)
    print(f"saved {len(paths)} figures to {out_dir}")
    return paths


def plot_training_curves(runs_root: Path, dataset: str, fold: int = 0, seed: int = 0):
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    for method, color in METHOD_COLORS.items():
        path = runs_root / dataset / method / f"fold{fold}_seed{seed}" / "history.json"
        if path.exists():
            history = json.loads(path.read_text())
            axis.plot([h["epoch"] for h in history], [h["val_dice"] for h in history], color=color, linewidth=2,
                      label=method)
    _style(axis, f"Validation Dice during training (fold {fold}, seed {seed})", "epoch", "val Dice (%)")
    axis.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=TEXT)
    figure.tight_layout()
    return figure
