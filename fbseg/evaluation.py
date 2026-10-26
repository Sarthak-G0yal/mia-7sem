"""Evaluation and reporting.

Questions the evaluation is built to answer:
  1. Does adaptive/feedback smoothing segment better than no smoothing or fixed smoothing?   (Dice, IoU)
  2. Does it keep boundaries and small defects?                     (boundary F1, HD95, small-defect recall)
  3. Is it robust when the scan gets noisier?                       (all metrics at extra speckle sigma)
  4. Does lambda behave as intended?  More smoothing on noisier scans, less on boundaries.  (lambda stats)
  5. Does the loop help?  Dice at each feedback iteration.
  6. Does the hyperspherical head matter?  feedback vs feedback_linear.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
import torch
from torch.utils.data import DataLoader

from .data import SegData, SegSubset, boundary_band, speckle_noise
from .metrics import SegmentationMeter
from .models import FeedbackSegmenter
from .training import get_device

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)
REPORT_METRICS = ("dice", "iou", "boundary_f1", "hd95", "small_defect_recall", "large_defect_recall",
                  "false_components_per_image")


@torch.no_grad()
def evaluate_segmenter(model: FeedbackSegmenter, data: SegData, indices: np.ndarray, seed: int = 0,
                       noise_levels: tuple[float, ...] = NOISE_LEVELS) -> dict:
    device = get_device()
    model.eval()
    subset = SegSubset(data, indices)
    results = {}
    for sigma in noise_levels:
        generator = torch.Generator(device=device).manual_seed(seed + int(sigma * 1000))
        final = SegmentationMeter()
        per_iteration = [SegmentationMeter() for _ in range(model.spec.iterations)]
        lam_sums = defaultdict(float)
        lam_counts = defaultdict(float)
        for images, masks in DataLoader(subset, batch_size=8):
            images = speckle_noise(images.to(device), sigma, generator)
            masks = masks.to(device)
            outputs = model(images)
            for meter, logits in zip(per_iteration, outputs["logits"]):
                for p, m in zip(logits.argmax(1).cpu().numpy(), masks[:, 0].cpu().numpy()):
                    meter.update(p, m)
            for p, m in zip(outputs["logits"][-1].argmax(1).cpu().numpy(), masks[:, 0].cpu().numpy()):
                final.update(p, m)
            band = boundary_band(masks)
            regions = {"boundary": band, "defect_interior": masks * (1 - band),
                       "background": (1 - masks) * (1 - band), "all": torch.ones_like(masks)}
            for step, lam in enumerate(outputs["lambda"]):
                for region, weight in regions.items():
                    lam_sums[(step, region)] += float((lam * weight).sum())
                    lam_counts[(step, region)] += float(weight.sum())
        results[f"sigma_{sigma:.1f}"] = {
            **final.compute(),
            "dice_per_iteration": [m.compute()["dice"] for m in per_iteration],
            "lambda": {f"iter{step}_{region}": lam_sums[(step, region)] / max(1.0, lam_counts[(step, region)])
                       for step, region in lam_sums},
        }
    return results


def inter_expert_reference(data: SegData, indices: np.ndarray) -> dict | None:
    """Expert 2 scored against expert 1 on the same test frames: a practical ceiling for any model."""
    if data.masks_expert2 is None:
        return None
    meter = SegmentationMeter()
    for index in indices:
        meter.update(data.masks_expert2[index, 0].numpy(), data.masks[index, 0].numpy())
    return meter.compute()


@torch.no_grad()
def save_qualitative(model: FeedbackSegmenter, data: SegData, indices: np.ndarray, path: Path,
                     sigma: float = 0.1, count: int = 3, title: str = "") -> None:
    """Noisy input, lambda at the first and last iteration, smoothed image, prediction and ground truth."""
    device = get_device()
    model.eval()
    chosen = indices[np.argsort(-data.masks[indices].sum(dim=(1, 2, 3)).numpy())[:count]]
    images = speckle_noise(data.images[chosen].to(device), sigma, torch.Generator(device=device).manual_seed(0))
    outputs = model(images)
    columns = ["noisy input", "λ (iter 0)", f"λ (iter {model.spec.iterations - 1})", "smoothed", "prediction",
               "ground truth"]
    # A pyplot-free Figure works headless and leaves the notebook's inline backend alone.
    figure = Figure(figsize=(3.2 * len(columns), 2.4 * len(chosen)))
    axes = figure.subplots(len(chosen), len(columns), squeeze=False)
    for row in range(len(chosen)):
        panels = [images[row, 0], outputs["lambda"][0][row, 0], outputs["lambda"][-1][row, 0],
                  outputs["smoothed"][-1][row, 0], outputs["logits"][-1][row].argmax(0).float(), data.masks[chosen[row], 0]]
        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            is_lambda = column in (1, 2)
            shown = axis.imshow(panel.cpu().numpy(), cmap="magma" if is_lambda else "gray", vmin=0, vmax=1)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(columns[column])
            if column == 2:
                figure.colorbar(shown, ax=axis, fraction=0.04, pad=0.02, label="λ")
    figure.suptitle(f"{title or model.spec.smoothing} — extra test-time speckle σ = {sigma}", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def _fmt(values: list[float], digits: int = 2) -> str:
    values = [v for v in values if v == v]  # drop NaN
    if not values:
        return "–"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def build_report(runs_root: Path, dataset: str) -> tuple[str, dict]:
    """Markdown report plus a compact JSON summary (mean over seeds) to paste back for discussion."""
    runs = defaultdict(list)
    for path in sorted((runs_root / dataset).glob("*/fold*_seed*/metrics.json")):
        runs[path.parts[-3]].append(json.loads(path.read_text()))
    order = ["unet_linear", "unet", "fixed_0.33", "fixed_0.67", "fixed_1.0", "open_loop", "feedback", "feedback_linear"]
    methods = [m for m in order if m in runs] + sorted(m for m in runs if m not in order)
    summary = {"dataset": dataset, "methods": {},
               "runs": {m: sorted(f"fold{r['fold']}_seed{r['seed']}" for r in runs[m]) for m in methods}}
    lines = [f"# Feedback smoothing segmentation — {dataset}", "",
             "Mean ± std over (fold, seed) runs; each fold tests different subjects. Runs per method: "
             + ", ".join(f"{m} ({len(runs[m])})" for m in methods), ""]
    reference = next((r["inter_expert"] for rs in runs.values() for r in rs if r.get("inter_expert")), None)
    levels = sorted({k for rs in runs.values() for r in rs for k in r["test"]})
    for level in levels:
        lines += [f"## Test set, extra speckle {level.replace('sigma_', 'σ = ')}", "",
                  "| Method | Dice | IoU | Boundary F1 | HD95 (px) | Small-defect recall | Large-defect recall | FP comps/img |",
                  "|---|---|---|---|---|---|---|---|"]
        for method in methods:
            cells = [_fmt([r["test"][level][k] for r in runs[method]], 3 if k == "false_components_per_image" else 2)
                     for k in REPORT_METRICS]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
            summary["methods"].setdefault(method, {})[level] = {
                k: round(float(np.nanmean([r["test"][level][k] for r in runs[method]])), 3) for k in REPORT_METRICS}
        if reference and level == "sigma_0.0":
            lines.append("| *expert 2 vs expert 1* | " + " | ".join(
                f"{reference[k]:.2f}" if reference[k] == reference[k] else "–" for k in REPORT_METRICS) + " |")
            summary["inter_expert"] = {k: round(float(reference[k]), 3) for k in REPORT_METRICS}
        lines.append("")

    smoothing = [m for m in methods if any("iter0_all" in r["test"][levels[0]]["lambda"] for r in runs[m])
                 and not m.startswith(("unet", "fixed"))]
    if smoothing:
        lines += ["## Smoothing strength λ (final iteration, mean)", "",
                  "Intended behaviour: λ rises with noise, and is lower on defect boundaries than on background. "
                  "λ = 0 raw, 1/3 learned denoiser, 2/3 + Gaussian σ=1, 1 + Gaussian σ=2. Note: low λ on the "
                  "boundary band is partly enforced by the edge prior during training; λ in the defect interior and "
                  "λ vs noise level are not.", "",
                  "| Method | Noise | λ all | λ boundary | λ defect interior | λ background | Dice per iteration |",
                  "|---|---|---|---|---|---|---|"]
        for method in smoothing:
            for level in levels:
                last = max(int(k.split("_")[0][4:]) for k in runs[method][0]["test"][level]["lambda"])
                cells = [_fmt([r["test"][level]["lambda"][f"iter{last}_{region}"] for r in runs[method]], 3)
                         for region in ("all", "boundary", "defect_interior", "background")]
                per_iter = np.mean([r["test"][level]["dice_per_iteration"] for r in runs[method]], axis=0)
                lines.append(f"| {method} | {level.replace('sigma_', 'σ=')} | " + " | ".join(cells) + " | "
                             + " → ".join(f"{d:.2f}" for d in per_iter) + " |")
                summary["methods"][method][level]["lambda"] = {
                    region: round(float(np.mean([r["test"][level]["lambda"][f"iter{last}_{region}"] for r in runs[method]])), 3)
                    for region in ("all", "boundary", "defect_interior", "background")}
                summary["methods"][method][level]["dice_per_iteration"] = [round(float(d), 2) for d in per_iter]
        collapsed = [m for m in smoothing if summary["methods"][m][levels[0]]["lambda"]["all"] < 0.05]
        if collapsed:
            lines += ["", f"**λ collapsed to (almost) no smoothing for: {', '.join(collapsed)}** — the task loss "
                          "found smoothing unhelpful, so those runs behave like a plain U-Net."]
        lines.append("")
    return "\n".join(lines) + "\n", summary
