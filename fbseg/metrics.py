"""Segmentation metrics that expose blurred boundaries and lost small defects, not only overlap."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _edges(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


class SegmentationMeter:
    """Accumulates binary-segmentation statistics over a test set.

    - Dice / IoU / precision / recall: global (pooled pixels) plus mean per-image Dice (empty-vs-empty = 1).
    - Boundary F1 with a pixel tolerance (pooled), HD95 averaged over images where both masks are non-empty.
    - Component recall split into small and large defects (a component counts as found when >= 25 % of its
      pixels are predicted), plus false-positive components per image.
    """

    def __init__(self, tolerance: int = 2, small_area: int = 150, found_fraction: float = 0.25):
        self.tolerance, self.small_area, self.found_fraction = tolerance, small_area, found_fraction
        self.tp = self.fp = self.fn = 0
        self.per_image_dice: list[float] = []
        self.boundary = np.zeros(4)  # matched pred edges, pred edges, matched gt edges, gt edges
        self.hd95: list[float] = []
        self.components = {"small": [0, 0], "large": [0, 0]}  # found, total
        self.false_components = 0
        self.images = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction, target = prediction.astype(bool), target.astype(bool)
        self.images += 1
        tp = int((prediction & target).sum())
        fp, fn = int((prediction & ~target).sum()), int((~prediction & target).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        self.per_image_dice.append(1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))

        pred_edges, gt_edges = _edges(prediction), _edges(target)
        if pred_edges.any() and gt_edges.any():
            to_gt = ndimage.distance_transform_edt(~gt_edges)
            to_pred = ndimage.distance_transform_edt(~pred_edges)
            self.boundary += [(to_gt[pred_edges] <= self.tolerance).sum(), pred_edges.sum(),
                              (to_pred[gt_edges] <= self.tolerance).sum(), gt_edges.sum()]
            self.hd95.append(float(np.percentile(np.concatenate([to_gt[pred_edges], to_pred[gt_edges]]), 95)))
        else:
            self.boundary += [0, pred_edges.sum(), 0, gt_edges.sum()]

        labels, count = ndimage.label(target)
        for component in range(1, count + 1):
            region = labels == component
            key = "small" if region.sum() < self.small_area else "large"
            self.components[key][1] += 1
            self.components[key][0] += int(prediction[region].mean() >= self.found_fraction)
        pred_labels, pred_count = ndimage.label(prediction)
        if pred_count:
            hits = np.unique(pred_labels[target & prediction])
            self.false_components += pred_count - len(hits[hits > 0])

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        matched_pred, pred_edges, matched_gt, gt_edges = self.boundary
        bp = matched_pred / pred_edges if pred_edges else 0.0
        br = matched_gt / gt_edges if gt_edges else 0.0

        def ratio(found_total: list[int]) -> float:
            return 100.0 * found_total[0] / found_total[1] if found_total[1] else float("nan")

        return {
            "dice": 100.0 * 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": 100.0 * tp / max(1, tp + fp + fn),
            "precision": 100.0 * tp / max(1, tp + fp),
            "recall": 100.0 * tp / max(1, tp + fn),
            "mean_image_dice": 100.0 * float(np.mean(self.per_image_dice)) if self.per_image_dice else float("nan"),
            "boundary_f1": 100.0 * (2 * bp * br / (bp + br) if bp + br else 0.0),
            "hd95": float(np.mean(self.hd95)) if self.hd95 else float("nan"),
            "small_defect_recall": ratio(self.components["small"]),
            "large_defect_recall": ratio(self.components["large"]),
            "n_small_defects": self.components["small"][1],
            "n_large_defects": self.components["large"][1],
            "false_components_per_image": self.false_components / max(1, self.images),
        }
