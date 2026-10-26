"""Evaluation metrics (Sec. 2.6): diagnostic, reliability, uncertainty, conformal, and bootstrap statistics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, fbeta_score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE with equal-mass (quantile) bins over max-probability confidence."""
    confidence, correct = probs.max(1), (probs.argmax(1) == labels).astype(np.float64)
    order = np.argsort(confidence, kind="stable")
    return float(sum(len(chunk) * abs(correct[chunk].mean() - confidence[chunk].mean())
                     for chunk in np.array_split(order, bins) if len(chunk)) / len(labels))


def error_uncertainty(probs: np.ndarray, labels: np.ndarray, percentile: float = 75.0) -> float:
    """Eq. (9): % of misclassified samples whose predictive entropy exceeds the 75th percentile."""
    errors = probs.argmax(1) != labels
    if not errors.any():
        return float("nan")
    uncertainty = entropy(probs)
    return float(100.0 * np.mean(uncertainty[errors] > np.percentile(uncertainty, percentile)))


def selective_risk(probs: np.ndarray, labels: np.ndarray, coverage: float = 0.8) -> float:
    """Risk@80%: error rate (%) on the most confident (lowest-entropy) fraction of samples."""
    keep = np.argsort(entropy(probs), kind="stable")[: int(np.ceil(coverage * len(labels)))]
    return float(100.0 * np.mean(probs[keep].argmax(1) != labels[keep]))


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Table 3 / Table 6 metrics. Sensitivity and specificity are macro-averaged; F2 is support-weighted."""
    predictions = probs.argmax(1)
    classes = np.arange(probs.shape[1])
    matrix = confusion_matrix(labels, predictions, labels=classes)
    present = matrix.sum(1) > 0
    tp = np.diag(matrix)
    fn, fp = matrix.sum(1) - tp, matrix.sum(0) - tp
    tn = matrix.sum() - tp - fn - fp
    return {
        "weighted_f2": 100.0 * fbeta_score(labels, predictions, beta=2, average="weighted", zero_division=0),
        "accuracy": 100.0 * np.mean(predictions == labels),
        "sensitivity": 100.0 * np.mean(tp[present] / (tp + fn)[present]),
        "specificity": 100.0 * np.mean(tn[present] / (tn + fp)[present]),
        "kappa": cohen_kappa_score(labels, predictions),
        "ece": expected_calibration_error(probs, labels),
        "err_unc": error_uncertainty(probs, labels),
        "risk_at_80": selective_risk(probs, labels),
    }


def mondrian_thresholds(probs: np.ndarray, labels: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """Eq. (10): class-conditional (1 - eps) quantiles of nonconformity scores 1 - p_{i,y} with the
    finite-sample correction ceil((n + 1)(1 - eps)) / n; classes with too few samples get q = 1."""
    thresholds = np.ones(probs.shape[1])
    for c in range(probs.shape[1]):
        scores = np.sort(1.0 - probs[labels == c, c])
        rank = int(np.ceil((len(scores) + 1) * (1.0 - epsilon)))
        if 0 < rank <= len(scores):
            thresholds[c] = scores[rank - 1]
    return thresholds


def conformal_metrics(prediction_sets: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    sizes = prediction_sets.sum(1)
    return {"coverage": 100.0 * np.mean(prediction_sets[np.arange(len(labels)), labels]),
            "set_size": float(np.mean(sizes)),
            "singleton": 100.0 * np.mean(sizes == 1)}


def mondrian_conformal(cal_probs: np.ndarray, cal_labels: np.ndarray, test_probs: np.ndarray,
                       test_labels: np.ndarray, epsilon: float = 0.05) -> dict:
    """Eq. (11): C(p) = {y : 1 - p_y <= q_y}. Returns overall and per-class (Table 5) metrics."""
    thresholds = mondrian_thresholds(cal_probs, cal_labels, epsilon)
    sets = (1.0 - test_probs) <= thresholds[None, :]
    per_class = {int(c): conformal_metrics(sets[test_labels == c], test_labels[test_labels == c])
                 for c in np.unique(test_labels)}
    return {**conformal_metrics(sets, test_labels), "per_class": per_class, "thresholds": thresholds.tolist()}


def patient_calibration_split(patients: np.ndarray, labels: np.ndarray, fraction: float = 0.10,
                              min_per_class: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw ~10% of target-domain patients for site-specific calibration, adding patients until every
    class has at least `min_per_class` calibration samples. Returns (calibration_mask, evaluation_mask)."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(patients))
    chosen = set(unique[: max(1, int(round(fraction * len(unique))))])
    remaining = list(unique[len(chosen):])
    calibration = np.isin(patients, list(chosen))
    for c in np.unique(labels):
        candidates = [p for p in remaining if np.any(labels[patients == p] == c)]
        while np.sum(labels[calibration] == c) < min_per_class and candidates:
            patient = candidates.pop(0)
            remaining.remove(patient)
            calibration |= patients == patient
    return calibration, ~calibration


def _fast_metric(name: str, predictions: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray,
                 num_classes: int) -> float:
    if name == "accuracy":
        return 100.0 * np.mean(predictions == labels)
    if name == "risk_at_80":
        keep = np.argsort(uncertainty, kind="stable")[: int(np.ceil(0.8 * len(labels)))]
        return 100.0 * np.mean(predictions[keep] != labels[keep])
    matrix = np.bincount(labels * num_classes + predictions, minlength=num_classes ** 2).reshape(num_classes, -1)
    tp, support, predicted = np.diag(matrix), matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support > 0)
    denom = 4 * precision + recall
    f2 = np.divide(5 * precision * recall, denom, out=np.zeros(num_classes), where=denom > 0)
    return 100.0 * float(np.sum(f2 * support) / support.sum())


def hierarchical_bootstrap(probs_a: list[np.ndarray], probs_b: list[np.ndarray], labels: np.ndarray,
                           patients: np.ndarray, metrics: tuple[str, ...] = ("risk_at_80", "weighted_f2", "accuracy"),
                           iterations: int = 2000, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Paired hierarchical bootstrap of metric(A) - metric(B): resample seeds, then patients within each
    sampled seed. Returns {metric: (mean delta, 2.5th, 97.5th percentile)}."""
    rng = np.random.default_rng(seed)
    num_classes = probs_a[0].shape[1]
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    cached = [[(p.argmax(1), entropy(p)) for p in probs] for probs in (probs_a, probs_b)]
    deltas = {m: np.empty(iterations) for m in metrics}
    for it in range(iterations):
        seeds = rng.integers(0, len(probs_a), len(probs_a))
        values = {m: [] for m in metrics}
        for s in seeds:
            index = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for m in metrics:
                a = _fast_metric(m, cached[0][s][0][index], labels[index], cached[0][s][1][index], num_classes)
                b = _fast_metric(m, cached[1][s][0][index], labels[index], cached[1][s][1][index], num_classes)
                values[m].append(a - b)
        for m in metrics:
            deltas[m][it] = np.mean(values[m])
    return {m: (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) for m, d in deltas.items()}
