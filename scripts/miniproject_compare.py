"""MINI-PROJECT: comparative analysis across runs (no training, no GPU; reads metrics.json files only).

    uv run python scripts/miniproject_compare.py
    uv run python scripts/miniproject_compare.py --runs run2=runs_seg_v2 run3=runs_seg_v3 --reference unet

For every run folder, method, test noise level and metric it reports the mean, the spread, and a 95% confidence
interval; and, paired by (fold, seed) against a reference method, the mean difference with a 95% CI, how many
runs and folds are ahead, and two significance tests:

  * fold-level: an exact sign-flip permutation test on the 5 fold means. This is the honest test, because the
    3 seeds of one fold share the same test subjects. With 5 folds its smallest possible p is 2/32 = 0.0625,
    so nothing can reach p < 0.05 at this dataset size.
  * run-level: Wilcoxon signed-rank on the 15 run differences. It treats seeds as independent, so it is
    optimistic; it is shown for reference only.

CIs come from a cluster bootstrap that resamples whole folds. Writes <out>/comparison.md and comparison.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ("run1=results_mini_project_1/runs_seg", "run2=runs_seg_v2", "run3=runs_seg_v3")
METRICS = {"dice": "Dice", "boundary_f1": "Boundary F1", "hd95": "HD95 (px, lower better)",
           "small_defect_recall": "Small-defect recall", "false_components_per_image": "FP blobs/img (lower better)"}
LOWER_IS_BETTER = {"hd95", "false_components_per_image"}
EXTRA_PAIRS = (("open_loop_noise", "feedback_noise"), ("feedback_noise", "feedback"), ("open_loop_noise", "open_loop"))
REGIONS = ("boundary", "defect_interior", "background")
RNG = np.random.default_rng(0)


def load(root: Path, dataset: str) -> dict:
    """runs[method][(fold, seed)] = metrics.json content."""
    runs = defaultdict(dict)
    for path in sorted((root / dataset).glob("*/fold*_seed*/metrics.json")):
        result = json.loads(path.read_text())
        runs[path.parts[-3]][(result["fold"], result["seed"])] = result
    return runs


def cluster_ci(values_by_fold: dict[int, list[float]], draws: int = 5000) -> tuple[float, float]:
    """95% CI of the mean, resampling whole folds with replacement."""
    sums = np.array([np.sum(v) for v in values_by_fold.values()])
    counts = np.array([len(v) for v in values_by_fold.values()])
    picked = RNG.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[picked].sum(1) / counts[picked].sum(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p(fold_means: list[float]) -> float:
    """Exact two-sided sign-flip permutation p on fold means."""
    observed = abs(np.mean(fold_means))
    flips = [abs(np.mean([s * d for s, d in zip(signs, fold_means)]))
             for signs in itertools.product((1, -1), repeat=len(fold_means))]
    return float(np.mean([f >= observed - 1e-12 for f in flips]))


def describe(runs: dict, method: str, level: str, metric: str) -> dict:
    by_fold = defaultdict(list)
    for (fold, _), result in runs[method].items():
        by_fold[fold].append(result["test"][level][metric])
    values = [v for vs in by_fold.values() for v in vs]
    low, high = cluster_ci(by_fold)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "ci": [low, high], "n": len(values),
            "fold_means": {int(f): float(np.mean(v)) for f, v in sorted(by_fold.items())}}


def paired(runs: dict, method: str, reference: str, level: str, metric: str) -> dict | None:
    keys = sorted(set(runs[method]) & set(runs[reference]))
    if not keys:
        return None
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0  # positive difference = method is better
    by_fold = defaultdict(list)
    for key in keys:
        diff = sign * (runs[method][key]["test"][level][metric] - runs[reference][key]["test"][level][metric])
        by_fold[key[0]].append(diff)
    diffs = [d for ds in by_fold.values() for d in ds]
    fold_means = [float(np.mean(v)) for v in by_fold.values()]
    low, high = cluster_ci(by_fold)
    nonzero = [d for d in diffs if d != 0]
    return {"diff": float(np.mean(diffs)), "ci": [low, high], "runs_ahead": sum(d > 0 for d in diffs), "runs": len(diffs),
            "folds_ahead": sum(d > 0 for d in fold_means), "folds": len(fold_means),
            "p_fold": sign_flip_p(fold_means),
            "p_run_wilcoxon": float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else None,
            "fold_diffs": fold_means}


def drop_vs_reference(runs: dict, method: str, reference: str, clean: str, noisy: str) -> dict:
    """Paired: how many fewer Dice points the method loses from `clean` to `noisy` than the reference does."""
    by_fold = defaultdict(list)
    for key in sorted(set(runs[method]) & set(runs[reference])):
        def drop(name: str) -> float:
            return runs[name][key]["test"][clean]["dice"] - runs[name][key]["test"][noisy]["dice"]
        by_fold[key[0]].append(drop(reference) - drop(method))
    diffs = [d for ds in by_fold.values() for d in ds]
    fold_means = [float(np.mean(v)) for v in by_fold.values()]
    low, high = cluster_ci(by_fold)
    return {"diff": float(np.mean(diffs)), "ci": [low, high], "runs_ahead": sum(d > 0 for d in diffs), "runs": len(diffs),
            "folds_ahead": sum(d > 0 for d in fold_means), "folds": len(fold_means), "p_fold": sign_flip_p(fold_means)}


def analyse(runs: dict, reference: str) -> dict:
    methods = sorted(runs)
    levels = sorted(next(iter(runs[methods[0]].values()))["test"], key=lambda k: float(k.split("_")[1]))
    out = {"levels": levels, "methods": methods, "describe": {}, "vs_reference": {}, "pairs": {}, "lambda": {},
           "dice_per_iteration": {}, "robustness_drop": {}, "drop_vs_reference": {}}
    for method in methods:
        out["describe"][method] = {lv: {m: describe(runs, method, lv, m) for m in METRICS} for lv in levels}
        if method != reference and reference in runs:
            out["vs_reference"][method] = {lv: {m: paired(runs, method, reference, lv, m) for m in METRICS} for lv in levels}
            out["drop_vs_reference"][method] = drop_vs_reference(runs, method, reference, levels[0], levels[-1])
        drops = defaultdict(list)
        for (fold, _), result in runs[method].items():
            drops[fold].append(result["test"][levels[0]]["dice"] - result["test"][levels[-1]]["dice"])
        all_drops = [d for ds in drops.values() for d in ds]
        out["robustness_drop"][method] = {"mean": float(np.mean(all_drops)), "ci": list(cluster_ci(drops))}
        first = next(iter(runs[method].values()))
        if first["test"][levels[0]]["lambda"]:
            last = max(int(k.split("_")[0][4:]) for k in first["test"][levels[0]]["lambda"])
            out["lambda"][method] = {lv: {r: float(np.mean([res["test"][lv]["lambda"][f"iter{last}_{r}"]
                                                             for res in runs[method].values()])) for r in REGIONS}
                                     for lv in levels}
        out["dice_per_iteration"][method] = {lv: [float(x) for x in np.mean(
            [res["test"][lv]["dice_per_iteration"] for res in runs[method].values()], axis=0)] for lv in levels}
    for a, b in EXTRA_PAIRS:
        if a in runs and b in runs:
            out["pairs"][f"{a} vs {b}"] = {lv: {m: paired(runs, a, b, lv, m) for m in METRICS} for lv in levels}
    return out


def fmt_p(p: float | None) -> str:
    return "–" if p is None else f"{p:.3f}"


def report(results: dict, reference: str) -> str:
    lines = ["# Mini-project: comparative analysis", "",
             "Positive differences mean *better than the comparison method* (sign flipped for HD95 and FP blobs). "
             "CIs: 95%, cluster bootstrap over folds. p (fold): exact sign-flip test on 5 fold means, smallest "
             "possible value 0.0625. p (runs): Wilcoxon on 15 runs, optimistic (seeds share test subjects).", ""]
    for run, res in results.items():
        lines += [f"## {run}", "", "### Means (95% CI)", ""]
        for lv in res["levels"]:
            lines += [f"**{lv.replace('sigma_', 'σ = ')}**", "", "| Method | " + " | ".join(METRICS.values()) + " |",
                      "|---" * (len(METRICS) + 1) + "|"]
            for method in res["methods"]:
                cells = [f"{d['mean']:.2f} [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}]"
                         for d in (res["describe"][method][lv][m] for m in METRICS)]
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
            lines.append("")
        comparisons = {f"{m} vs {reference}": v for m, v in res["vs_reference"].items()} | res["pairs"]
        lines += ["### Paired comparisons (Dice)", "",
                  "| Comparison | Noise | Difference [95% CI] | Runs ahead | Folds ahead | p (fold) | p (runs) |",
                  "|---|---|---|---|---|---|---|"]
        for name, by_level in comparisons.items():
            for lv, metrics in by_level.items():
                d = metrics["dice"]
                lines.append(f"| {name} | {lv.replace('sigma_', 'σ=')} | {d['diff']:+.2f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] | "
                             f"{d['runs_ahead']}/{d['runs']} | {d['folds_ahead']}/{d['folds']} | {fmt_p(d['p_fold'])} | "
                             f"{fmt_p(d['p_run_wilcoxon'])} |")
        lines += ["", "### Dice lost from cleanest to noisiest level", "", "| Method | Drop [95% CI] |", "|---|---|"]
        for method, drop in sorted(res["robustness_drop"].items(), key=lambda kv: kv[1]["mean"]):
            lines.append(f"| {method} | {drop['mean']:.2f} [{drop['ci'][0]:.1f}, {drop['ci'][1]:.1f}] |")
        lines += ["", f"### Dice drop saved compared with {reference} (positive = degrades less)", "",
                  "| Method | Points saved [95% CI] | Runs ahead | Folds ahead | p (fold) |", "|---|---|---|---|---|"]
        for method, d in res["drop_vs_reference"].items():
            lines.append(f"| {method} | {d['diff']:+.2f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] | {d['runs_ahead']}/{d['runs']} | "
                         f"{d['folds_ahead']}/{d['folds']} | {fmt_p(d['p_fold'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS), help="label=folder pairs")
    parser.add_argument("--dataset", default="duke_dme")
    parser.add_argument("--reference", default="unet")
    parser.add_argument("--out", default="analysis")
    args = parser.parse_args()
    results = {}
    for spec in args.runs:
        label, folder = spec.split("=", 1)
        runs = load(ROOT / folder, args.dataset)
        if not runs:
            print(f"skip {label}: no metrics under {ROOT / folder / args.dataset}")
            continue
        results[label] = analyse(runs, args.reference)
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(results, indent=1))
    (out / "comparison.md").write_text(report(results, args.reference))
    print(f"wrote {out / 'comparison.md'} and {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
