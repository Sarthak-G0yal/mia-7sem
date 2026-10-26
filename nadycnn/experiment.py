"""Experiment orchestration shared by main.py and the scripts/ entry points.

- run_one / run_experiment: train, predict on every zero-shot target, score, write runs/report.md
- reevaluate: score existing checkpoints again (optionally re-predicting from best.pt) without training
"""

from __future__ import annotations

import itertools
import json
from dataclasses import fields
from pathlib import Path

import torch

from .evaluation import build_report, evaluate_run, profile_models, save_predictions
from .models import build_model
from .training import METHODS, TrainConfig, get_device, train

CACHE_ROOT = Path("cache")
RUNS_ROOT = Path("runs")
MAIN_METHODS = ("basic", "dynamic", "nadycnn", "mobilenetv3", "resnet18")


def _print_results(results: dict) -> None:
    for shift, result in results.items():
        m = result["metrics"]
        site = result.get("conformal_site_specific")
        print(f"  {shift}: F2={m['weighted_f2']:.2f} acc={m['accuracy']:.2f} kappa={m['kappa']:.3f}"
              + (f" | site-specific CP coverage={site['coverage']:.1f}% set={site['set_size']:.2f}" if site else ""))


def run_one(config: TrainConfig, cache_root: Path = CACHE_ROOT, runs_root: Path = RUNS_ROOT,
            overwrite: bool = False) -> dict:
    """Train one (method, source, seed), save target predictions, and write metrics.json."""
    out_dir = runs_root / config.source / config.method / f"seed{config.seed}"
    if (out_dir / "metrics.json").exists() and not overwrite:
        print(f"skip {out_dir} (already evaluated)")
        return json.loads((out_dir / "metrics.json").read_text())
    model = train(config, cache_root, out_dir)
    save_predictions(model, config.source, out_dir, cache_root)
    results = evaluate_run(out_dir, config.source, config.seed)
    _print_results(results)
    return results


def run_experiment(sources: list[str], seeds: list[int], methods: list[str] = list(MAIN_METHODS),
                   cache_root: Path = CACHE_ROOT, runs_root: Path = RUNS_ROOT, overwrite: bool = False,
                   bootstrap: int = 2000, **overrides) -> str:
    """Every run behind Tables 3-8 (adds the dynamic_aug ablation for OCT-C8), then the report.
    `overrides` are TrainConfig fields (epochs, patience, max_train_samples, num_workers, ...)."""
    methods = list(dict.fromkeys(methods + (["dynamic_aug"] if "d2" in sources else [])))
    for source, seed, method in itertools.product(sources, seeds, methods):
        if method == "dynamic_aug" and source != "d2":
            continue  # the ablation (Table 8) is only reported for OCT-C8 -> NEH
        run_one(TrainConfig(method=method, source=source, seed=seed, **overrides), cache_root, runs_root, overwrite)
    return write_report(runs_root, bootstrap)


def load_trained(run_dir: Path) -> tuple[torch.nn.Module, TrainConfig]:
    """Rebuild a model from a run directory's config.json and best.pt."""
    stored = json.loads((run_dir / "config.json").read_text())
    config = TrainConfig(**{f.name: stored[f.name] for f in fields(TrainConfig) if f.name in stored})
    model = build_model(METHODS[config.method][0]).to(get_device())
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=get_device()))
    return model.eval(), config


def reevaluate(runs_root: Path = RUNS_ROOT, cache_root: Path = CACHE_ROOT, repredict: bool = False,
               bootstrap: int = 2000) -> str:
    """Score every finished run again without training. With repredict=True (or when predictions are
    missing) the checkpoint is reloaded and all source/target splits are predicted again first."""
    for run_dir in sorted(runs_root.glob("d[12]/*/seed*")):
        if not (run_dir / "best.pt").exists():
            continue
        source, seed = run_dir.parts[-3], int(run_dir.name[4:])
        if repredict or not (run_dir / f"pred_{source}_val.npz").exists():
            model, _ = load_trained(run_dir)
            save_predictions(model, source, run_dir, cache_root)
        print(f"[{run_dir}]")
        _print_results(evaluate_run(run_dir, source, seed))
    return write_report(runs_root, bootstrap)


def write_report(runs_root: Path = RUNS_ROOT, bootstrap: int = 2000) -> str:
    report = build_report(runs_root, bootstrap)
    (runs_root / "report.md").write_text(report)
    print(f"wrote {runs_root / 'report.md'}")
    return report


def write_profile(runs_root: Path = RUNS_ROOT) -> str:
    """Table 2 (params, MACs, batch-1 latency) for this machine, saved as runs/profile.md."""
    lines = ["| Model | Params (M) | MACs (M) | Latency (ms) | FPS |", "|---|---|---|---|---|"]
    for name, row in profile_models().items():
        lines.append(f"| {name} | {row['params_m']:.2f} | {row['macs_m']:.1f} | "
                     f"{row['latency_ms']:.2f} ± {row['latency_std_ms']:.2f} | {row['fps']:.0f} |")
    table = "\n".join(lines) + "\n"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "profile.md").write_text(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else get_device()}\n\n" + table)
    return table
