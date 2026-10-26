"""Train and evaluate feedback-smoothing segmenters with subject-level k-fold cross-validation.

CLI (also importable from the notebook: `from fbseg.experiment import run, evaluate_saved, write_report`):

    uv run python -m fbseg run --dataset duke_dme                       # all methods, 5 folds, 1 seed, 30 epochs
    uv run python -m fbseg run --dataset duke_dme --folds 0 --epochs 10 # quick look: one fold
    uv run python -m fbseg run --dataset duke_dme --methods unet feedback --seeds 0 1 2
    uv run python -m fbseg evaluate --dataset duke_dme                  # score saved checkpoints, no training
    uv run python -m fbseg report --dataset duke_dme                    # rebuild runs_seg/<dataset>/report.md
    add --gpu 1 (nvidia-smi index) or --gpu cpu to choose the device
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import fields
from pathlib import Path

import torch

from nadycnn.devices import describe_device, select_device

from . import DATA_ROOT, RUNS_ROOT
from .data import SegData
from .evaluation import NOISE_LEVELS, build_report, evaluate_segmenter, inter_expert_reference, save_qualitative
from .models import METHODS, FeedbackSegmenter, count_parameters
from .training import SegConfig, get_device, train_segmenter


def evaluate_and_save(model: FeedbackSegmenter, config: SegConfig, data: SegData, out_dir: Path,
                      noise_levels: tuple[float, ...] = NOISE_LEVELS) -> dict:
    """Score a trained model on its fold's test subjects; writes metrics.json (+ qualitative.png)."""
    test = data.split(config.fold, config.folds, config.split_seed)["test"]
    history_path = out_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    results = {"method": config.method, "seed": config.seed, "fold": config.fold,
               "params": count_parameters(model),
               "best_val_dice": max((h["val_dice"] for h in history), default=float("nan")),
               "test": evaluate_segmenter(model, data, test, seed=config.seed, noise_levels=noise_levels),
               "inter_expert": inter_expert_reference(data, test)}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    if METHODS[config.method].smoothing != "none":
        save_qualitative(model, data, test, out_dir / "qualitative.png", title=f"{config.method}, fold {config.fold}")
    clean = results["test"]["sigma_0.0"]
    print(f"  test σ=0: dice={clean['dice']:.2f} boundary_f1={clean['boundary_f1']:.2f} "
          f"small_recall={clean['small_defect_recall']:.1f}")
    return results


def run(dataset: str, methods: list[str], seeds: list[int], epochs: int, folds: list[int] | None = None,
        num_folds: int = 5, overwrite: bool = False, data_root: Path = Path(DATA_ROOT),
        runs_root: Path = Path(RUNS_ROOT)) -> tuple[str, dict]:
    """Train + evaluate every (fold, seed, method); `folds=None` runs all `num_folds` folds."""
    data = SegData.load(dataset, data_root)
    for fold, seed, method in itertools.product(folds if folds is not None else range(num_folds), seeds, methods):
        out_dir = runs_root / dataset / method / f"fold{fold}_seed{seed}"
        if (out_dir / "metrics.json").exists() and not overwrite:
            print(f"skip {out_dir} (done)")
            continue
        config = SegConfig(method=method, dataset=dataset, seed=seed, fold=fold, folds=num_folds, epochs=epochs)
        model = train_segmenter(config, data, out_dir)
        evaluate_and_save(model, config, data, out_dir)
    return write_report(dataset, runs_root)


def load_trained(run_dir: Path) -> tuple[FeedbackSegmenter, SegConfig]:
    """Rebuild a segmenter from a run directory's config.json and best.pt."""
    stored = json.loads((run_dir / "config.json").read_text())
    config = SegConfig(**{f.name: stored[f.name] for f in fields(SegConfig) if f.name in stored})
    model = FeedbackSegmenter(METHODS[config.method]).to(get_device())
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=get_device()))
    return model.eval(), config


def evaluate_saved(dataset: str, methods: list[str] | None = None, noise_levels: tuple[float, ...] = NOISE_LEVELS,
                   data_root: Path = Path(DATA_ROOT), runs_root: Path = Path(RUNS_ROOT)) -> tuple[str, dict]:
    """Evaluate every saved checkpoint under runs_seg/<dataset>/ again (no training) and rebuild the report."""
    data = SegData.load(dataset, data_root)
    run_dirs = sorted(d for d in (runs_root / dataset).glob("*/fold*_seed*") if (d / "best.pt").exists()
                      and (methods is None or d.parent.name in methods))
    if not run_dirs:
        raise FileNotFoundError(f"no checkpoints under {runs_root / dataset}; train first (run / miniproject_train.py)")
    for run_dir in run_dirs:
        print(f"[{run_dir}]")
        model, config = load_trained(run_dir)
        evaluate_and_save(model, config, data, run_dir, noise_levels)
    return write_report(dataset, runs_root)


def write_report(dataset: str, runs_root: Path = Path(RUNS_ROOT)) -> tuple[str, dict]:
    report, summary = build_report(runs_root, dataset)
    (runs_root / dataset / "report.md").write_text(report)
    (runs_root / dataset / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {runs_root / dataset / 'report.md'} and summary.json")
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "evaluate", "report"])
    parser.add_argument("--dataset", default="duke_dme")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--folds", nargs="+", type=int, help="subset of folds to run (default: all)")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", help="GPU index as listed by nvidia-smi (e.g. 0 or 1), or 'cpu'")
    args = parser.parse_args()
    select_device(args.gpu)
    print(f"device: {describe_device()}")
    if args.command == "run":
        print(run(args.dataset, args.methods, args.seeds, args.epochs, args.folds, args.num_folds, args.overwrite)[0])
    elif args.command == "evaluate":
        print(evaluate_saved(args.dataset, args.methods)[0])
    else:
        print(write_report(args.dataset)[0])


if __name__ == "__main__":
    main()
