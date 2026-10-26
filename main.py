"""Command-line entry point for the NA-DyCNN reproduction.

uv run python main.py prepare                         # QC + 256x256 cache for D1/D2/D3
uv run python main.py train --method nadycnn --source d2 --seeds 0 1 2
uv run python main.py experiment --sources d1 d2 --seeds 0 1 2   # all Table 3-8 runs
uv run python main.py tune --source d2                # sigma/beta/N grid (Fig. 4)
uv run python main.py evaluate --repredict           # score saved checkpoints, no training
uv run python main.py report                          # writes runs/report.md
uv run python main.py profile                         # Table 2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from nadycnn import experiment
from nadycnn.data import build_cache
from nadycnn.evaluation import build_report, profile_models
from nadycnn.training import METHODS, TrainConfig, train

CACHE_ROOT = Path("cache")
RUNS_ROOT = Path("runs")


def run_one(
    method: str,
    source: str,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    config = TrainConfig(
        method=method,
        source=source,
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        sigma=args.sigma,
        beta=args.beta,
        realizations=args.realizations,
        max_train_samples=args.max_train_samples,
        num_workers=args.workers,
    )
    return experiment.run_one(config, CACHE_ROOT, RUNS_ROOT, args.overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train",
            "experiment",
            "tune",
            "evaluate",
            "report",
            "profile",
        ],
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="nadycnn")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["basic", "dynamic", "nadycnn", "mobilenetv3", "resnet18"],
    )
    parser.add_argument("--source", choices=["d1", "d2"], default="d2")
    parser.add_argument(
        "--sources", nargs="+", choices=["d1", "d2"], default=["d1", "d2"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--realizations", type=int)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="subsample the source training split (compute budget)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--repredict",
        action="store_true",
        help="evaluate: reload best.pt and predict again before scoring",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        for dataset in ("d2", "d3", "d1"):
            build_cache(dataset, CACHE_ROOT)
    elif args.command == "train":
        for seed in args.seeds:
            run_one(args.method, args.source, seed, args)
    elif args.command == "experiment":
        methods = list(
            dict.fromkeys(
                args.methods + (["dynamic_aug"] if "d2" in args.sources else [])
            )
        )
        for source, seed, method in itertools.product(
            args.sources, args.seeds, methods
        ):
            if method == "dynamic_aug" and source != "d2":
                continue  # the ablation (Table 8) is only reported for OCT-C8 -> NEH
            run_one(method, source, seed, args)
        (RUNS_ROOT / "report.md").write_text(build_report(RUNS_ROOT, args.bootstrap))
    elif args.command == "tune":
        # Fig. 4 grid, selected on source-validation balanced accuracy only (no target access).
        results = []
        for sigma, beta, realizations in itertools.product(
            (0.05, 0.10, 0.15, 0.20), (0.10, 0.30), (1, 2, 4, 6, 8)
        ):
            if realizations != 4 and (sigma, beta) != (0.10, 0.10):
                continue  # paper sweeps N only at sigma = beta = 0.10
            args.sigma, args.beta, args.realizations = sigma, beta, realizations
            out_dir = (
                RUNS_ROOT / "tune" / args.source / f"s{sigma}_b{beta}_n{realizations}"
            )
            config = TrainConfig(
                method="nadycnn",
                source=args.source,
                seed=args.seeds[0],
                epochs=args.epochs,
                patience=args.patience,
                sigma=sigma,
                beta=beta,
                realizations=realizations,
                max_train_samples=args.max_train_samples,
                num_workers=args.workers,
            )
            if not (out_dir / "summary.json").exists() or args.overwrite:
                train(config, CACHE_ROOT, out_dir)
            summary = json.loads((out_dir / "summary.json").read_text())
            results.append({"sigma": sigma, "beta": beta, "N": realizations, **summary})
            print(results[-1])
        best = max(results, key=lambda r: r["best_val_balanced_acc"])
        print(
            f"best: sigma={best['sigma']} beta={best['beta']} N={best['N']} "
            f"(val balanced acc {best['best_val_balanced_acc']:.4f})"
        )
        (RUNS_ROOT / "tune" / f"{args.source}.json").write_text(
            json.dumps(results, indent=2)
        )
    elif args.command == "evaluate":
        experiment.reevaluate(RUNS_ROOT, CACHE_ROOT, args.repredict, args.bootstrap)
    elif args.command == "report":
        report = build_report(RUNS_ROOT, args.bootstrap)
        (RUNS_ROOT / "report.md").write_text(report)
        print(report)
    elif args.command == "profile":
        print(
            "| Model | Params (M) | MACs (M) | Latency (ms) | FPS |\n|---|---|---|---|---|"
        )
        for name, row in profile_models().items():
            print(
                f"| {name} | {row['params_m']:.2f} | {row['macs_m']:.1f} | "
                f"{row['latency_ms']:.2f} ± {row['latency_std_ms']:.2f} | {row['fps']:.0f} |"
            )


if __name__ == "__main__":
    main()
