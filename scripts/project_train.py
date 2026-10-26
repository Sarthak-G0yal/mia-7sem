"""PROJECT: NA-DyCNN paper reproduction, end to end (prepare -> train -> zero-shot evaluation -> report).

    uv run python scripts/project_train.py                                   # full protocol: D1 + D2, seeds 0 1 2
    uv run python scripts/project_train.py --sources d2 --seeds 0            # one source, one seed
    uv run python scripts/project_train.py --methods nadycnn dynamic --max-train-samples 20000

Writes runs/<source>/<method>/seed<k>/{best.pt, history.json, pred_*.npz, metrics.json}, runs/report.md
(Tables 3, 4, 6, 7, 8 and the routing probe) and runs/profile.md (Table 2). Finished runs are skipped,
so the command can be restarted after an interruption.
"""

from __future__ import annotations

import argparse
import time

from _common import gpu_argument, setup

setup()

from pathlib import Path  # noqa: E402

from nadycnn.data import build_cache  # noqa: E402
from nadycnn.experiment import CACHE_ROOT, MAIN_METHODS, RUNS_ROOT, run_experiment, write_profile  # noqa: E402
from nadycnn.training import METHODS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=["d1", "d2"], default=["d2", "d1"],
                        help="training datasets (d2 first: it is ~8x smaller than d1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(MAIN_METHODS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, help="subsample the source training split")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-prepare", action="store_true", help="cache/ already built")
    parser.add_argument("--no-profile", action="store_true", help="skip the Table 2 latency benchmark")
    parser.add_argument("--overwrite", action="store_true", help="retrain runs that already have metrics")
    gpu_argument(parser)
    args = parser.parse_args()

    started = time.time()
    if not args.skip_prepare:
        for dataset in ("d2", "d3", "d1"):  # quality control + 256x256 uint8 cache (skipped if present)
            build_cache(dataset, CACHE_ROOT)
    run_experiment(args.sources, args.seeds, args.methods, CACHE_ROOT, RUNS_ROOT, args.overwrite, args.bootstrap,
                   epochs=args.epochs, patience=args.patience, max_train_samples=args.max_train_samples,
                   num_workers=args.workers)
    if not args.no_profile:
        print(write_profile(RUNS_ROOT))
    print(f"done in {(time.time() - started) / 3600:.2f} h -> {Path(RUNS_ROOT) / 'report.md'}")


if __name__ == "__main__":
    main()
