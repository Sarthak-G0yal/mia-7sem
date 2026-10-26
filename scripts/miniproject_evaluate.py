"""MINI-PROJECT: evaluate already-trained feedback-smoothing segmenters (no training).

    uv run python scripts/miniproject_evaluate.py                       # every checkpoint under runs_seg/duke_dme/
    uv run python scripts/miniproject_evaluate.py --methods unet feedback
    uv run python scripts/miniproject_evaluate.py --noise 0 0.1 0.2 0.3 0.4

Rebuilds each run's metrics.json and qualitative.png, runs_seg/<dataset>/{report.md, summary.json} and the
figures in runs_seg/<dataset>/figures/.
"""

from __future__ import annotations

import argparse

from _common import setup

setup()

from pathlib import Path  # noqa: E402

from fbseg import DATA_ROOT, RUNS_ROOT  # noqa: E402
from fbseg.data import SegData  # noqa: E402
from fbseg.evaluation import NOISE_LEVELS  # noqa: E402
from fbseg.experiment import evaluate_saved  # noqa: E402
from fbseg.models import METHODS  # noqa: E402
from fbseg.plots import save_all_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="duke_dme")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), help="default: all with checkpoints")
    parser.add_argument("--noise", nargs="+", type=float, default=list(NOISE_LEVELS),
                        help="extra test-time speckle levels")
    args = parser.parse_args()
    report, _ = evaluate_saved(args.dataset, args.methods, tuple(args.noise))
    print(report)
    save_all_figures(Path(RUNS_ROOT), args.dataset, SegData.load(args.dataset, Path(DATA_ROOT)))


if __name__ == "__main__":
    main()
