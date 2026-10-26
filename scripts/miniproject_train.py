"""MINI-PROJECT: feedback-controlled smoothing for OCT defect segmentation, end to end.

Script version of notebooks/feedback_segmentation.ipynb: data -> train -> evaluate -> report -> figures.

    uv run python scripts/miniproject_train.py                         # Duke DME, 8 methods, 5 folds, seed 0, 30 epochs
    uv run python scripts/miniproject_train.py --folds 0 --epochs 10   # quick look (~10 min)
    uv run python scripts/miniproject_train.py --seeds 0 1 2           # more seeds for the final numbers

Writes runs_seg/<dataset>/<method>/fold<f>_seed<s>/{best.pt, history.json, metrics.json, qualitative.png},
runs_seg/<dataset>/{report.md, summary.json} and runs_seg/<dataset>/figures/*.png. Finished runs are skipped.
"""

from __future__ import annotations

import argparse
import time
import zipfile

from _common import gpu_argument, setup

setup()

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import download_data  # noqa: E402
from fbseg import DATA_ROOT, RUNS_ROOT  # noqa: E402
from fbseg.data import SegData  # noqa: E402
from fbseg.experiment import run  # noqa: E402
from fbseg.models import METHODS  # noqa: E402
from fbseg.plots import save_all_figures  # noqa: E402


def ensure_dataset(dataset: str, download: bool, fallback: bool) -> str:
    """Make sure data/seg/<dataset> exists: download Duke DME / generate synthetic, optionally falling back."""
    if (Path(DATA_ROOT) / dataset / "images").is_dir():
        return dataset
    if not download:
        raise SystemExit(f"{Path(DATA_ROOT) / dataset} missing; run scripts/download_data.py {dataset}")
    try:
        if dataset == "duke_dme":
            download_data.prepare_duke()
        elif dataset == "synthetic":
            download_data.prepare_synthetic()
        else:
            raise SystemExit(f"no automatic download for {dataset}; see scripts/download_data.py {dataset}")
    except (RuntimeError, OSError, zipfile.BadZipFile) as error:
        if not fallback:
            raise
        print(f"download failed ({error}); falling back to synthetic data")
        download_data.prepare_synthetic()
        dataset = "synthetic"
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="duke_dme", help="duke_dme | synthetic | a folder under data/seg/")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, help="subset of folds (default: all)")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-fallback", action="store_true", help="fail instead of using synthetic data")
    parser.add_argument("--overwrite", action="store_true", help="retrain runs that already have metrics")
    gpu_argument(parser)
    args = parser.parse_args()

    started = time.time()
    dataset = ensure_dataset(args.dataset, not args.no_download, not args.no_fallback)
    data = SegData.load(dataset, Path(DATA_ROOT))
    has_defect = (data.masks.sum(dim=(1, 2, 3)) > 0).float().mean().item()
    print(f"{dataset}: {len(data.names)} B-scans, {len(np.unique(data.subjects))} subjects, "
          f"{100 * has_defect:.0f}% contain a defect, defect pixels {100 * data.masks.mean().item():.2f}%, "
          f"second annotator: {data.masks_expert2 is not None}")

    report, _ = run(dataset, args.methods, args.seeds, args.epochs, args.folds, args.num_folds, args.overwrite)
    print(report)
    save_all_figures(Path(RUNS_ROOT), dataset, data)
    print(f"done in {(time.time() - started) / 60:.1f} min -> {Path(RUNS_ROOT) / dataset / 'summary.json'}")


if __name__ == "__main__":
    main()
