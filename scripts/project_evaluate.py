"""PROJECT: evaluate already-trained NA-DyCNN runs (no training).

    uv run python scripts/project_evaluate.py               # re-score saved predictions, rebuild runs/report.md
    uv run python scripts/project_evaluate.py --repredict   # reload every best.pt and predict all splits again
    uv run python scripts/project_evaluate.py --profile     # also benchmark Table 2 on this machine

Needs runs/<source>/<method>/seed<k>/{config.json, best.pt} and the preprocessed cache/.
"""

from __future__ import annotations

import argparse

from _common import setup

setup()

from nadycnn.experiment import CACHE_ROOT, RUNS_ROOT, reevaluate, write_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repredict", action="store_true", help="predict again from best.pt before scoring")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(reevaluate(RUNS_ROOT, CACHE_ROOT, args.repredict, args.bootstrap))
    if args.profile:
        print(write_profile(RUNS_ROOT))


if __name__ == "__main__":
    main()
