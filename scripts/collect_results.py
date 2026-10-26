"""Bundle every report, metrics file and figure (no checkpoints or predictions) into one small zip to share.

    uv run python scripts/collect_results.py            # -> results_bundle/results_<timestamp>.zip

Contents: runs/report.md, runs/profile.md, runs/**/{metrics,history,summary,config}.json and the training log
(project); runs_seg/<dataset>/{report.md, summary.json, figures/*.png} and each run's
{metrics,history,config}.json and qualitative.png (mini-project); logs/*.log.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    "runs/report.md", "runs/profile.md", "runs/*.log",
    "runs/d[12]/*/seed*/metrics.json", "runs/d[12]/*/seed*/history.json",
    "runs/d[12]/*/seed*/summary.json", "runs/d[12]/*/seed*/config.json", "runs/tune/*.json",
    "runs_seg/*/report.md", "runs_seg/*/summary.json", "runs_seg/*/figures/*.png",
    "runs_seg/*/*/fold*_seed*/metrics.json", "runs_seg/*/*/fold*_seed*/history.json",
    "runs_seg/*/*/fold*_seed*/config.json", "runs_seg/*/*/fold*_seed*/qualitative.png",
    "logs/*.log",
)


def main() -> None:
    files = sorted({path for pattern in PATTERNS for path in ROOT.glob(pattern) if path.is_file()})
    if not files:
        raise SystemExit("nothing to collect yet: run scripts/project_train.py or scripts/miniproject_train.py first")
    out_dir = ROOT / "results_bundle"
    out_dir.mkdir(exist_ok=True)
    archive = out_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(ROOT))
    print(f"{len(files)} files, {archive.stat().st_size / 1e6:.1f} MB -> {archive}")
    for summary in ("runs/report.md", *(str(p.relative_to(ROOT)) for p in ROOT.glob("runs_seg/*/report.md"))):
        if (ROOT / summary).exists():
            print(f"  includes {summary}")


if __name__ == "__main__":
    main()
