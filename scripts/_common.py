"""Shared setup for the scripts: run from the repository root, headless plotting, importable packages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup() -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")  # no display on the workstation server
    os.chdir(ROOT)  # every package path (data/, cache/, runs/, runs_seg/) is relative to the repo root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "no CUDA (CPU/MPS)"
    print(f"repo: {ROOT} | torch {torch.__version__} | {device}", flush=True)
    return ROOT
