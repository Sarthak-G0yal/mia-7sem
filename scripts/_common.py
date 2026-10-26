"""Shared setup for the scripts: device choice, run from the repository root, headless plotting, importable packages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nadycnn.devices import describe_device, select_device  # noqa: E402


def gpu_argument(parser: argparse.ArgumentParser) -> None:
    """Document --gpu in a script's --help (setup() has already applied it)."""
    parser.add_argument("--gpu", help="GPU index as listed by nvidia-smi (e.g. 0 or 1), or 'cpu'; "
                                      "default: whatever CUDA_VISIBLE_DEVICES selects, else GPU 0")


def setup() -> Path:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu")
    select_device(pre.parse_known_args()[0].gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")  # no display on the workstation server
    os.chdir(ROOT)  # every package path (data/, cache/, runs/, runs_seg/) is relative to the repo root
    import torch
    print(f"repo: {ROOT} | torch {torch.__version__} | {describe_device()}", flush=True)
    return ROOT
