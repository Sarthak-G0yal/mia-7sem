"""Choose which GPU (or the CPU) a run uses. Shared by main.py, fbseg, the scripts and the notebook.

CUDA_VISIBLE_DEVICES is set before CUDA is first used, so the process only ever sees the chosen card and
"cuda" inside the code always means that card. PCI bus order makes the index match nvidia-smi.
"""

from __future__ import annotations

import os


def select_device(choice: str | int | None) -> None:
    """choice: a GPU index as listed by nvidia-smi ("0", "1", ...), "cpu", or None to keep the environment."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if choice is None:
        return
    choice = str(choice).strip().lower()
    if choice == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hides every GPU -> torch falls back to the CPU
    elif choice.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = choice
    else:
        raise ValueError(f"--gpu must be a GPU index (0, 1, ...) or 'cpu', got {choice!r}")


def describe_device() -> str:
    import torch
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        return f"GPU {visible} ({torch.cuda.get_device_name()})"
    return "CPU" + (" (GPUs hidden by --gpu cpu)" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "")
