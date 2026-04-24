"""Device selection. CUDA → MPS → hard error. No silent CPU fallback."""
from __future__ import annotations

import sys

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    sys.stderr.write(
        "[device] No CUDA or MPS device available. CPU fallback is disabled. "
        "Run on a CUDA host (e.g. 3090) or Apple Silicon with MPS.\n"
    )
    sys.exit(2)


def device_str() -> str:
    d = pick_device()
    if d.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "mps"
