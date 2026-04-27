"""Length-vs-latency sweep on `openai/privacy-filter`. batch=1.

Inputs are synthetic English text padded to target token lengths with
interspersed PII. P50/P95/P99 over N runs each, reported per length.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .device import pick_device
from .systems import openai_pf

# 16384 is omitted from the default sweep: the model's attention is eager-only
# (_supports_sdpa = False; sink-token concat), so memory grows as O(L^2) and a
# 16k forward pass exceeds 24 GB. Pass --lengths explicitly to opt back in on
# larger GPUs.
LENGTHS = [64, 256, 1024, 4096]

PII_BITS = [
    "Alice Smith works at alice.smith@example.com.",
    "Call +1-202-555-0143 for the order placed on 2025-03-12.",
    "Routing 021000021, account 123456789012.",
    "Bob lives at 1600 Pennsylvania Ave NW, Washington DC.",
]

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
)


def _build_input(target_tokens: int, tok) -> str:
    text = ""
    pii_idx = 0
    while len(tok.encode(text)) < target_tokens:
        text += PII_BITS[pii_idx % len(PII_BITS)] + " " + FILLER
        pii_idx += 1
    ids = tok.encode(text)[:target_tokens]
    return tok.decode(ids)


def _percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="trials per length")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    device = pick_device()
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
    print("# privacy-filter-test :: latency sweep")
    for k, v in env.items():
        print(f"# {k}: {v}")
    print(f"# n_trials_per_length: {args.n}, warmup: {args.warmup}")
    print(f"# lengths: {args.lengths}")

    sys_pf = openai_pf.load(device)
    tok = sys_pf.tok
    model = sys_pf.model

    rows = []
    for L in args.lengths:
        text = _build_input(L, tok)
        inputs = tok(text, return_tensors="pt").to(device)
        actual_len = int(inputs["input_ids"].shape[1])

        # warmup
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times_ms = []
        for _ in range(args.n):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        row = {
            "target_length": L,
            "actual_length": actual_len,
            "p50_ms": round(statistics.median(times_ms), 3),
            "p95_ms": round(_percentile(times_ms, 95), 3),
            "p99_ms": round(_percentile(times_ms, 99), 3),
            "mean_ms": round(statistics.mean(times_ms), 3),
            "min_ms": round(min(times_ms), 3),
            "max_ms": round(max(times_ms), 3),
            "raw_ms": [round(x, 3) for x in times_ms],
        }
        rows.append(row)
        print(
            f"L={L:>5}  actual={actual_len:>5}  "
            f"P50={row['p50_ms']:>8.3f}  P95={row['p95_ms']:>8.3f}  "
            f"P99={row['p99_ms']:>8.3f}  mean={row['mean_ms']:>8.3f}  "
            f"min={row['min_ms']:>8.3f}  max={row['max_ms']:>8.3f}  ms"
        )

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = {"env": env, "n_trials": args.n, "warmup": args.warmup, "rows": rows}
    fpath = Path(args.results_dir) / f"latency_{int(time.time())}.json"
    fpath.write_text(json.dumps(out, indent=2))
    print(f"\n# results written: {fpath}")


if __name__ == "__main__":
    main()
