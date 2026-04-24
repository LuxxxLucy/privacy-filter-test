"""Run every system over every dataset, print + dump report-grade metrics.

Stdout is the source of truth — everything needed to write the report appears
on stdout. Same payload also written to results/compare_<timestamp>.json.

Combinations executed:
    Gretel-finance EN     × {OpenAI-PF, Presidio-EN, AI4Privacy}
    AI4Privacy 400k EN    × {OpenAI-PF, Presidio-EN, AI4Privacy}
    Synthetic zh-PII      × {OpenAI-PF, Presidio-ZH, bert4ner-zh}
    peoples_daily_ner ZH  × {OpenAI-PF, Presidio-ZH, bert4ner-zh}
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .device import pick_device
from .metrics import score


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_block(device):
    info = {
        "timestamp_utc": _now(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return info


def _run_combo(system_name, system, dataset_name, sample_iter, limit_for_log):
    pred_all = []
    gold_all = []
    t0 = time.perf_counter()
    n = 0
    for text, gold in sample_iter:
        pred = system.predict(text)
        pred_all.append(pred)
        gold_all.append(gold)
        n += 1
    elapsed = time.perf_counter() - t0
    # AI4Privacy English model is a binary PII detector. Collapse both pred and
    # gold to label `_pii` so per-category metrics are not all-zero by mismatch.
    if system_name == "ai4privacy_en":
        pred_all = [[(s, e, "_pii") for s, e, _ in spans] for spans in pred_all]
        gold_all = [[(s, e, "_pii") for s, e, _ in spans] for spans in gold_all]
    metrics = score(pred_all, gold_all)
    metrics["_meta"] = {
        "system": system_name,
        "dataset": dataset_name,
        "n_samples": n,
        "wall_seconds": round(elapsed, 3),
        "mean_seconds_per_sample": round(elapsed / max(n, 1), 4),
    }
    return metrics


def _print_metrics(m):
    meta = m["_meta"]
    print(
        f"\n=== {meta['system']} × {meta['dataset']}  "
        f"(n={meta['n_samples']}, {meta['wall_seconds']}s, "
        f"{meta['mean_seconds_per_sample']*1000:.1f} ms/sample)"
    )
    for mode in ("strict", "relaxed"):
        print(f"  [{mode}]")
        labels = [k for k in m[mode].keys() if k != "micro"]
        for lab in labels + ["micro"]:
            row = m[mode][lab]
            print(
                f"    {lab:18s}  P={row['precision']:.4f}  "
                f"R={row['recall']:.4f}  F1={row['f1']:.4f}  "
                f"(tp={row['tp']} fp={row['fp']} fn={row['fn']})"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="50 samples per dataset")
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
    )
    args = ap.parse_args()

    device = pick_device()
    env = _env_block(device)
    print("# privacy-filter-test :: compare")
    for k, v in env.items():
        print(f"# {k}: {v}")
    print(f"# mode: {'TEST (50/dataset)' if args.test else 'FULL'}")

    from .datasets import (
        ai4privacy_400k,
        gretel_finance,
        peoples_daily,
        synth_zh,
    )
    from .systems import (
        ai4privacy as ai4p_sys,
        bert4ner_zh,
        openai_pf,
        presidio_en,
        presidio_zh,
    )

    print("\n# loading systems …")
    sys_openai = openai_pf.load(device)
    sys_presidio_en = presidio_en.load(device)
    sys_ai4p = ai4p_sys.load(device)
    sys_presidio_zh = presidio_zh.load(device)
    sys_bert4ner = bert4ner_zh.load(device)

    combos = [
        ("openai_pf", sys_openai, "gretel_finance_en", lambda: gretel_finance.iter_samples(args.test)),
        ("presidio_en", sys_presidio_en, "gretel_finance_en", lambda: gretel_finance.iter_samples(args.test)),
        ("ai4privacy_en", sys_ai4p, "gretel_finance_en", lambda: gretel_finance.iter_samples(args.test)),
        ("openai_pf", sys_openai, "ai4privacy_400k_en", lambda: ai4privacy_400k.iter_samples(args.test)),
        ("presidio_en", sys_presidio_en, "ai4privacy_400k_en", lambda: ai4privacy_400k.iter_samples(args.test)),
        ("ai4privacy_en", sys_ai4p, "ai4privacy_400k_en", lambda: ai4privacy_400k.iter_samples(args.test)),
        ("openai_pf", sys_openai, "synth_zh", lambda: synth_zh.iter_samples(args.test)),
        ("presidio_zh", sys_presidio_zh, "synth_zh", lambda: synth_zh.iter_samples(args.test)),
        ("bert4ner_zh", sys_bert4ner, "synth_zh", lambda: synth_zh.iter_samples(args.test)),
        ("openai_pf", sys_openai, "peoples_daily_zh", lambda: peoples_daily.iter_samples(args.test)),
        ("presidio_zh", sys_presidio_zh, "peoples_daily_zh", lambda: peoples_daily.iter_samples(args.test)),
        ("bert4ner_zh", sys_bert4ner, "peoples_daily_zh", lambda: peoples_daily.iter_samples(args.test)),
    ]

    all_results = []
    for sname, sys_, dname, mk_iter in combos:
        try:
            m = _run_combo(sname, sys_, dname, mk_iter(), args.test)
            _print_metrics(m)
            all_results.append(m)
        except Exception as e:
            print(f"\n!! FAILED {sname} × {dname}: {type(e).__name__}: {e}", file=sys.stderr)
            all_results.append({"_meta": {"system": sname, "dataset": dname, "error": str(e)}})

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = {"env": env, "test_mode": args.test, "results": all_results}
    fname = f"compare_{'test_' if args.test else ''}{int(time.time())}.json"
    fpath = Path(args.results_dir) / fname
    fpath.write_text(json.dumps(out, indent=2))
    print(f"\n# results written: {fpath}")


if __name__ == "__main__":
    main()
