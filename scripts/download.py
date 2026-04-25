"""Pre-fetch model weights + datasets into the HF cache.

Idempotent: if already cached, this is a no-op. Run once before benchmarks
to isolate download time from measurement time. spaCy models come in via
pyproject.toml URL deps and are already installed by `uv sync`.
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

MODEL_IDS = [
    "openai/privacy-filter",
    "ai4privacy/llama-ai4privacy-english-anonymiser-openpii",
    "shibing624/bert4ner-base-chinese",
]


def fetch_models() -> None:
    for repo_id in MODEL_IDS:
        print(f"[download] {repo_id}")
        snapshot_download(repo_id=repo_id)


def fetch_datasets() -> None:
    from datasets import load_dataset
    print("[download] gretelai/synthetic_pii_finance_multilingual")
    load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test")
    print("[download] ai4privacy/pii-masking-400k")
    load_dataset("ai4privacy/pii-masking-400k", split="validation")
    print("[download] peoples-daily-ner/peoples_daily_ner (parquet branch)")
    load_dataset(
        "peoples-daily-ner/peoples_daily_ner",
        split="validation",
        revision="refs/convert/parquet",
    )


def main() -> int:
    fetch_models()
    fetch_datasets()
    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
