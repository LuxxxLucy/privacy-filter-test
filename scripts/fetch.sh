#!/usr/bin/env bash
# Pre-cache models + datasets so subsequent runs are offline-ish.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "# fetching spaCy models"
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download zh_core_web_lg

echo "# pre-fetching HuggingFace models"
uv run python - <<'PY'
from transformers import AutoModelForTokenClassification, AutoTokenizer
for mid in [
    "openai/privacy-filter",
    "ai4privacy/llama-ai4privacy-english-anonymiser-openpii",
    "shibing624/bert4ner-base-chinese",
]:
    print(f"  -> {mid}")
    AutoTokenizer.from_pretrained(mid)
    AutoModelForTokenClassification.from_pretrained(mid)
PY

echo "# pre-fetching datasets"
uv run python - <<'PY'
from datasets import load_dataset
load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test")
print("  -> gretelai/synthetic_pii_finance_multilingual ok")
load_dataset("ai4privacy/pii-masking-400k", split="validation")
print("  -> ai4privacy/pii-masking-400k ok")
load_dataset(
    "peoples-daily-ner/peoples_daily_ner",
    split="validation",
    revision="refs/convert/parquet",
)
print("  -> peoples-daily-ner/peoples_daily_ner ok (parquet branch)")
PY

echo "# done"
