#!/usr/bin/env bash
# Pre-cache models + datasets so subsequent runs are offline-ish.
#
# Behind a corporate proxy that MITMs HTTPS with a self-signed CA chain:
# Python's `requests` library uses certifi's bundle, not the system one, so
# even when curl + uv work, HF / spaCy fail with CERTIFICATE_VERIFY_FAILED.
# We auto-point requests + ssl + huggingface_hub at the system CA bundle,
# which curl already trusts. Override by exporting REQUESTS_CA_BUNDLE
# yourself before running this script.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${REQUESTS_CA_BUNDLE:-}" ]; then
    for p in /etc/ssl/certs/ca-certificates.crt \
             /etc/pki/tls/certs/ca-bundle.crt \
             /etc/ssl/cert.pem \
             /usr/local/etc/openssl/cert.pem; do
        if [ -f "$p" ]; then
            export REQUESTS_CA_BUNDLE="$p"
            export SSL_CERT_FILE="$p"
            export CURL_CA_BUNDLE="$p"
            echo "# using system CA bundle: $p"
            break
        fi
    done
fi

SPACY_VER="3.8.0"
echo "# installing spaCy model wheels (direct, bypasses compatibility lookup)"
uv pip install \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-${SPACY_VER}/en_core_web_lg-${SPACY_VER}-py3-none-any.whl" \
    "https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-${SPACY_VER}/zh_core_web_lg-${SPACY_VER}-py3-none-any.whl"

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
