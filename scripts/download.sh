#!/usr/bin/env bash
# Pre-fetch spaCy wheels, HF models, HF datasets. Idempotent.
# Pure curl + hf — no Python heredocs.
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-point Python TLS at the system CA bundle (curl + uv already trust it,
# but Python requests defaults to certifi's bundle which omits proxy CAs).
if [ -z "${REQUESTS_CA_BUNDLE:-}" ]; then
    for p in /etc/ssl/certs/ca-certificates.crt \
             /etc/pki/tls/certs/ca-bundle.crt \
             /etc/ssl/cert.pem \
             /usr/local/etc/openssl/cert.pem; do
        if [ -f "$p" ]; then
            export REQUESTS_CA_BUNDLE="$p"
            export SSL_CERT_FILE="$p"
            export CURL_CA_BUNDLE="$p"
            echo "# CA bundle: $p"
            break
        fi
    done
fi

# spaCy models via HF mirror (Explosion publishes to spacy/<model>).
# Avoids GitHub release URLs which redirect to release-assets.githubusercontent.com
# → Azure blob — that handshake fails on some corporate proxies.
mkdir -p .cache/spacy
for m in en_core_web_lg zh_core_web_lg; do
    echo "# hf model -> spacy/${m}"
    uv run hf download "spacy/${m}" --local-dir ".cache/spacy/${m}" >/dev/null
done

# HF models.
for m in openai/privacy-filter \
         ai4privacy/llama-ai4privacy-english-anonymiser-openpii \
         shibing624/bert4ner-base-chinese; do
    echo "# hf model -> $m"
    uv run hf download "$m" >/dev/null
done

# HF datasets.
for d in gretelai/synthetic_pii_finance_multilingual ai4privacy/pii-masking-400k; do
    echo "# hf dataset -> $d"
    uv run hf download "$d" --repo-type dataset >/dev/null
done

# peoples_daily_ner has only a script loader on main; use the parquet auto-convert branch.
echo "# hf dataset -> peoples-daily-ner/peoples_daily_ner (parquet branch)"
uv run hf download peoples-daily-ner/peoples_daily_ner \
    --repo-type dataset --revision refs/convert/parquet >/dev/null

echo "# done"
