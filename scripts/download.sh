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

# spaCy model wheels — not on PyPI; only available on Explosion's GitHub releases.
SPACY_VER="3.8.0"
mkdir -p .cache/wheels
for m in en_core_web_lg zh_core_web_lg; do
    whl=".cache/wheels/${m}-${SPACY_VER}-py3-none-any.whl"
    if [ ! -f "$whl" ]; then
        echo "# curl -> ${m}-${SPACY_VER}.whl"
        # -C - resumes partials; --retry covers transient proxy throttles.
        curl -fL --retry 5 --retry-delay 5 -C - -o "$whl" \
            "https://github.com/explosion/spacy-models/releases/download/${m}-${SPACY_VER}/${m}-${SPACY_VER}-py3-none-any.whl"
    fi
done
uv pip install --no-deps .cache/wheels/*.whl

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
