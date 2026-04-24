#!/usr/bin/env bash
# Length-vs-latency sweep for openai/privacy-filter, batch=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
exec uv run python -m src.latency "$@"
