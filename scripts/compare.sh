#!/usr/bin/env bash
# Run the full comparison matrix. Pass --test for a 50-sample-per-dataset smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
exec uv run python -m src.compare "$@"
