# privacy-filter-test

Benchmark OpenAI's `openai/privacy-filter` against Microsoft Presidio and AI4Privacy on PII span detection. English (Gretel finance + AI4Privacy 400k) and Chinese (synthetic zh-PII + peoples_daily_ner).

## Setup

```sh
uv sync
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download zh_core_web_lg
./scripts/fetch.sh
```

## Run

```sh
./scripts/compare.sh --test       # ~50 samples/dataset, smoke test
./scripts/compare.sh              # full eval
./scripts/test_latency.sh         # length sweep, openai/privacy-filter only
```

Device priority: CUDA → MPS → hard error. No CPU fallback.

Results land in `results/` as JSON; full report-grade numbers also stream to stdout.
