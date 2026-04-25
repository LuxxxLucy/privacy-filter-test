# privacy-filter-test

Benchmark OpenAI's `openai/privacy-filter` against Microsoft Presidio and AI4Privacy on PII span detection. English (Gretel finance + AI4Privacy 400k) and Chinese (synthetic zh-PII + peoples_daily_ner).

## Setup

```sh
uv sync
./scripts/download.sh    # curl spaCy wheels + hf download for models/datasets
```

## Run

```sh
./scripts/compare.sh --test          # ~50 samples/dataset, smoke test
./scripts/compare.sh                 # full eval
./scripts/test_latency.sh            # length sweep, openai/privacy-filter only
```

Device priority: CUDA → MPS → hard error. No CPU fallback.

Results land in `results/` as JSON; full report-grade numbers also stream to stdout.

## CUDA driver compatibility

`pyproject.toml` pins `torch>=2.5,<2.7` so uv resolves cu124 wheels, which work with NVIDIA drivers reporting CUDA 12.4+. Same constraint as `qwen3guard-test`.
