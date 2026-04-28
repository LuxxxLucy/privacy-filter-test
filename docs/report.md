# OpenAI Privacy Filter Performance Report

[[_TOC_]]

# Introduction

[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) is a 1.5B-parameter (50M active) bidirectional token classifier for PII span detection, [released April 2026](https://openai.com/index/introducing-openai-privacy-filter/) under Apache 2.0.
Source on [GitHub](https://github.com/openai/privacy-filter), demo on [Hugging Face Spaces](https://huggingface.co/spaces/openai/privacy-filter), full [model card PDF](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf).
This report measures it on four held-out datasets (two English, two Chinese) against [Microsoft Presidio](https://microsoft.github.io/presidio/) and a small fine-tuned classifier on each track, plus a length-vs-latency sweep on an RTX 3090.

Findings:

1. **Strong on English.** On finance text held out from training, Privacy Filter is the strongest of three systems at relaxed micro F1 0.49, ahead of Presidio (0.41) and a [ModernBERT-100M binary detector](https://huggingface.co/ai4privacy/llama-ai4privacy-english-anonymiser-openpii) (0.24). The 96% F1 on the [model card](https://huggingface.co/openai/privacy-filter) is largely a training-set artifact (measured on [pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k), in the training corpus); on the leakage-clean validation sibling [pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) we measure 0.40.
2. **Weak on Chinese.** Privacy Filter handles ASCII-shaped PII embedded in Chinese (emails, +86 phones, account numbers) at 0.64 relaxed micro F1, but collapses to 0.04 on plain Chinese names and place names, where [`bert4ner-base-chinese`](https://huggingface.co/shibing624/bert4ner-base-chinese) (which pretrains on Chinese) is at 0.71. Format coverage is not language coverage. For Chinese deployments, pair with a Chinese-native NER.
3. **Latency: just above a 200 ms P99 budget on 3090.** P50 is 222–418 ms across 64–4,096 input tokens, BF16 batch=1; even at 64 input tokens the P99 is 230 ms. The "50M active" MoE saving is FFN-only; attention is dense banded with sink-token concat, eager-only (`_supports_sdpa = False`), so wall-clock is attention-bound. A smaller variant or a kernel-level optimization (Viterbi decoding aside) would be needed to bring it under budget.

# 1. Background

The gateway sits between end users and a hosted LLM. One feature of its Output Security Engine is **PII span detection**: identify spans of personally identifiable information so that downstream policy can mask, redact, or block them. This document evaluates Privacy Filter as a building block for that feature.

We comapre three candidate:

- **Rule baseline.** [Microsoft Presidio](https://microsoft.github.io/presidio/) (regex + spaCy NER), permissive license, deterministic.
- **Small fine-tuned classifier.** [`llama-ai4privacy-english-anonymiser-openpii`](https://huggingface.co/ai4privacy/llama-ai4privacy-english-anonymiser-openpii), a ModernBERT-100M binary detector trained on the AI4Privacy corpus.
- **Larger trained-for-the-task classifier.** Privacy Filter has 1.5B parameters with MoE routing, native BIOES head over an 8-category PII taxonomy, 128k context, BF16 ~3 GB on disk.

Notes:

- The training corpus is "primarily English" per the model card. Cross-lingual robustness is reported as a "selected multilingual evaluation" without details, so Chinese is not declared coverage.
- The 96% F1 headline is on `pii-masking-300k`, which is the AI4Privacy corpus Privacy Filter is trained on. We treat that figure as in-distribution training-set performance and measure on different data as heldout out-of-distribution tests.

# 2. Privacy Filter

## 2.1 Architecture and label scheme

Privacy Filter is a pre-norm transformer encoder stack derived from a [gpt-oss](https://github.com/openai/gpt-oss)-architecture autoregressive checkpoint, post-trained as a bidirectional banded-attention token classifier. Per the [model card](https://huggingface.co/openai/privacy-filter):

- 8 transformer blocks, residual width `d_model = 640`.
- Grouped-query attention (14 query heads, 2 KV heads), rotary positional embeddings.
- Banded attention with band size 128 (effective window 257 tokens).
- Sparse MoE feed-forward: 128 experts, top-4 routing per token.
- Token-classification head with 33 logits per position.

The 33 classes are `O` plus a BIOES tag (Begin, Inside, End, Single) for each of 8 categories: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`. BIOES rather than BIO matters for decoding (§2.3).

## 2.2 From logits to char spans

The model outputs `(B, T, 33)` logits. Our adapter takes `text: str` and returns `[(char_start, char_end, category)]` in the 8 OpenAI categories:

1. Tokenize with `return_offsets_mapping=True`.
2. Forward pass; argmax per position over the 33 classes.
3. Walk BIOES into spans: open on `B-X` or `S-X`; extend on `I-X` (same X); close on `E-X` (same X), `O`, or label change.
4. Map subword span boundaries to char offsets via the offset mapping.
5. Drop the `private_` prefix.

Empty/whitespace input short-circuits to `[]`; the eager attention reshape crashes on 0-length sequences.

## 2.3 What we did not do: constrained Viterbi

The [model card](https://huggingface.co/openai/privacy-filter) documents a constrained Viterbi decoder with linear-chain transition scoring and six runtime transition-bias parameters that calibrate precision/recall. Our pipeline uses argmax + a lenient BIOES walk, no Viterbi. Two consequences: strict-boundary F1 in this report under-states the model's achievable strict F1 (a constrained decoder enforces consistent BIOES transitions, reducing boundary fragmentation); and the precision/recall tradeoff is fixed in our setup. Source for the published decoder: [`openai/privacy-filter`](https://github.com/openai/privacy-filter).

We accordingly report both **strict** (`(start, end, label)` exact match) and **relaxed** (label match + char-IoU ≥ 0.5) span F1. Relaxed is the fairer metric for cross-system comparison since boundary conventions differ between datasets, baselines, and the model's tokenizer; strict is reported alongside as a lower bound.

# 3. Methodology

## 3.1 Datasets

| Track | Dataset | Role | Notes |
|---|---|---|---|
| EN | [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) (test, 2,962 EN) | primary | Apache 2.0; held out from any candidate's training. |
| EN | [`ai4privacy/pii-masking-400k`](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) (validation, 17,046 EN) | leakage-adjacent | Validation sibling of [`pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) (the training set). Bounds the leakage gap. |
| zh | Synthetic zh-PII (2,000 samples) | primary | Names from `peoples_daily_ner` PER spans; addresses from [`jiaqianjing/chinese-address-ner`](https://huggingface.co/datasets/jiaqianjing/chinese-address-ner) plus a fallback pool; Chinese ID-cards (mod-11 checksum), +86 phones, 16–19-digit bank cards, ASCII emails, `https://` URLs, `2024年5月3日`-format dates. |
| zh | [`peoples_daily_ner`](https://huggingface.co/datasets/peoples-daily-ner/peoples_daily_ner) (validation, 2,319) | sanity | Plain Chinese news, PER/LOC/ORG ground truth, no formatted PII. |

## 3.2 Baselines

| Track | Rule baseline | Small SLM |
|---|---|---|
| EN | Presidio + `en_core_web_lg` | [`llama-ai4privacy-english-anonymiser-openpii`](https://huggingface.co/ai4privacy/llama-ai4privacy-english-anonymiser-openpii) (ModernBERT-100M, binary) |
| zh | Presidio + `zh_core_web_lg` + custom regex (身份证, 手机号, 银行卡) | [`shibing624/bert4ner-base-chinese`](https://huggingface.co/shibing624/bert4ner-base-chinese) (BERT-base-100M, PER/LOC/ORG) |

The AI4Privacy ModernBERT model is binary (PII / not-PII), so per-category F1 is unavailable; we collapse both prediction and gold to a single `_pii` label for that system only.

## 3.3 Crosswalk and scoring

Each system's native labels are coarsened into the 8 OpenAI categories (`person, address, email, phone, url, date, account_number, secret`) before scoring. The crosswalk was verified by enumerating the actual `pii_spans` / `privacy_mask` label vocabularies on each split. A schema-only crosswalk would silently drop ~2,500 gold spans on `ai4privacy_400k` (`DATEOFBIRTH`, `ACCOUNTNUM`, `PASSWORD` are not in the documented schema).

Span-level F1 at the 8-coarse-class taxonomy. **Strict**: `(start, end, label)` exact match. **Relaxed**: label match + char-IoU ≥ 0.5. Per-category P / R / F1 plus micro-averaged.

Hardware: NVIDIA RTX 3090, CUDA 12.4, PyTorch 2.6, BF16, batch=1. Latency sweep uses 5 warmup + 100 timed iterations per length.

# 4. Quality

## 4.1 English

Relaxed micro F1, full splits:

| System | gretel_finance (n=2,962) | ai4privacy_400k val (n=17,046) |
|---|---:|---:|
| **Privacy Filter** | **0.487** | 0.398 |
| Presidio | 0.405 | 0.299 |
| AI4Privacy ModernBERT-100M | 0.240 | **0.413** |

Privacy Filter wins on the held-out finance corpus. On `ai4privacy_400k`, the small ModernBERT slightly edges it because that dataset's annotation conventions match what ModernBERT was trained on.

Per-category on gretel (relaxed F1, model vs Presidio):

| Category | Privacy Filter | Presidio | Gold count |
|---|---:|---:|---:|
| person | **0.775** | 0.596 | 4,588 |
| address | **0.488** | 0.022 | 2,332 |
| phone | **0.551** | 0.418 | 532 |
| email | 0.661 | **0.964** | 842 |
| date | 0.337 | **0.680** | 6,640 |
| account_number | **0.174** | 0.117 | 502 |
| url | **0.286** | 0.109 | 157 |
| secret | **0.153** | 0.057 | 511 |

Privacy Filter wins on address (Presidio's spaCy `LOCATION` flags cities and countries, not street addresses), person (precision 0.79 vs 0.51, Presidio over-flags company names), and the long-tail identifier categories where Privacy Filter's training covers more formats than Presidio's regex set. Presidio wins on email (a well-formed regex is hard to beat) and date (Privacy Filter is conservative about labelling raw numeric tuples; recall 0.24).

The strict/relaxed gap is large: 0.224 strict vs 0.487 relaxed micro F1 on gretel. The same ≥2× drop appears for every system on every dataset, so the cause is annotation-vs-tokenizer boundary mismatch. The constrained Viterbi decoder (§2.3) could have close some of this gap.

## 4.2 Chinese

Relaxed micro F1, full splits:

| System | synth_zh (n=2,000) | peoples_daily_zh (n=2,319) |
|---|---:|---:|
| **Privacy Filter** | **0.642** | 0.043 |
| Presidio (zh + 身份证/手机号/银行卡) | 0.370 | 0.682 |
| `bert4ner-base-chinese` | 0.315 | **0.712** |

`synth_zh` mixes Chinese names and place names with **ASCII-format PII** (`+86` phones, ASCII emails, 18-digit ID cards, 16–19-digit bank cards, `https://` URLs). Privacy Filter dominates at 0.64. `peoples_daily` is bare Chinese news with Chinese names and place names only, no formatted PII; Privacy Filter collapses to 0.04, while a Chinese-NER baseline trained on the same family scores 0.71.

**Format coverage is not language coverage.** Privacy Filter recognizes the *shapes* of PII across languages because its training data contains those shapes (an `@` in any language is an email). It does not generalize to Chinese personal names embedded in Chinese narrative, because that is a language-specific token pattern absent from a primarily-English training mix.

Per-category on synth_zh (relaxed):

| Category | Privacy Filter | Presidio_zh | bert4ner_zh |
|---|---:|---:|---:|
| email | **0.941** | 0 | 0 |
| account_number | **0.581** | 0 | 0 |
| phone (+86) | **0.640** | 0 | 0 |
| address | **0.817** | 0 | 0.220 |
| URL (`https://`) | **0.544** | 0 | 0 |
| person (Chinese names) | 0.684 | 0.828 | **0.899** |
| date (`2024年5月3日`) | 0.412 | **0.843** | 0 |
| secret (Chinese 身份证) | 0 | 0 | 0 |

Per-category on peoples_daily_zh (relaxed):

| Category | Privacy Filter | Presidio_zh | bert4ner_zh |
|---|---:|---:|---:|
| person | 0.123 | 0.805 | **0.945** |
| address (LOC-coarsened) | 0.002 | 0.793 | **0.828** |

Privacy Filter labels 7% of personal names and ~0% of locations. Some of the address gap is semantic (`peoples_daily` LOC includes country and region names, while Privacy Filter's `private_address` is concrete street addresses). The person gap (0.12 vs 0.95) is the load-bearing finding.

# 5. Latency

Our gateway target is **200 ms P99** for an output-side moderation hop. P50 / P99 latency on RTX 3090 (BF16, batch=1, 100 timed iterations per length, 5 warmup):

| Input tokens | P50 (ms) | P95 (ms) | P99 (ms) | mean (ms) |
|---:|---:|---:|---:|---:|
| 64 | 222.4 | 227.8 | 229.9 | 221.8 |
| 256 | 278.8 | 283.2 | 287.5 | 276.0 |
| 1,024 | 298.0 | 312.3 | 315.1 | 301.3 |
| 4,096 | 417.6 | 426.4 | 430.2 | 419.0 |

P99 is above the 200 ms target at every length, by 30 ms at the shortest input and by 230 ms at 4,096 tokens. The 50M-active MoE saving is FFN-only (top-4 of 128 experts per token). Attention is dense banded with sink-token concat in the kernel, which materialises the full L×L attention matrix per layer. SDPA is unavailable: the modeling code declares `_supports_sdpa = False` because the sink concatenation is not compatible with the standard SDPA path. Eager is the only path, so attention dominates wall-clock; that is also why a smaller variant or a custom-kernel rewrite (sink-token-aware flash-attention) is the most plausible route under budget.

16,384 tokens do not fit on 24 GB: the eager kernel allocates a 16k×16k attention matrix per head with sinks concatenated; the request (~14 GB) plus the 11 GB the weights occupy exceeds VRAM. On 3090, 4,096 is the practical ceiling. Larger GPUs can opt back in via the `--lengths` CLI override.

# 6. Conclusion

1. **English: strong.** On gretel-finance held out from training, Privacy Filter wins at relaxed micro F1 0.49 (Presidio 0.41, ModernBERT-100M 0.24). The 0.96 model-card F1 is in-distribution; the leakage-clean reading on `pii-masking-400k` is 0.40.
2. **Chinese: weak.** ASCII-format PII embedded in Chinese transfers (0.64). Bare Chinese narrative does not (0.04 vs 0.71 for `bert4ner-base-chinese`). For Chinese deployments, pair with a Chinese-native NER.
3. **Latency: above the 200 ms P99 budget.** 222–418 ms P50 across 64–4,096 input tokens; P99 is 30–230 ms over budget. Attention is dense banded and eager-only with no SDPA path, so the MoE active-parameter saving doesn't translate to wall-clock. A smaller variant or a sink-token-aware flash-attention kernel is the route to fit.
4. Two known caveats. The published constrained Viterbi decoder ([source](https://github.com/openai/privacy-filter)) is not in our pipeline; strict F1 should improve and the precision/recall tradeoff would become tunable. Truncation at 4,096 tokens hurts long-document recall (visible in gretel email recall 0.53); production needs a chunking strategy.

# References

- OpenAI Privacy Filter: [model card](https://huggingface.co/openai/privacy-filter) (Apache 2.0), [PDF](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf), [announcement](https://openai.com/index/introducing-openai-privacy-filter/), [source](https://github.com/openai/privacy-filter), [Space](https://huggingface.co/spaces/openai/privacy-filter).
- [Microsoft Presidio](https://microsoft.github.io/presidio/), open-source PII detection (regex + spaCy NER).
- AI4Privacy: [training set `pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k); [validation `pii-masking-400k`](https://huggingface.co/datasets/ai4privacy/pii-masking-400k); [ModernBERT-100M detector](https://huggingface.co/ai4privacy/llama-ai4privacy-english-anonymiser-openpii).
- [Gretel `synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual), Apache 2.0.
- [`peoples_daily_ner`](https://huggingface.co/datasets/peoples-daily-ner/peoples_daily_ner), CoNLL-format PER/LOC/ORG.
- [`shibing624/bert4ner-base-chinese`](https://huggingface.co/shibing624/bert4ner-base-chinese), BERT-base-100M Chinese NER.
- [NVIDIA RTX 3090 spec](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/), 24 GB GDDR6X, ~936 GB/s peak memory bandwidth.

# Appendix

Strict micro F1 across all systems × datasets, for completeness. Strict numbers are dominated by tokenizer/annotation boundary differences and are not the right metric for cross-system comparison.

| System | gretel_en | ai4priv_400k | synth_zh | peoples_daily_zh |
|---|---:|---:|---:|---:|
| Privacy Filter | 0.224 | 0.126 | 0.505 | 0.024 |
| Presidio | 0.343 | 0.289 | 0.343 | 0.652 |
| AI4Privacy 100M | 0.018 | 0.007 | – | – |
| bert4ner-zh | – | – | 0.259 | 0.694 |

Environment: Linux 6.8, Python 3.10, PyTorch 2.6.0+cu124, transformers 4.x, BF16, no SDPA. Quality runs use full splits. Latency runs are single-input forwards (no HF pipeline), 5 warmup + 100 timed iterations per length.
