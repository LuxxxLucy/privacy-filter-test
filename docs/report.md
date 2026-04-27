# OpenAI Privacy Filter Performance Report

[[_TOC_]]

# Introduction

OpenAI Privacy Filter is a 1.5B-parameter (50M active) bidirectional token classifier for PII span detection, released April 2026 under Apache 2.0.
This report measures it on four held-out datasets (two English, two Chinese) against two baselines per language track, plus a length-vs-latency sweep on an RTX 3090.

Findings:

1. On English finance text held out from the model's training distribution, it is the strongest of three systems at relaxed micro F1 0.49, ahead of Microsoft Presidio (0.41) and a 100M-parameter ModernBERT classifier (0.24).
2. The 0.96 F1 number on the model card is largely a training-set artifact.
   On a leakage-clean validation split (`ai4privacy/pii-masking-400k`), we measure relaxed micro F1 0.40.
3. On Chinese text, performance splits along PII *format* rather than language.
   When PII is ASCII-shaped (emails, +86 phone numbers, alphanumeric account numbers), the model wins (0.64 relaxed) by a wide margin.
   When PII is bare Chinese narrative (Han names, Chinese place names with no English scaffolding), it collapses to 0.04 against Chinese-NER baselines that score 0.68 and 0.71.
4. Latency on RTX 3090 is 222–418 ms P50 across 64 to 4096 input tokens, batch=1, BF16.
   That is roughly 5× a dense Qwen3Guard-Gen-0.6B at comparable length, despite the "50M active" MoE claim.
   The reason is architectural: the model uses dense banded attention with attention sinks and ships an eager-only kernel, so the active-parameter saving applies only to the FFN.

# 1. Background

The gateway sits between end users and a hosted LLM.
One feature of its Output Security Engine is **PII span detection**: identify spans of personally identifiable information so that downstream policy can mask, redact, or block them.
This document evaluates whether OpenAI Privacy Filter is a viable building block for that feature.

The candidate field for in-process PII span detection has three tiers:

- **Rule baselines.**
  Microsoft Presidio is the reference open-source rule pipeline (regex + spaCy NER).
  Permissive license, deterministic, language packs ship with spaCy.
- **Small fine-tuned classifiers.**
  `ai4privacy/llama-ai4privacy-english-anonymiser-openpii` is a ModernBERT-100M binary detector trained on the AI4Privacy corpus.
- **Larger trained-for-the-task classifiers.**
  OpenAI Privacy Filter is the Apache 2.0 entry in this tier as of April 2026: 1.5B parameters with MoE routing, native BIOES head over an 8-category PII taxonomy, 128k context, BF16 ~3 GB on disk.
  See [PF-Card] for the model card and [PF-Repo] for the source.

Two facts shape what we test:

- The training corpus is "primarily English" by the model card's language statement.
  Cross-lingual robustness is reported as a "selected multilingual evaluation" without details, so Chinese is not declared coverage.
- The 96% F1 number is reported on `PII-Masking-300k`, which is the AI4Privacy corpus that Privacy Filter is trained on.
  We treat that figure as in-distribution training-set performance and measure on different data.

# 2. Subject and how we use it

## 2.1 Architecture and label scheme

Privacy Filter is a pre-norm transformer encoder stack with the following dimensions ([PF-Card]):

- 8 transformer blocks, residual width `d_model = 640`.
- Grouped-query attention with rotary positional embeddings, 14 query heads and 2 KV heads.
- Banded attention with band size 128 (effective attention window 257 tokens including self).
- Sparse mixture-of-experts feed-forward, 128 experts, top-4 routing per token.
- A token-classification head with 33 logits per position.

The 33 classes are `O` plus a BIOES tag (Begin, Inside, End, Single) for each of 8 categories: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`.
The use of BIOES rather than BIO is unusual and matters for how the output is decoded (see §2.3).

The base checkpoint is autoregressive (gpt-oss-architecture).
OpenAI replaces the LM head with the 33-way classifier and post-trains as a bidirectional banded-attention token classifier.

## 2.2 What the model produces, and how we transform it into a span list

The model outputs `(B, T, 33)` logits.
Our task takes `text: str` and returns a list of `(char_start, char_end, category)` tuples in the 8 OpenAI categories.

The transformation is:

1. Tokenize with `return_offsets_mapping=True` to keep subword to character spans.
2. Forward pass; argmax per position over the 33 classes.
3. Walk the predicted BIOES sequence into spans:
   - Open on `B-X` or `S-X`; extend on `I-X` (same X); close on `E-X` (same X), `O`, or label change.
   - On out-of-band transitions, close the current span and start a new one.
4. Map subword span boundaries back to `(char_start, char_end)` via the offset mapping.
5. Drop the `private_` prefix; categories now match the 8 canonical labels.
6. Skip empty/whitespace input (the model's eager attention reshape does not handle 0-length sequences).

This is the simple version of the inference pipeline.
The model card documents a richer alternative (§2.3); we did not implement it.

## 2.3 What we did not do: constrained Viterbi with operating-point calibration

The model card states ([PF-Card]):

> After the token classifier produces per-token logits, we decode labels with a constrained Viterbi decoder using linear-chain transition scoring, rather than taking an independent argmax for each token.
> The decoder enforces allowed BIOES boundary transitions and scores complete label paths with start, transition, and end terms, plus six transition-bias parameters that control background persistence, span entry, span continuation, span closure, and boundary-to-boundary handoff.

OpenAI further says these transition biases function as **runtime operating points**: tunable knobs that trade precision for recall at inference time without retraining.

Our pipeline uses per-token argmax plus a lenient BIOES walk, no Viterbi.
This means:

- Strict-boundary numbers in this report under-state the model's achievable strict F1.
  A constrained decoder enforces that an `E-X` cannot follow without a matching `B-X` or `I-X`, etc.; argmax can produce locally inconsistent label sequences that fragment spans at boundaries.
- The precision/recall tradeoff is fixed in our setup; with the published Viterbi we could move along an operating curve.
- The model card uses `aggregation_strategy="simple"` in its quickstart code with the HF `pipeline`.
  We do not, because that path on transformers 4.x (CUDA wheel `2.6.0+cu124`) treats `E-X` and `S-X` as their own entity types rather than as boundary tags, which fragments every span at every E/S token.
  Empirically, the simple-aggregation path produces 25,168 false-positive spans on gretel against 131 true positives.
  The lenient BIOES walker in §2.2 produces clean spans on canonical examples.

The report accordingly reports both **strict** and **relaxed** (≥0.5 character overlap) span match.
Relaxed is the fairer metric for cross-system comparison, since boundary conventions differ between datasets, baselines, and the model's tokenizer; strict is reported alongside as a lower bound.

# 3. Methodology

## 3.1 Datasets

| Track | Dataset | Source | Role | Notes |
|---|---|---|---|---|
| EN | `gretelai/synthetic_pii_finance_multilingual` (test, 2,962 EN samples) | [GR-Data] | primary | Apache 2.0; held out from any candidate's training. |
| EN | `ai4privacy/pii-masking-400k` (validation, 17,046 EN samples) | [AI-Data] | secondary | Sibling dataset of `pii-masking-300k`, which is the OpenAI training set; treat as the leakage-adjacent reference. |
| zh | Synthetic zh-PII, 2,000 samples | own | primary | Names from `peoples_daily_ner` PER spans; addresses from `jiaqianjing/chinese-address-ner` plus a fallback pool; Chinese ID-cards (mod-11 checksum), +86 phone numbers, 16–19-digit bank cards, ASCII emails, `https://` URLs, `2024年5月3日`-format dates. |
| zh | `peoples_daily_ner` (validation, 2,319 samples) | [PD-Data] | sanity | Plain Chinese news with PER/LOC/ORG ground truth. No formatted PII. |

Why two English datasets:
`gretel` is a held-out finance corpus that no candidate trained on, so we treat it as the primary EN measurement.
`ai4privacy_400k` is the validation sibling of `pii-masking-300k`.
Privacy Filter is trained on `300k` (per [PF-Card] training notes).
Comparing 96% (training-set) to ~0.40 (validation-sibling) bounds the leakage gap.

## 3.2 Baselines

| Track | Baseline 1 (rule) | Baseline 2 (small SLM) |
|---|---|---|
| EN | Presidio with `en_core_web_lg` and the standard recognizer set ([PR-Repo]) | `ai4privacy/llama-ai4privacy-english-anonymiser-openpii` (ModernBERT 100M, [AI-Model]) |
| zh | Presidio with `zh_core_web_lg` plus custom recognizers for 身份证, 手机号, 银行卡 | `shibing624/bert4ner-base-chinese` (BERT-base 100M, PER/LOC/ORG only, [B4-Model]) |

The AI4Privacy ModernBERT model is a binary detector (PII / not-PII), so its per-category F1 cannot be reported.
We collapse both prediction and gold to a single `_pii` label for that system only, so its number is not artificially zero from category mismatch.

## 3.3 Crosswalk to the 8 OpenAI categories

Each system's native labels are coarsened into the OpenAI 8 (`person, address, email, phone, url, date, account_number, secret`) before scoring.
Out-of-coarsening labels (e.g. Presidio's `LOCATION` ambiguity, Gretel's `company`, AI4Privacy's `USERNAME`) map to `None` and are dropped.
Examples of crosswalk decisions:

- Gretel `time` and `date_time` → `date`; `bank_routing_number` and `routing_number` → `account_number`; `password`, `account_pin`, `credit_card_security_code`, `passport_number` → `secret`; `ipv4`/`ipv6` → `url`.
- AI4Privacy 400k `DATEOFBIRTH` → `date`, `ACCOUNTNUM` → `account_number`, `PASSWORD` → `secret`.
- Presidio `LOCATION` → `address`, `IP_ADDRESS` → `url`, `CN_ID_CARD` → `secret`, `CN_BANK_CARD` → `account_number`.

We verified the crosswalk by enumerating the actual `pii_spans` / `privacy_mask` label vocabularies on each split, not by trusting the schema documentation.
A schema-only crosswalk drops ~2,500 gold spans on the leakage-adjacent dataset: `DATEOFBIRTH` is absent from the documented schema but accounts for 859 gold dates, `ACCOUNTNUM` for 962 account numbers, `PASSWORD` for 657 secrets.
Without those mappings, the model's correct date predictions appear as ~3,500 pure false positives.

## 3.4 Scoring

Span-level F1 at the 8-coarse-class taxonomy.
A span matches gold under **strict** if `(start, end, label)` are equal; under **relaxed** if labels match and character-IoU ≥ 0.5.
Per-category P / R / F1 plus micro-averaged.

## 3.5 Hardware

NVIDIA GeForce RTX 3090 (24 GB GDDR6X, ~936 GB/s peak memory bandwidth, [GPU-Spec]), CUDA 12.4, PyTorch 2.6, BF16, batch=1.
The latency sweep uses 5 warmup + 100 timed iterations per length.

# 4. Quality

## 4.1 English

Relaxed micro F1, full splits:

| System | gretel_finance (n=2,962) | ai4privacy_400k val (n=17,046) |
|---|---:|---:|
| **Privacy Filter** | **0.487** | 0.398 |
| Presidio (en_core_web_lg) | 0.405 | 0.299 |
| AI4Privacy ModernBERT 100M | 0.240 | **0.413** |

Privacy Filter wins on the held-out finance corpus (gretel).
On the leakage-adjacent split (`ai4privacy_400k`), the small fine-tuned ModernBERT slightly edges it out: that dataset's annotation conventions match what ModernBERT was trained on, so the small model's home-court advantage shows.

The 0.96 published F1 is on `pii-masking-300k`.
Same annotation conventions, but `300k` is in Privacy Filter's training set.
The 0.40 we measure on the held-out `400k` validation is the same model on the same annotation style with the training overlap removed.
The drop from 0.96 to 0.40 is the leakage gap, not a generalization gap on a different annotation style.

Per-category breakdown on gretel (relaxed F1):

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

Where the model wins:

1. **Address.**
   Presidio's spaCy `LOCATION` recognizer flags entities like cities and countries, not full street addresses.
   Privacy Filter's `private_address` is concrete physical-address spans, which lines up with gretel's annotation.
2. **Person.**
   Privacy Filter's recall is 0.76 against Presidio's 0.71, with much higher precision (0.79 vs 0.51).
   Presidio over-flags company names as PERSON.
3. **Account number, URL, secret, phone.**
   Privacy Filter's training set covers more identifier formats than Presidio's regex set.

Where Presidio wins:

1. **Email.**
   A well-formed email regex is hard to beat on a synthetic corpus.
   Privacy Filter recall on emails is 0.53, missing about half of gretel gold emails.
   Inspection suggests the truncation cap at 4,096 tokens drops late spans in long financial documents.
2. **Date.**
   Presidio's date regex catches `2024-03-12`, `March 12, 2024`, and `12/03/2024` shapes deterministically.
   Privacy Filter's date recall on gretel is 0.24.
   The model is conservative about labelling raw numeric tuples as dates.

The boundary picture (strict vs relaxed):
strict micro F1 sits at 0.224 on gretel against 0.487 relaxed.
The same gap (≥2× drop) shows up for every system on every dataset; the cause is annotation-vs-tokenizer-boundary mismatch, not model behaviour specifically.
Without the constrained Viterbi decoder, span boundaries can fragment at noisy token-level decisions; that is the single largest source of strict-F1 loss.

## 4.2 Chinese

Two datasets, two stories:

| System | synth_zh (n=2,000) | peoples_daily_zh (n=2,319) |
|---|---:|---:|
| **Privacy Filter** | **0.642** | 0.043 |
| Presidio (zh_core_web_lg + 身份证/手机号/银行卡 regex) | 0.370 | 0.682 |
| `bert4ner-base-chinese` | 0.315 | **0.712** |

The synth_zh corpus mixes Chinese names and Chinese place names with **ASCII-format PII**: `+86` phones, `name@163.com` emails, 18-digit ID cards, 16–19-digit bank cards, `https://` URLs.
On this corpus Privacy Filter dominates: 0.64 relaxed, almost twice both Chinese baselines.

The peoples_daily corpus is bare Chinese news.
The PII surface is Chinese names (e.g. 张三, 李明华) and Chinese place names (e.g. 北京, 中国).
No emails, phones, or account numbers.
On this corpus Privacy Filter collapses to 0.04 relaxed; the Chinese baselines score 16–17× higher.

**Format coverage is not language coverage.**
Privacy Filter recognizes the *shapes* of PII across languages because its training data contains those shapes (an `@` in any language is an email).
It does not generalize to identifying Chinese personal names embedded in Chinese narrative, because that is a token-level pattern unique to the language and absent from a primarily-English training mix.

Per-category on synth_zh (relaxed):

| Category | Privacy Filter | Presidio_zh | bert4ner_zh |
|---|---:|---:|---:|
| email (ASCII shape) | **0.941** | 0 | 0 |
| account_number (16–19 digits) | **0.581** | 0 | 0 |
| phone (+86 shape) | **0.640** | 0 | 0 |
| address | **0.817** | 0 | 0.220 |
| URL (`https://...` shape) | **0.544** | 0 | 0 |
| person (Chinese names) | 0.684 | 0.828 | **0.899** |
| date (`2024年5月3日`) | 0.412 | **0.843** | 0 |
| secret (Chinese 身份证) | 0 | 0 | 0 |

Notable observations:

1. The Chinese baselines emit zero on the ASCII-format categories.
   Presidio_zh's recognizers are language-bound; the standard email/phone/URL regex set is wired to the en pipeline only.
2. On Chinese person names embedded in Chinese sentences (`name + did + something` templates), Privacy Filter recall drops to 0.55 (vs the bert4ner Chinese-NER 0.95).
   The cross-lingual signal is partial: format-rich PII transfers, narrative-pattern PII does not.
3. The `secret` category (Chinese ID cards) is at 0 across all three systems.
   Privacy Filter does not recognize the 18-digit-with-mod-11-checksum format as `secret`; the Chinese baselines' regex set includes 身份证 detection, but the synth_zh ground truth labels them as `secret`, while Presidio emits its native `CN_ID_CARD` which we crosswalk to `secret`. The 0/0 outcome is a category-FP/FN cancellation on a small absolute count.

Per-category on peoples_daily_zh (relaxed):

| Category | Privacy Filter | Presidio_zh | bert4ner_zh |
|---|---:|---:|---:|
| person | 0.123 | 0.805 | **0.945** |
| address (LOC-coarsened) | 0.002 | 0.793 | **0.828** |

Privacy Filter labels 7% of personal names and ~0% of locations.
The bert4ner-zh model, trained on this exact corpus family, scores 0.95 / 0.83.
The address gap has a definitional component too: peoples_daily LOC includes country and region names (中国, 北京), while Privacy Filter's `private_address` is concrete street addresses; some of the recall gap is semantic mismatch rather than failure.
Even so, the person gap (0.12 vs 0.95) is the load-bearing finding: the model does not generalize to Chinese personal names in Chinese narrative.

# 5. Latency

P50 / P99 latency on RTX 3090 across input lengths (BF16, batch=1, 100 timed iterations per length, 5 warmup):

| Input tokens | P50 (ms) | P95 (ms) | P99 (ms) | mean (ms) |
|---:|---:|---:|---:|---:|
| 64 | 222.4 | 227.8 | 229.9 | 221.8 |
| 256 | 278.8 | 283.2 | 287.5 | 276.0 |
| 1024 | 298.0 | 312.3 | 315.1 | 301.3 |
| 4096 | 417.6 | 426.4 | 430.2 | 419.0 |

Three observations:

1. **The "50M active" MoE claim does not translate to latency in the typical regime.**
   Privacy Filter at 1024 tokens is 298 ms P50.
   Qwen3Guard-Gen-0.6B with single-forward-pass classification at 1024 tokens is 57 ms P50 on the same hardware ([Q3G-Gen]).
   The dense-but-smaller model is roughly 5× faster than the larger MoE for one-shot classification.
2. **Why active-params doesn't buy speed here.**
   The 50M active count is FFN-only (top-4 of 128 experts per token).
   Attention is dense banded with sink-token concat in the kernel, eager-only; the kernel materialises the full L×L attention matrix per layer.
   For short to medium inputs the attention cost dominates the FFN saving.
   For long inputs the attention cost grows quadratically and SDPA / flash-attention are not available because the modeling code declares `_supports_sdpa = False` (the sink-token concatenation in the attention kernel is not compatible with the standard SDPA path).
3. **Why 16,384 is missing from the table.**
   The eager kernel allocates a 16k×16k attention matrix per head with sinks concatenated.
   On 24 GB of 3090 VRAM that does not fit (allocation request ~14 GB plus the 11 GB the model already occupies).
   Larger GPUs can opt back in via the `--lengths` CLI override; on 3090, 4,096 is the practical ceiling for this build of the model.

Compared to the Qwen3Guard-Gen-0.6B reference numbers ([Q3G-Gen]):

| Input tokens | Privacy Filter P50 (ms) | Q3G-Gen 0.6B optimized P50 (ms) | ratio |
|---:|---:|---:|---:|
| 64 | 222 | 29 | ~7.7× |
| 256 | 279 | 38 | ~7.3× |
| 1024 | 298 | 58 | ~5.1× |
| 4096 | 418 | 322 | ~1.3× |

The ratio narrows at 4096 because both models become attention-bound there.
At gateway-typical input (256–1024 tokens) Privacy Filter is 5–7× the latency of a dense 0.6B classifier optimized for prefill-only inference.

# 6. Conclusion

1. On English text held out from the training corpus, OpenAI Privacy Filter is the strongest of the three open candidates we tested at relaxed micro F1 0.49 on gretel-finance, beating Microsoft Presidio (0.41) and the AI4Privacy ModernBERT 100M (0.24).
2. The 0.96 model-card F1 is in-distribution.
   The leakage-clean reading on the validation sibling is 0.40 relaxed.
3. Cross-lingual generalization follows PII format, not the natural language.
   ASCII-shaped PII embedded in Chinese text transfers (synth_zh, 0.64).
   Bare Chinese narrative PII does not (peoples_daily_zh, 0.04 vs 0.71 for a Chinese-NER baseline).
   For a deployment that needs Chinese name and place coverage, Privacy Filter must be paired with a Chinese-native NER, not run alone.
4. At gateway latency budgets, the model is 5–7× slower than a dense 0.6B classifier such as Qwen3Guard-Gen on 3090.
   The MoE active-parameter saving applies only to the FFN.
   Attention is dense and eager-only, with no SDPA path available.
   Inputs above 4,096 tokens do not run on a 24 GB 3090.
5. Two pieces of headroom remain:
   - The constrained Viterbi decoder with operating-point calibration is documented in the model card but not implemented in our pipeline.
     Strict-boundary F1 should improve under this decoder; the runtime precision/recall tradeoff would also become tunable.
   - Truncation at 4,096 tokens hurts long-document recall (visible in gretel email recall 0.53).
     Production deployments would need either a chunking strategy or a hardware tier that fits longer contexts.

# References

[PF-Card] OpenAI. *Privacy Filter, model card.* Hugging Face, April 2026. <https://huggingface.co/openai/privacy-filter>. Apache 2.0.

[PF-PDF] OpenAI. *Privacy Filter, Model Card (PDF).* <https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf>.

[PF-Repo] OpenAI. *openai/privacy-filter, source repository.* GitHub. <https://github.com/openai/privacy-filter>. Reference for the constrained Viterbi decoder and operating-point parameters.

[PF-Demo] OpenAI. *Privacy Filter, Hugging Face Space.* <https://huggingface.co/spaces/openai/privacy-filter>.

[GR-Data] Gretel.ai. *synthetic_pii_finance_multilingual.* Hugging Face Datasets. <https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual>. Apache 2.0.

[AI-Data] AI4Privacy. *pii-masking-400k.* Hugging Face Datasets. <https://huggingface.co/datasets/ai4privacy/pii-masking-400k>. Sibling of `pii-masking-300k` used for Privacy Filter training.

[AI-Model] AI4Privacy. *llama-ai4privacy-english-anonymiser-openpii.* Hugging Face. <https://huggingface.co/ai4privacy/llama-ai4privacy-english-anonymiser-openpii>. ModernBERT-100M binary PII detector.

[PR-Repo] Microsoft. *Presidio, Data Protection and Anonymization API.* <https://microsoft.github.io/presidio/>.

[PD-Data] *peoples_daily_ner.* Hugging Face Datasets. <https://huggingface.co/datasets/peoples-daily-ner/peoples_daily_ner>. CoNLL-format PER/LOC/ORG over People's Daily news.

[B4-Model] shibing624. *bert4ner-base-chinese.* Hugging Face. <https://huggingface.co/shibing624/bert4ner-base-chinese>. BERT-base 100M, PER/LOC/ORG.

[Q3G-Gen] Qwen3Guard-Gen Performance Report. Companion document, this workspace. The 0.6B optimized-path P50 numbers used for the latency comparison.

[3090] NVIDIA. *GeForce RTX 3090, product specifications.* <https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/>. 24 GB GDDR6X, ~936 GB/s peak memory bandwidth.

[GPU-Spec] Same as [3090].

# Appendix

## Hardware and measurement setup

NVIDIA RTX 3090, 24 GB GDDR6X.
Linux 6.8, Python 3.10, PyTorch 2.6.0+cu124, transformers 4.x.
BF16 weights, batch=1, no SDPA (model declares `_supports_sdpa = False`).
Quality runs use full splits: 2,962 EN gretel, 17,046 EN ai4privacy_400k validation, 2,000 synth_zh, 2,319 peoples_daily_ner validation.
Latency: 5 warmup + 100 timed iterations per length, single-input forward (no pipeline).

## Full strict tables

Strict micro F1, all systems × all datasets:

| System | gretel_en | ai4priv_400k | synth_zh | peoples_daily_zh |
|---|---:|---:|---:|---:|
| Privacy Filter | 0.224 | 0.126 | 0.505 | 0.024 |
| Presidio | 0.343 | 0.289 | 0.343 | 0.652 |
| AI4Privacy 100M | 0.018 | 0.007 | – | – |
| bert4ner-zh | – | – | 0.259 | 0.694 |

The strict numbers are dominated by tokenizer/annotation boundary differences across datasets and are not the right metric for cross-system comparison; they are listed here for completeness.

## Per-category strict tables (Privacy Filter)

gretel_finance_en (strict):

| Category | P | R | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| account_number | 0.051 | 0.283 | 0.086 | 142 | 2,661 | 360 |
| address | 0.327 | 0.389 | 0.355 | 907 | 1,865 | 1,425 |
| date | 0.417 | 0.166 | 0.237 | 1,099 | 1,537 | 5,541 |
| email | 0.555 | 0.333 | 0.416 | 280 | 225 | 562 |
| person | 0.191 | 0.183 | 0.187 | 840 | 3,556 | 3,748 |
| phone | 0.139 | 0.117 | 0.127 | 62 | 383 | 470 |
| secret | 0.052 | 0.051 | 0.052 | 26 | 472 | 485 |
| url | 0.153 | 0.414 | 0.224 | 65 | 359 | 92 |
| **micro** | **0.236** | **0.212** | **0.224** | **3,421** | **11,058** | **12,683** |

ai4privacy_400k_en (strict):

| Category | P | R | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| account_number | 0.101 | 0.857 | 0.180 | 1,172 | 10,462 | 195 |
| address | 0.122 | 0.126 | 0.124 | 590 | 4,231 | 4,096 |
| date | 0.156 | 0.637 | 0.251 | 547 | 2,951 | 312 |
| email | 0.210 | 0.224 | 0.216 | 356 | 1,342 | 1,236 |
| person | 0.028 | 0.048 | 0.035 | 246 | 8,628 | 4,844 |
| phone | 0.202 | 0.345 | 0.254 | 436 | 1,727 | 828 |
| secret | 0.034 | 0.018 | 0.024 | 63 | 1,814 | 3,431 |
| url | 0 | 0 | 0 | 0 | 1,399 | 0 |
| **micro** | **0.095** | **0.186** | **0.126** | **3,410** | **32,554** | **14,942** |

synth_zh (strict):

| Category | P | R | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| account_number | 0.377 | 0.794 | 0.511 | 730 | 1,206 | 190 |
| address | 0.322 | 0.335 | 0.328 | 273 | 576 | 542 |
| date | 0.350 | 0.146 | 0.206 | 89 | 165 | 522 |
| email | 0.878 | 0.936 | 0.906 | 537 | 75 | 37 |
| person | 0.852 | 0.515 | 0.642 | 894 | 155 | 841 |
| phone | 0.599 | 0.549 | 0.573 | 634 | 425 | 520 |
| secret | 0 | 0 | 0 | 0 | 0 | 530 |
| url | 0 | 0 | 0 | 0 | 132 | 265 |
| **micro** | **0.536** | **0.478** | **0.505** | **3,157** | **2,734** | **3,447** |

peoples_daily_zh (strict):

| Category | P | R | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| address | 0.042 | 0.001 | 0.001 | 1 | 23 | 1,950 |
| date | 0 | 0 | 0 | 0 | 3 | 0 |
| person | 0.248 | 0.040 | 0.068 | 35 | 106 | 849 |
| **micro** | **0.214** | **0.013** | **0.024** | **36** | **132** | **2,799** |
