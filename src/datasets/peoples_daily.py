"""peoples_daily_ner — Chinese PER/LOC/ORG sanity set."""
from __future__ import annotations

from datasets import load_dataset

from ..crosswalk import PEOPLES_DAILY_MAP, coarsen

DATASET_ID = "peoples-daily-ner/peoples_daily_ner"
TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def _bio_to_spans(tokens, tag_ids):
    """Reconstruct char-spans by joining tokens (zh tokens are single chars)."""
    text = "".join(tokens)
    char_offsets = []
    cur = 0
    for t in tokens:
        char_offsets.append((cur, cur + len(t)))
        cur += len(t)
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = TAG_NAMES[tag_ids[i]]
        if tag.startswith("B-"):
            ent = tag[2:]
            j = i + 1
            while j < n and TAG_NAMES[tag_ids[j]] == f"I-{ent}":
                j += 1
            spans.append((char_offsets[i][0], char_offsets[j - 1][1], ent))
            i = j
        else:
            i += 1
    return text, spans


def iter_samples(test_subset: bool = False):
    ds = load_dataset(DATASET_ID, split="validation", revision="refs/convert/parquet")
    if test_subset:
        n = min(50, len(ds))
        ds = ds.select(range(n))
    for r in ds:
        text, raw = _bio_to_spans(r["tokens"], r["ner_tags"])
        gold = []
        for s, e, ent in raw:
            lab = coarsen(ent, PEOPLES_DAILY_MAP)
            if lab is None:
                continue
            gold.append((s, e, lab))
        yield text, gold
