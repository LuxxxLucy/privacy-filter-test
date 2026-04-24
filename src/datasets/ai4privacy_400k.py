"""ai4privacy/pii-masking-400k validation split."""
from __future__ import annotations

from datasets import load_dataset

from ..crosswalk import AI4PRIVACY_400K_MAP, coarsen

DATASET_ID = "ai4privacy/pii-masking-400k"


def iter_samples(test_subset: bool = False, language: str = "en"):
    ds = load_dataset(DATASET_ID, split="validation")
    ds = ds.filter(lambda r: r.get("language") == language)
    if test_subset:
        n = min(50, len(ds))
        ds = ds.select(range(n))
    for r in ds:
        text = r.get("source_text") or r.get("text") or r.get("unmasked_text")
        gold = []
        for s in r.get("privacy_mask", []) or []:
            lab = coarsen(s.get("label", ""), AI4PRIVACY_400K_MAP)
            if lab is None:
                continue
            gold.append((int(s["start"]), int(s["end"]), lab))
        yield text, gold
