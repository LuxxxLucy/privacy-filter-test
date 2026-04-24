"""gretelai/synthetic_pii_finance_multilingual loader (English test split)."""
from __future__ import annotations

from datasets import load_dataset

from ..crosswalk import GRETEL_MAP, coarsen

DATASET_ID = "gretelai/synthetic_pii_finance_multilingual"


def iter_samples(test_subset: bool = False, language: str = "English"):
    ds = load_dataset(DATASET_ID, split="test")
    ds = ds.filter(lambda r: r.get("language") == language)
    if test_subset:
        n = min(50, len(ds))
        ds = ds.select(range(n))
    for r in ds:
        text = r["generated_text"]
        gold = []
        # `pii_spans` may be a JSON string or a list of dicts depending on version.
        spans = r.get("pii_spans") or r.get("entities") or []
        if isinstance(spans, str):
            import json
            spans = json.loads(spans)
        for s in spans:
            lab = coarsen(s.get("label") or s.get("type") or "", GRETEL_MAP)
            if lab is None:
                continue
            gold.append((int(s["start"]), int(s["end"]), lab))
        yield text, gold
