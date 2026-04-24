"""Span-level F1, strict and relaxed.

Span schema: tuple (start_char, end_char, coarse_label). Both are half-open.
Strict: exact (start, end, label) match. Relaxed: same label and overlap ≥ 0.5
of the union.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


Span = tuple[int, int, str]


def _overlap_iou(a: Span, b: Span) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(p, r, f, tp, fp, fn)


def score(
    pred: list[list[Span]],
    gold: list[list[Span]],
    relaxed_threshold: float = 0.5,
) -> dict:
    """Returns micro + per-category PRF for both strict and relaxed."""
    cats = set()
    for spans in pred + gold:
        for _, _, lab in spans:
            cats.add(lab)

    tp_s = defaultdict(int)
    fp_s = defaultdict(int)
    fn_s = defaultdict(int)
    tp_r = defaultdict(int)
    fp_r = defaultdict(int)
    fn_r = defaultdict(int)

    for ps, gs in zip(pred, gold):
        gold_set = set(gs)
        # Strict.
        matched_g_strict: set[Span] = set()
        for sp in ps:
            if sp in gold_set:
                tp_s[sp[2]] += 1
                matched_g_strict.add(sp)
            else:
                fp_s[sp[2]] += 1
        for g in gs:
            if g not in matched_g_strict:
                fn_s[g[2]] += 1

        # Relaxed: greedy match by overlap iou ≥ threshold within same label.
        gs_by_lab = defaultdict(list)
        for g in gs:
            gs_by_lab[g[2]].append(g)
        used = set()
        for sp in ps:
            best = -1
            best_iou = relaxed_threshold
            for i, g in enumerate(gs_by_lab[sp[2]]):
                if (sp[2], i) in used:
                    continue
                iou = _overlap_iou(sp, g)
                if iou >= best_iou:
                    best_iou = iou
                    best = i
            if best >= 0:
                tp_r[sp[2]] += 1
                used.add((sp[2], best))
            else:
                fp_r[sp[2]] += 1
        for lab, gs_list in gs_by_lab.items():
            for i, _ in enumerate(gs_list):
                if (lab, i) not in used:
                    fn_r[lab] += 1

    out = {"strict": {}, "relaxed": {}}
    for lab in sorted(cats):
        out["strict"][lab] = _prf(tp_s[lab], fp_s[lab], fn_s[lab]).__dict__
        out["relaxed"][lab] = _prf(tp_r[lab], fp_r[lab], fn_r[lab]).__dict__
    out["strict"]["micro"] = _prf(
        sum(tp_s.values()), sum(fp_s.values()), sum(fn_s.values())
    ).__dict__
    out["relaxed"]["micro"] = _prf(
        sum(tp_r.values()), sum(fp_r.values()), sum(fn_r.values())
    ).__dict__
    return out
