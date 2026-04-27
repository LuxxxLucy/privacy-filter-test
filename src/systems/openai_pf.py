"""OpenAI privacy-filter adapter.

The model has a 33-way BIOES head (B-/I-/E-/S- × 8 entities + O). HF's
`pipeline(..., aggregation_strategy="simple")` only knows BIO; fed BIOES it
fragments every span at E-/S- tokens, producing massive over-prediction. We
do the BIOES decode ourselves from offset-mapped subwords.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..crosswalk import OPENAI_MAP, coarsen

MODEL_ID = "openai/privacy-filter"

# Eager attention is O(L^2) in memory and the model does not support SDPA
# (_supports_sdpa = False — sink-token concat in the attention kernel).
# 4096 fits comfortably on a 24 GB GPU; longer inputs are truncated.
MAX_TOKENS = 4096


def _bioes_spans(label_ids, id2label, offsets):
    """Decode BIOES tag sequence -> [(char_start, char_end, coarse_label)].

    Lenient: opens on B-X or S-X, extends on I-X (same X), closes on E-X
    (same X), O, label change, or end. Out-of-band transitions don't crash —
    they just close the current span.
    """
    spans = []
    cur_type = None
    cur_start = None
    cur_end = None

    def _flush():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None and cur_start is not None and cur_end > cur_start:
            coarse = coarsen(cur_type, OPENAI_MAP)
            if coarse is not None:
                spans.append((cur_start, cur_end, coarse))
        cur_type = None
        cur_start = None
        cur_end = None

    for tok_idx, lab_id in enumerate(label_ids):
        s, e = offsets[tok_idx]
        if s == 0 and e == 0:
            # special token (CLS/SEP/PAD) — does not break a span by itself
            continue
        lab = id2label[int(lab_id)]
        if lab == "O":
            _flush()
            continue
        prefix, _, ent = lab.partition("-")
        if prefix not in ("B", "I", "E", "S") or not ent:
            _flush()
            continue
        if prefix == "B":
            _flush()
            cur_type, cur_start, cur_end = ent, s, e
        elif prefix == "S":
            _flush()
            cur_type, cur_start, cur_end = ent, s, e
            _flush()
        elif prefix == "I":
            if cur_type == ent:
                cur_end = e
            else:
                _flush()
                cur_type, cur_start, cur_end = ent, s, e
        elif prefix == "E":
            if cur_type == ent:
                cur_end = e
            else:
                _flush()
                cur_type, cur_start, cur_end = ent, s, e
            _flush()
    _flush()
    return spans


class OpenAIPF:
    name = "openai_privacy_filter"

    def __init__(self, device: torch.device):
        self.device = device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, dtype=dtype
        ).to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str):
        if not text or not text.strip():
            return []
        enc = self.tok(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"]
        # Defensive: tokenizer rarely returns truly empty ids, but the model's
        # eager attention reshape crashes on 0-length sequences.
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return []
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]  # (seq, num_labels)
        label_ids = logits.argmax(dim=-1).tolist()
        return _bioes_spans(label_ids, self.id2label, offsets)


def load(device: torch.device) -> OpenAIPF:
    return OpenAIPF(device)
