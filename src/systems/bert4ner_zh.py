"""shibing624/bert4ner-base-chinese adapter — coarse PER/LOC/ORG/TIME."""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..crosswalk import BERT4NER_MAP, coarsen

MODEL_ID = "shibing624/bert4ner-base-chinese"


class Bert4NerZH:
    name = "bert4ner_zh"

    def __init__(self, device: torch.device):
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, dtype=dtype
        ).to(device)
        self.model.eval()
        self.pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tok,
            device=device,
            aggregation_strategy="simple",
        )

    def predict(self, text: str):
        out = self.pipe(text)
        spans = []
        for r in out:
            lab = coarsen(r["entity_group"], BERT4NER_MAP)
            if lab is None:
                continue
            spans.append((int(r["start"]), int(r["end"]), lab))
        return spans


def load(device: torch.device) -> Bert4NerZH:
    return Bert4NerZH(device)
