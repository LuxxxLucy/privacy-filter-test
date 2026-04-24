"""OpenAI privacy-filter adapter."""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..crosswalk import OPENAI_MAP, coarsen

MODEL_ID = "openai/privacy-filter"


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
            lab = coarsen(r["entity_group"], OPENAI_MAP)
            if lab is None:
                continue
            spans.append((int(r["start"]), int(r["end"]), lab))
        return spans


def load(device: torch.device) -> OpenAIPF:
    return OpenAIPF(device)
