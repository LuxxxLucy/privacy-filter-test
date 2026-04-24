"""AI4Privacy English ModernBERT adapter."""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

PII_BINARY = "_pii"  # Model is a binary PII vs O detector — single label.

MODEL_ID = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"


class AI4Privacy:
    name = "ai4privacy_en"

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
        return [(int(r["start"]), int(r["end"]), PII_BINARY) for r in out]


def load(device: torch.device) -> AI4Privacy:
    return AI4Privacy(device)
