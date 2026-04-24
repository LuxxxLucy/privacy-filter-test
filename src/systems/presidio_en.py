"""Presidio adapter — English."""
from __future__ import annotations

import torch
from presidio_analyzer import AnalyzerEngine

from ..crosswalk import PRESIDIO_MAP, coarsen


class PresidioEN:
    name = "presidio_en"

    def __init__(self, device: torch.device):
        # Presidio + spaCy is CPU-bound; device is recorded for reporting only.
        self.device = device
        self.analyzer = AnalyzerEngine()

    def predict(self, text: str):
        results = self.analyzer.analyze(text=text, language="en")
        spans = []
        for r in results:
            lab = coarsen(r.entity_type, PRESIDIO_MAP)
            if lab is None:
                continue
            spans.append((int(r.start), int(r.end), lab))
        return spans


def load(device: torch.device) -> PresidioEN:
    return PresidioEN(device)
