"""Presidio adapter — English. Loads spaCy model from local snapshot path."""
from __future__ import annotations

from pathlib import Path

import torch
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from ..crosswalk import PRESIDIO_MAP, coarsen

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACY_EN_PATH = REPO_ROOT / ".cache" / "spacy" / "en_core_web_lg"


class PresidioEN:
    name = "presidio_en"

    def __init__(self, device: torch.device):
        self.device = device
        cfg = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": str(SPACY_EN_PATH)}],
        }
        provider = NlpEngineProvider(nlp_configuration=cfg)
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry(supported_languages=["en"])
        registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
        registry.add_recognizer(SpacyRecognizer(supported_language="en"))
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["en"]
        )

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
