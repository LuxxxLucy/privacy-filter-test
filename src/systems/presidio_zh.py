"""Presidio adapter — Chinese.

Default Presidio has no zh recognizer pack. We register a zh NLP engine
plus three custom regex recognizers: 手机号 / 身份证号 / 银行卡号, and a
manually-registered SpacyRecognizer so spaCy zh NER (PERSON / GPE / DATE /
LOC / ORG) flows through.
"""
from __future__ import annotations

import torch
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from ..crosswalk import PRESIDIO_MAP, coarsen

CN_MOBILE = PatternRecognizer(
    supported_entity="CN_MOBILE",
    supported_language="zh",
    patterns=[Pattern(name="cn_mobile", regex=r"\b1[3-9]\d{9}\b", score=0.85)],
)

CN_ID_CARD = PatternRecognizer(
    supported_entity="CN_ID_CARD",
    supported_language="zh",
    patterns=[
        Pattern(
            name="cn_id_18",
            regex=r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
            score=0.9,
        )
    ],
)

CN_BANK_CARD = PatternRecognizer(
    supported_entity="CN_BANK_CARD",
    supported_language="zh",
    patterns=[Pattern(name="cn_bank", regex=r"\b\d{16,19}\b", score=0.4)],
)


class PresidioZH:
    name = "presidio_zh"

    def __init__(self, device: torch.device):
        self.device = device
        cfg = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "zh", "model_name": "zh_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=cfg)
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry(supported_languages=["zh"])
        registry.add_recognizer(SpacyRecognizer(supported_language="zh"))
        for r in (CN_MOBILE, CN_ID_CARD, CN_BANK_CARD):
            registry.add_recognizer(r)
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["zh"]
        )

    def predict(self, text: str):
        results = self.analyzer.analyze(text=text, language="zh")
        spans = []
        for r in results:
            lab = coarsen(r.entity_type, PRESIDIO_MAP)
            if lab is None:
                continue
            spans.append((int(r.start), int(r.end), lab))
        return spans


def load(device: torch.device) -> PresidioZH:
    return PresidioZH(device)
