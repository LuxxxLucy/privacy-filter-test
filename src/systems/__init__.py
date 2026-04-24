"""System adapters. Each exposes:

    load(device) -> system
    system.predict(text: str) -> list[(start, end, coarse_label)]

Coarse label is one of crosswalk.OPENAI_LABELS or `None` (then dropped).
"""
