"""Dataset loaders.

Each loader exposes:
    iter_samples(test_subset=False) -> Iterator[(text:str, gold:list[(s,e,coarse)])]

Gold spans are already coarsened to the OpenAI 8-class taxonomy. Spans whose
label coarsens to None are dropped from the gold set.
"""
