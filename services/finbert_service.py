from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

# FinBERT via transformers
# pip install transformers torch sentencepiece
@lru_cache(maxsize=1)
def _get_finbert_pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)

def finbert_score(text: str) -> float:
    """
    Return float score in [-1, 1].
    positive -> +score, negative -> -score, neutral -> 0.
    """
    text = (text or "").strip()
    if not text:
        return 0.0

    pipe = _get_finbert_pipe()

    # keep text length reasonable
    # pipeline already truncates; we also guard
    text = text[:4000]

    out = pipe(text)
    if not out:
        return 0.0

    label = out[0].get("label", "").lower()
    score = float(out[0].get("score", 0.0))

    if "positive" in label:
        return +score
    if "negative" in label:
        return -score
    return 0.0

def safe_mean(scores) -> float:
    arr = np.array(list(scores), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))