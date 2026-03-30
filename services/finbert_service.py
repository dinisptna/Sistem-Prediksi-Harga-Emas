from __future__ import annotations

import streamlit as st
import numpy as np

@st.cache_resource
def get_finbert_pipe():
    try:
        from transformers import pipeline
        # Gunakan model yang lebih kecil atau pastikan device adalah cpu
        pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1 # Paksa pakai CPU
        )
        return pipe
    except Exception as e:
        st.error(f"Gagal memuat FinBERT: {e}")
        return None


def finbert_score(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0

    pipe = get_finbert_pipe()

    # kalau model gagal load jangan crash
    if pipe is None:
        return 0.0

    try:
        text = text[:1000]  # lebih aman di cloud

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

    except Exception as e:
        print("❌ Error FinBERT inference:", e)
        return 0.0


def safe_mean(scores) -> float:
    arr = np.array(list(scores), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))