from __future__ import annotations

import streamlit as st
import pandas as pd

from core.style import load_css
from core.state import ensure_state_defaults
from services.market_data import fetch_gold_ohlcv
from services.news_store import (
    _impact_from_news_rule_based,
    ensure_tsv,
    load_tsv,
)

# ---------- helpers ----------
def keep_up_to_last_close_day(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    return df[df.index.normalize() < today].copy()

def format_float(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"

def _kpi_card(title: str, value: str, sub: str, extra_html: str = ""):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(0,0,0,.12); border-radius:16px; padding:16px; background:rgba(255,255,255,.9);">
          <div style="font-size:13px; font-weight:900; opacity:.9;">{title}</div>
          <div style="font-size:34px; font-weight:900; line-height:1; margin-top:10px;">{value}</div>
          <div style="font-size:12px; opacity:.7; margin-top:8px;">{sub}</div>
          {extra_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prob_from_score(mean_score: float) -> tuple[float, float]:
    s = max(-1.0, min(1.0, float(mean_score)))
    prob_up = 0.5 + 0.3 * s   # ~ [0.2, 0.8]
    prob_up = max(0.01, min(0.99, prob_up))
    prob_dn = 1.0 - prob_up
    return prob_up, prob_dn

def berita_page():
    ensure_state_defaults()
    load_css()

    # Header
    st.markdown(
        '<div style="font-size:16px; opacity:.7; margin-top:6px;">Created by Dini Septiana & Ayu Andani</div>',
        unsafe_allow_html=True
    )
    st.image("assets/Berita.jpg", use_container_width=True)

    # fixed start date
    start_news = pd.Timestamp("2026-01-01")

    # end date = last close day from yfinance GC=F (same concept as beranda)
    df_gold = fetch_gold_ohlcv(start="2021-01-19", end=None, interval="1d")
    df_gold = keep_up_to_last_close_day(df_gold)
    if df_gold.empty:
        st.error("Data harga emas (yfinance) kosong. Cek koneksi atau simbol GC=F.")
        return
    end_news = pd.Timestamp(df_gold.index[-1]).normalize()

    ensure_tsv()

    # ========= UI HEADER + Tombol update (tanpa input max_total) =========
    # max_total dikunci di kode (revisi #1)
    MAX_TOTAL = 1000

    st.markdown(
        '<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:10px;">Informasi Berita</div>',
        unsafe_allow_html=True
    )
    st.caption("Pengambilan berita global via RSS disimpan ke data/berita.tsv agar tidak fetch ulang terus.")

    df = load_tsv()

    # filter to window (2026-01-01 .. last close)
    if not df.empty:
        df["tanggal_dt"] = pd.to_datetime(df["tanggal"], errors="coerce")
        df = df[(df["tanggal_dt"] >= start_news) & (df["tanggal_dt"] <= end_news)].copy()
        df = df.drop(columns=["tanggal_dt"], errors="ignore")

    range_label = f"({start_news.date()} s/d {end_news.date()})"

    # KPI Row 1: info umum (total, start, end)
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        _kpi_card("Tanggal Mulai", str(start_news.date()), range_label)
    with k2:
        _kpi_card("Tanggal Terakhir", str(end_news.date()), range_label)
    with k3:
        _kpi_card("Jumlah Berita", str(int(len(df))) if not df.empty else "0", range_label)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:20px; margin-top:10px;">Ringkasan Berita</div>',
        unsafe_allow_html=True
    )

    # KPI Row 2: relevan emas only
    if df.empty:
        st.info("Belum ada berita. Klik 'Run Update' pada Halaman Beranda.")
        return

    # pastikan kolom relevan ada (biar ga KeyError)
    if "relevan" not in df.columns:
        df["relevan"] = False

    df_rel = df[df["relevan"] == True].copy()

    if df_rel.empty or "sentiment_score" not in df_rel.columns or "faktor" not in df_rel.columns:
        mean_sent = 0.0
    else:
        # tentukan faktor dominan
        faktor_dom = (
            df_rel["faktor"]
            .astype(str)
            .value_counts()
            .idxmax()
        )

        # ambil hanya berita faktor dominan
        d = df_rel[df_rel["faktor"] == faktor_dom].copy()

        d["sentiment_score"] = pd.to_numeric(d["sentiment_score"], errors="coerce")
        d = d.dropna(subset=["sentiment_score"])

        mean_sent = float(d["sentiment_score"].mean()) if not d.empty else 0.0

    # dominant factor over relevan; exclude Ekonomi if possible
    faktor_dom = "Ekonomi"
    if not df_rel.empty and "faktor" in df_rel.columns:
        counts = df_rel["faktor"].fillna("Ekonomi").astype(str)
        counts_non = counts[counts != "Ekonomi"]
        if len(counts_non) > 0:
            faktor_dom = counts_non.value_counts().index[0]
        else:
            faktor_dom = counts.value_counts().index[0]

        impact, prob_up, prob_dn, gold_score_mean = _impact_from_news_rule_based(df_rel)


    # ========= KPI Row 2 (revisi #2: hapus "Jumlah Berita Relevan") =========
    cA, cB, cC = st.columns(3, gap="medium")
    with cA:
        _kpi_card("Sentimen Score (Mean)", format_float(mean_sent, 4), range_label)
    with cB:
        _kpi_card("Faktor Dominan", faktor_dom, range_label)
    with cC:
        impact_color = "#dc2626" if impact == "Bearish" else "#16a34a"
        
        # badge probabilitas
        up = prob_up >= prob_dn
        arrow = "▲" if up else "▼"
        prob_color = "#16a34a" if up else "#dc2626"
        prob_bg = "rgba(22, 163, 74, 0.12)" if up else "rgba(220, 38, 38, 0.12)"

        extra = f"""
        <div style="margin-top:10px;">
        <span style="
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 10px; border-radius:999px;
            background:{prob_bg};
            color:{prob_color};
            font-weight:900; font-size:12px;">
            {arrow} Prob Up: {prob_up*100:.4f}% • Prob Down: {prob_dn*100:.4f}%
        </span>
        </div>
        """

        _kpi_card(
            "Dampak Emas",
            impact,
            range_label,
            extra_html=extra
        )


    st.markdown("---")
    st.markdown(
        '<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:10px;">Dataset Berita</div>',
        unsafe_allow_html=True
    )

    # ========= Dataset (revisi #3: LinkColumn, bukan markdown string) =========
    df_show = df.copy()

    # rename kolom
    df_show = df_show.rename(columns={
        "tanggal": "Tanggal",
        "sumber": "Sumber",
        "judul": "Judul",
        "link": "Link",
        "isi": "Isi",
        "sentiment_score": "Sentiment Score",
        "faktor": "Faktor",
    })

    # pastikan kolom ada
    for col in ["Tanggal", "Sumber", "Judul", "Link", "Isi", "Sentiment Score", "Faktor"]:
        if col not in df_show.columns:
            df_show[col] = ""

    # Link must be URL plain string
    df_show["Link"] = df_show["Link"].fillna("").astype(str)

    st.dataframe(
        df_show[["Tanggal", "Sumber", "Judul", "Link", "Isi", "Sentiment Score", "Faktor"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn(
                "Link",
                help="Klik untuk membuka sumber berita",
                display_text="Buka",
            )
        }
    )

    st.caption(
        "Catatan: kolom Link bisa diklik untuk membuka sumber berita. "
        "Sentimen memakai FinBERT (ProsusAI/finbert). Isi diambil dari ringkasan RSS agar stabil & tidak lambat."
    )