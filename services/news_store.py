from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
import feedparser

from services.finbert_service import finbert_score

DEFAULT_TSV = Path("data/berita.tsv")

# =========================
# RSS SOURCES (>= 50)
# =========================
RSS_SOURCES: dict[str, str] = {
    # --- Reuters ---
    "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets": "https://feeds.reuters.com/reuters/marketsNews",
    "Reuters Economy": "https://feeds.reuters.com/reuters/economicNews",
    "Reuters Commodities": "https://feeds.reuters.com/reuters/commoditiesNews",

    # --- CNBC ---
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC World": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",

    # --- Investing.com ---
    "Investing Commodities": "https://www.investing.com/rss/commodities.xml",
    "Investing Economy": "https://www.investing.com/rss/economicIndicators.xml",
    "Investing World News": "https://www.investing.com/rss/news_301.rss",

    # --- MarketWatch ---
    "MarketWatch Top": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "MarketWatch Economy": "https://feeds.marketwatch.com/marketwatch/economy/",

    # --- Yahoo Finance ---
    "Yahoo Finance Top": "https://finance.yahoo.com/rss/topstories",
    "Yahoo Finance Markets": "https://finance.yahoo.com/rss/markets",
    "Yahoo Finance Economy": "https://finance.yahoo.com/rss/economy",

    # --- Financial Times ---
    "Financial Times World": "https://www.ft.com/rss/world",
    "Financial Times Markets": "https://www.ft.com/rss/markets",
    "Financial Times Economy": "https://www.ft.com/rss/global-economy",

    # --- Bloomberg (safe RSS / podcast feeds) ---
    "Bloomberg Markets": "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "Bloomberg Economics": "https://www.bloomberg.com/feed/podcast/bloomberg-economics.xml",

    # --- Economist ---
    "The Economist Finance": "https://www.economist.com/finance-and-economics/rss.xml",

    # --- Global Institutions ---
    "World Bank News": "https://www.worldbank.org/en/news/all/rss",
    "IMF News": "https://www.imf.org/external/rss/feeds.aspx",

    # --- Central Banks ---
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB Press": "https://www.ecb.europa.eu/rss/press.xml",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",

    # --- Geopolitics ---
    "Al Jazeera World": "https://www.aljazeera.com/xml/rss/all.xml",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",

    # --- Commodities & Gold ---
    "OilPrice": "https://oilprice.com/rss/main",
    "Kitco Gold": "https://www.kitco.com/rss/news/gold",
    "Mining.com": "https://www.mining.com/feed/",
}

# =========================
# KEYWORDS (expanded)
# =========================
EMAS_DIRECT = [
    "gold", "bullion", "xau", "xauusd", "precious metal", "safe haven",
    "spot gold", "gold price", "gold prices", "gold futures", "gc=f",
]

# ---------- factor dictionary (expanded; no "Lainnya") ----------
FACTOR_KEYWORDS: dict[str, list[str]] = {
    "Suku Bunga": [
        "rate", "rates", "interest rate", "interest-rate", "hike", "cut", "hold", "pause",
        "tightening", "easing", "hawkish", "dovish", "dot plot",
        "yield", "yields", "bond", "bonds", "treasury", "treasuries", "gilt", "bund", "jgb",
        "yield curve", "inversion", "real yield", "nominal yield", "tips",
        "fed", "fomc", "powell", "federal reserve",
        "ecb", "lagarde", "boe", "boj", "pboc", "rba", "rbi", "snb",
        "policy rate", "benchmark rate", "terminal rate",
    ],
    "Inflasi": [
        "inflation", "disinflation", "deflation",
        "cpi", "core cpi", "pce", "core pce", "ppi",
        "price pressures", "cost pressures", "sticky inflation",
        "consumer prices", "price index", "wage inflation", "wage growth",
        "services inflation", "shelter inflation",
    ],
    "Dolar": [
        "dollar", "us dollar", "dxy", "greenback",
        "forex", "fx", "currency", "currencies", "exchange rate",
        "usd/jpy", "eur/usd", "gbp/usd", "usd/cny", "usd/idr",
        "depreciation", "appreciation", "devaluation",
        "risk-on", "risk-off", "safe haven currency",
        "capital outflows", "capital inflows",
    ],
    "Geopolitik": [
        "war", "conflict", "tensions", "escalation", "ceasefire",
        "israel", "gaza", "middle east", "iran", "syria",
        "russia", "ukraine", "nato",
        "china", "taiwan", "south china sea",
        "missile", "drone", "attack", "strike",
        "sanctions", "embargo", "blockade",
        "military", "troops",
    ],
    "Energi": [
        "oil", "crude", "brent", "wti",
        "opec", "opec+", "output cuts", "production cuts",
        "gas", "natural gas", "lng",
        "refinery", "supply disruption",
        "energy prices", "fuel prices",
    ],
    "Resesi": [
        "recession", "slowdown", "contraction", "downturn",
        "gdp", "economic growth", "growth outlook",
        "soft landing", "hard landing",
        "unemployment", "jobless claims", "layoffs",
        "pmi", "ism", "manufacturing", "services",
        "consumer confidence", "retail sales",
    ],
    "Pasar Saham": [
        "stocks", "equities", "index", "indices",
        "s&p", "nasdaq", "dow",
        "rally", "selloff", "correction",
        "volatility", "vix",
        "risk appetite", "risk sentiment",
        "flight to safety", "market turmoil",
    ],
    "Fiskal": [
        "budget", "deficit", "debt ceiling", "shutdown",
        "stimulus", "spending bill", "fiscal policy",
        "tax", "taxes", "tariff", "tariffs",
        "trade policy", "sanctions policy",
    ],
    "Banking": [
        "bank", "banks", "banking sector",
        "credit", "credit crunch", "liquidity",
        "default", "bankruptcy", "insolvency",
        "loan", "lending", "mortgage",
        "refinancing", "spreads", "cds",
        "stress test", "bailout",
    ],
    "Perdagangan": [
        "trade", "exports", "imports",
        "supply chain", "logistics", "shipping",
        "freight", "port congestion",
        "disruption", "shortages",
        "trade war",
    ],
}



def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _extract_summary(entry) -> str:
    # feedparser entries often have: summary, description, content
    for k in ["summary", "description"]:
        if hasattr(entry, k):
            v = getattr(entry, k)
            if v:
                return str(v)
    if hasattr(entry, "content") and entry.content:
        try:
            v = entry.content[0].value
            if v:
                return str(v)
        except Exception:
            pass
    return ""

def _detect_factor(text: str) -> Optional[str]:
    """
    Return best matching factor label, avoiding 'Lainnya'.
    If nothing matches, fallback to 'Ekonomi' (per request).
    """
    t = _norm(text)
    best = None
    best_hits = 0
    for factor, kws in FACTOR_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_hits = hits
            best = factor
    if best and best_hits > 0:
        return best
    return "Ekonomi"

def _impact_from_news_rule_based(df_rel: pd.DataFrame) -> tuple[str, float, float, float]:
    """
    Return:
    - impact_label: "Bullish" / "Bearish" (tanpa Neutral)
    - prob_up: probabilitas naik (0..1)
    - prob_dn: probabilitas turun (0..1)
    - gold_score_mean: skor agregat (kontinu)
    """

    if df_rel.empty or "sentiment_score" not in df_rel.columns or "faktor" not in df_rel.columns:
        return "Bearish", 0.50, 0.50, 0.0

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

    if d.empty:
        return "Bearish", 0.50, 0.50, 0.0


    # ========= OPSI B: faktor beda arah =========
    # Group 1: kalau berita "positif" di faktor ini, biasanya emas melemah (Bearish)
    INVERT_FACTORS = {"Suku Bunga", "Dolar"}

    # Group 2: faktor risiko (default) -> berita negatif = risk naik = emas sering menguat confirms safe haven
    # (kita pakai default untuk faktor lain)
    def factor_sign(f: str) -> int:
        # +1 artinya sentimen positif -> bullish emas (tidak dipakai di default kita),
        # -1 artinya sentimen positif -> bearish emas (dibalik)
        return -1 if str(f) in INVERT_FACTORS else -1

    # gold_score per berita:
    # - kalau skor FinBERT positif (+) dan factor_sign=-1 => gold_score negatif => bearish
    # - kalau skor FinBERT negatif (-) dan factor_sign=-1 => gold_score positif => bullish
    d["gold_score"] = d.apply(lambda r: float(r["sentiment_score"]) * factor_sign(r["faktor"]), axis=1)

    gold_score_mean = float(d["gold_score"].mean())

    # Label tanpa Neutral:
    impact = "Bullish" if gold_score_mean >= 0 else "Bearish"

    # Probabilitas: pakai mean gold_score (clamp), makin besar |score| makin yakin
    # gold_score_mean secara kasar di [-1,1]
    s = max(-1.0, min(1.0, gold_score_mean))
    prob_up = 0.5 + 0.3 * s     # range ~ [0.2, 0.8]
    prob_up = max(0.01, min(0.99, prob_up))
    prob_dn = 1.0 - prob_up

    return impact, prob_up, prob_dn, gold_score_mean

def ensure_tsv(path: Path = DEFAULT_TSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df = pd.DataFrame(columns=[
            "tanggal", "sumber", "judul", "link", "isi",
            "sentiment_score", "faktor",
        ])
        df.to_csv(path, sep="\t", index=False)
    return path

def load_tsv(path: Path = DEFAULT_TSV) -> pd.DataFrame:
    ensure_tsv(path)
    df = pd.read_csv(path, sep="\t")
    if not df.empty and "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    return df

def save_tsv(df: pd.DataFrame, path: Path = DEFAULT_TSV) -> None:
    ensure_tsv(path)
    df.to_csv(path, sep="\t", index=False)

def fetch_and_append_news(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_total: int = 100000,
    path: Path = DEFAULT_TSV,
) -> pd.DataFrame:
    """
    Fetch RSS news, compute FinBERT score, factor.
    Cache into TSV and append only new links.
    """
    df_old = load_tsv(path)
    existing_links = set(df_old["link"].astype(str).tolist()) if not df_old.empty and "link" in df_old.columns else set()

    items: List[dict] = []
    for source_name, url in RSS_SOURCES.items():
        feed = feedparser.parse(url)
        for entry in getattr(feed, "entries", []):
            title = str(getattr(entry, "title", "") or "")
            link = str(getattr(entry, "link", "") or "")
            if not link:
                continue
            if link in existing_links:
                continue

            # date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = pd.Timestamp(*entry.published_parsed[:6])
                except Exception:
                    published = None
            if published is None:
                published = pd.Timestamp.now()

            # filter by date range
            if published.normalize() < start_date.normalize() or published.normalize() > end_date.normalize():
                continue

            summary = _extract_summary(entry)
            if not summary:
                summary = title  # avoid NaN

            faktor = _detect_factor(f"{title} {summary}")

            # FinBERT score on title+summary (fast, stable)
            score = finbert_score(f"{title}. {summary}")

            items.append({
                "tanggal": str(published.normalize().date()),
                "sumber": source_name,
                "judul": title,
                "link": link,
                "isi": summary,
                "sentiment_score": float(score),
                "faktor": faktor,
            })

            if len(items) >= max_total:
                break
        if len(items) >= max_total:
            break

    if items:
        df_new = pd.DataFrame(items)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # drop duplicates by link
        df_all = df_all.drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)
        save_tsv(df_all, path)
        return df_all

    return df_old