import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services.news_store import FACTOR_KEYWORDS, _impact_from_news_rule_based
from core.style import load_css
from core.state import ensure_state_defaults, bump_data_version
from services.market_data import fetch_gold_ohlcv
from services.predictor import (
    compute_feature_frame,
    load_model,
    get_median_returns,
    predict_direction,
    predict_close_H_and_H1,
)
from services.news_store import (
    ensure_tsv,
    load_tsv,
)

# ---------- helpers ----------
def format_id_number(x: float, decimals: int = 3) -> str:
    """Format ala Indonesia: ribuan '.' dan desimal ',' """
    s = f"{x:,.{decimals}f}"          # 4,879.930
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 4.879,930
    return s

def next_business_days(last_date: pd.Timestamp, n: int = 2) -> list[pd.Timestamp]:
    d = pd.Timestamp(last_date)
    out = []
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.dayofweek < 5:  # Mon-Fri
            out.append(d)
    return out

def keep_up_to_last_close_day(df: pd.DataFrame) -> pd.DataFrame:
    """Paksa data aktual hanya sampai kemarin (hari ini tidak diambil)."""
    today = pd.Timestamp.now().normalize()
    return df[df.index.normalize() < today].copy()

def next_n_business_days(last_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    d = pd.Timestamp(last_date)
    out = []
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.dayofweek < 5:
            out.append(d)
    return out

def make_future_path(last_close: float, r: float, dates: list[pd.Timestamp]) -> list[float]:
    vals = []
    cur = last_close
    for _ in dates:
        cur = float(cur * (1 + r))
        vals.append(cur)
    return vals

def _prob_from_score(mean_score: float) -> tuple[float, float]:
    s = max(-1.0, min(1.0, float(mean_score)))
    prob_up = 0.5 + 0.3 * s
    prob_up = max(0.01, min(0.99, prob_up))
    prob_dn = 1.0 - prob_up
    return prob_up, prob_dn

@st.cache_data(show_spinner=False)
def get_news_cached(_data_version: int) -> pd.DataFrame:
    ensure_tsv()
    df = load_tsv()
    return df


@st.cache_data(show_spinner=False)
def get_gold_data_cached(data_version: int) -> pd.DataFrame:
    df = fetch_gold_ohlcv(start="2021-01-19", end=None, interval="1d")
    df = keep_up_to_last_close_day(df)
    return df

@st.cache_data(show_spinner=False)
def get_feature_df_cached(data_version: int) -> pd.DataFrame:
    df = get_gold_data_cached(data_version)
    feat = compute_feature_frame(df)
    return feat

@st.cache_data(show_spinner=False)
def get_prediction_cached(data_version: int) -> dict:
    df = get_gold_data_cached(data_version)
    feat = get_feature_df_cached(data_version)

    model = load_model("models/rf_gold_direction_model.pkl")
    med = get_median_returns(feat)

    latest = feat.iloc[-1]  # basis H-1
    last_close = float(latest["Close"])
    pred_dir = predict_direction(model, latest)

    pred_H, pred_H1, r_used = predict_close_H_and_H1(last_close, pred_dir, med)

    last_date = feat.index[-1]
    pred_dates = next_business_days(last_date, 2)

    return {
        "last_date": last_date,
        "last_open": float(latest["Open"]) if "Open" in latest.index else float(df["Open"].iloc[-1]),
        "last_close": last_close,
        "pred_dir": pred_dir,
        "r_used": r_used,
        "pred_date_H": pred_dates[0],
        "pred_date_H1": pred_dates[1],
        "pred_close_H": pred_H,
        "pred_close_H1": pred_H1,
    }

def make_chart(df: pd.DataFrame, pred: dict, horizon: int) -> go.Figure:
    hist = df.copy()

    last_date = pd.Timestamp(pred["last_date"])
    hist = hist.loc[:last_date].copy()

    # return harian actual (%)
    hist["return_daily"] = hist["Close"].pct_change() * 100
    hist["ret_sign"] = hist["return_daily"].apply(
        lambda x: "▲" if pd.notna(x) and x >= 0 else ("▼" if pd.notna(x) else "")
    )
    hist["ret_color"] = hist["return_daily"].apply(
        lambda x: "green" if pd.notna(x) and x >= 0 else ("red" if pd.notna(x) else "gray")
    )

    last_close = float(pred["last_close"])
    r = float(pred["r_used"])

    # future dates & values
    future_dates = next_n_business_days(last_date, horizon)
    future_vals = make_future_path(last_close, r, future_dates)

    # cone band (melebar seiring horizon) pakai vol10 approx
    daily_return = hist["Close"].pct_change()
    vol10 = float(daily_return.rolling(10).std().iloc[-1])
    if not (vol10 == vol10) or vol10 == 0:
        vol10 = 0.01

    upper, lower = [], []
    for i, v in enumerate(future_vals, start=1):
        widen = 1.0 + 0.06 * i
        upper.append(v * (1 + vol10 * widen))
        lower.append(v * (1 - vol10 * widen))

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name="Actual (yfinance)",
        customdata=list(zip(hist["return_daily"], hist["ret_sign"], hist["ret_color"])),
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            "Close: %{y:,.2f} USD<br>"
            "Return: <b><span style='color:%{customdata[2]}'>%{customdata[1]} %{customdata[0]:+.2f}%</span></b>"
            "<extra></extra>"
        ),
    ))

    # Prediksi
    ret_preds = [(v / last_close - 1) * 100 for v in future_vals]
    ret_signs = ["▲" if rr >= 0 else "▼" for rr in ret_preds]
    ret_colors = ["green" if rr >= 0 else "red" for rr in ret_preds]

    fig.add_trace(go.Scatter(
        x=[last_date] + future_dates,
        y=[last_close] + future_vals,
        mode="lines+markers",
        name="Prediksi (basis H-1)",
        line=dict(dash="dash", color="#fb923c"),
        customdata=[(0.0, "", "gray")] + list(zip(ret_preds, ret_signs, ret_colors)),
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            "Pred Close: %{y:,.2f} USD<br>"
            "Return: <b><span style='color:%{customdata[2]}'>%{customdata[1]} %{customdata[0]:+.2f}%</span></b>"
            "<extra></extra>"
        ),
    ))

    # cone fill (lower then upper fill)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Forecast Range (approx)",
        fillcolor="rgba(251, 146, 60, 0.28)",  # orange muda
        opacity=1.0,
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        yaxis=dict(title="USD", side="right", automargin=True),
        xaxis=dict(title="Date", automargin=True),
        dragmode="zoom",
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0),
    )

    # default view (H-3 s/d beberapa hari setelahnya)
    start_news = pd.Timestamp("2026-01-01")
    start_view = last_date - pd.Timedelta(days=5)
    end_view = future_dates[min(3, len(future_dates) - 1)] if future_dates else last_date
    fig.update_xaxes(range=[start_view, end_view])

    fig.update_xaxes(
        rangeslider=dict(visible=False),
        tickformatstops=[
            dict(dtickrange=[None, 1000 * 60 * 60 * 24 * 40], value="%d %b"),
            dict(dtickrange=[1000 * 60 * 60 * 24 * 40, 1000 * 60 * 60 * 24 * 370], value="%b %Y"),
            dict(dtickrange=[1000 * 60 * 60 * 24 * 370, None], value="%Y"),
        ],
    )

    if future_dates:
        fig.add_annotation(
            x=future_dates[0],
            y=upper[0],
            text="Prediksi Dimulai",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
        )

    return fig


def home_page():
    # NOTE: set_page_config ada di streamlit_app.py (router), bukan di sini
    ensure_state_defaults()
    load_css()

    # Creator text (left)
    st.markdown(
        '<div style="font-size:16px; opacity:.7; margin-top:6px;">Created by Dini Septiana & Ayu Andani</div>',
        unsafe_allow_html=True
    )

    # Cover image
    st.image("assets/Beranda.JPG", use_container_width=True)

    # Load data + prediction
    with st.spinner("Memuat data emas & prediksi..."):
        df_gold = get_gold_data_cached(st.session_state.data_version)
        pred = get_prediction_cached(st.session_state.data_version)

    # notif update (muncul sekali)
    if st.session_state.get("just_updated", False):
        st.success(f"✅ Data sudah paling update sampai tanggal {pred['last_date'].date()} (close terakhir).")
        st.session_state["just_updated"] = False

    # Row: price summary (left) | daily update (right)
    c1, c3 = st.columns([2.2, 1.7], vertical_alignment="center")

    with c1:
        close_str = format_id_number(pred["last_close"], decimals=3)
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:10px; margin-top:4px;">
              <div style="font-size:50px; font-weight:900; line-height:1;">{close_str}</div>
              <div style="font-size:18px; font-weight:850; opacity:.8;">USD</div>
            </div>
            <div style="font-size:16px; opacity:.7; margin-top:6px;">
              Data Harga Close Terbaru pada Tanggal {pred["last_date"].date()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown('<div style="font-size:33px; font-weight:900; line-height:1;">Daily Update</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:16px; opacity:.7; margin:6px 0 10px 0;">Klik untuk memperbarui semua data (maksimal sampai close terakhir).</div>',
            unsafe_allow_html=True
        )
        if st.button("Run Update", use_container_width=True):
            bump_data_version()
            st.session_state["just_updated"] = True
            st.rerun()

    # Chart section
    st.markdown('---')
    st.markdown('<div style="font-size:35px; font-weight:900; line-height:1;">Chart + Prediksi</div>', unsafe_allow_html=True)

    horizon = 22  # ~ 1 bulan hari kerja
    fig = make_chart(df_gold, pred, horizon)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    # KPI section
    st.markdown('<div style="font-size:35px; font-weight:900; line-height:1;">Fitur Prediksi Data</div>', unsafe_allow_html=True)

    open_str = format_id_number(float(df_gold.loc[pred["last_date"], "Open"]), decimals=3)
    close2_str = format_id_number(float(df_gold.loc[pred["last_date"], "Close"]), decimals=3)
    predH_str = format_id_number(pred["pred_close_H"], decimals=3)

    # nilai dasar
    open_val = float(df_gold.loc[pred["last_date"], "Open"])
    close_val = float(df_gold.loc[pred["last_date"], "Close"])
    pred_val = float(pred["pred_close_H"])

    close_prev = float(df_gold["Close"].iloc[-1])  # H-2

    # k1: Open vs Close kemarin-lagi
    delta_open = open_val - close_prev
    delta_open_pct = (delta_open / close_prev) * 100 if close_prev != 0 else 0.0

    # k2: Close vs Open hari yang sama
    delta_close = close_val - open_val
    delta_close_pct = (delta_close / open_val) * 100 if open_val != 0 else 0.0

    # k3: Prediksi vs Close terakhir
    delta_pred = pred_val - close_val
    delta_pred_pct = (delta_pred / close_val) * 100 if close_val != 0 else 0.0

    k1, k2, k3 = st.columns(3, gap="medium")

    def kpi_card(title: str, value_str: str, date_str: str, delta: float, delta_pct: float):
        up = delta >= 0
        arrow = "▲" if up else "▼"
        color = "#16a34a" if up else "#dc2626"
        bg = "rgba(22, 163, 74, 0.12)" if up else "rgba(220, 38, 38, 0.12)"

        delta_str = format_id_number(abs(delta), decimals=3)
        pct_str = f"{abs(delta_pct):.2f}%"

        st.markdown(
            f"""
            <div style="border:1px solid rgba(0,0,0,.12); border-radius:16px; padding:16px; background:rgba(255,255,255,.9); margin-top:20px;">
            <div style="font-size:13px; font-weight:900; opacity:.9;">{title}</div>

            <div style="display:flex; align-items:baseline; gap:8px; margin-top:10px;">
                <div style="font-size:30px; font-weight:900; line-height:1;">{value_str}</div>
                <div style="font-size:13px; font-weight:900; opacity:.8;">USD</div>
            </div>

            <div style="font-size:12px; opacity:.7; margin-top:6px;">
                {date_str}
            </div>

            <div style="margin-top:10px;">
                <span style="
                display:inline-flex; align-items:center; gap:6px;
                padding:4px 10px; border-radius:999px;
                background:{bg}; color:{color};
                font-weight:900; font-size:12px;
                ">
                {arrow} {delta_str} ({pct_str})
                </span>
            </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k1:
        kpi_card("Pembukaan Terakhir", open_str, str(pred["last_date"].date()), delta_open, delta_open_pct)
    with k2:
        kpi_card("Penutupan Terakhir", close2_str, str(pred["last_date"].date()), delta_close, delta_close_pct)
    with k3:
        kpi_card("Prediksi Besok", predH_str, str(pred["pred_date_H"].date()), delta_pred, delta_pred_pct)

    st.markdown('---')
    st.markdown('<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:10px; margin-top:20px;">Ringkasan Sentimen Berita</div>', unsafe_allow_html=True)

    df_news = get_news_cached(st.session_state.data_version)

    # tentukan range tanggal berita: 2026-01-01 sampai last_date (close terakhir dari prediksi beranda)
    start_news = pd.Timestamp("2026-01-01")
    end_news = pd.Timestamp(pred["last_date"]).normalize()
    range_label = f"({start_news.date()} s/d {end_news.date()})"

    if df_news.empty:
        st.info("Belum ada data berita. Silakan buka halaman Berita dan jalankan pengambilan berita.")
    else:
        df_news = df_news.copy()
        df_news["tanggal_dt"] = pd.to_datetime(df_news.get("tanggal", None), errors="coerce")
        df_news = df_news[(df_news["tanggal_dt"] >= start_news) & (df_news["tanggal_dt"] <= end_news)].copy()

        # ambil hanya yang relevan untuk KPI ringkasan (biar gak bias jadi 0 semua)
        if "relevan" in df_news.columns:
            df_rel = df_news[df_news["relevan"] == True].copy()
        else:
            df_rel = df_news.copy()

        # kalau masih kosong, fallback ke semua berita pada window
        if df_rel.empty:
            df_rel = df_news.copy()

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

        # faktor dominan: usahain bukan "Ekonomi" (kalau ada selain itu)
        faktor_dom = "Ekonomi"
        if "faktor" in df_rel.columns and not df_rel.empty:
            s = df_rel["faktor"].fillna("Ekonomi").astype(str)
            non_eco = s[s != "Ekonomi"]
            faktor_dom = (non_eco.value_counts().index[0] if len(non_eco) else s.value_counts().index[0])

        impact, prob_up, prob_dn, gold_score_mean = _impact_from_news_rule_based(df_rel)


        # Card style mirip berita.py tapi tetap seirama beranda.py
        def news_kpi_card(title: str, value_str: str, date_str: str, extra_html: str = ""):
            st.markdown(
                f"""
                <div style="border:1px solid rgba(0,0,0,.12); border-radius:16px; padding:16px; background:rgba(255,255,255,.9); margin-top:20px;">
                  <div style="font-size:13px; font-weight:900; opacity:.9;">{title}</div>
                  <div style="font-size:30px; font-weight:900; line-height:1; margin-top:10px;">{value_str}</div>
                  <div style="font-size:12px; opacity:.7; margin-top:6px;">
                    {date_str}
                  </div>
                  {extra_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        cA, cB, cC = st.columns(3, gap="medium")

        with cA:
            news_kpi_card("Sentimen Score (Mean)", f"{mean_sent:.4f}", range_label)

        with cB:
            news_kpi_card("Faktor Dominan", faktor_dom, range_label)

        with cC:
            impact_color = "#dc2626" if impact == "Bearish" else "#16a34a"

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
            """.strip()


            news_kpi_card("Dampak Emas", f"<span>{impact}</span>", range_label, extra_html=extra)

    # About section / analysis
    st.markdown(
        """
        <div style="font-size:16px; line-height:1.7; margin-top:18px; opacity:.9;">
        <b>Analisis Singkat:</b><br>
        Ketika hasil prediksi teknikal dan dampak emas dari berita menunjukkan <b>arah yang sama, maka:</b></br>
        maka hal ini mengindikasikan adanya konfirmasi antar kedua pendekatan (teknikal dan sentimen),</br> 
        sehingga meningkatkan tingkat kepercayaan terhadap prediksi.</br>
        </br> 
        Sebaliknya, apabila terjadi <b>perbedaan arah maka:</b></br>
        maka kondisi tersebut mencerminkan ketidakpastian keadaan pasar atau fase transisi,</br> 
        di mana keadaan ekonomi global oleh sentimen berita mendahului pergerakan harga aktual.</br> 
        sehingga disarankan untuk berhati-hati dalam mengambil keputusan investasi.</br>
        
        </div>
        """,
        unsafe_allow_html=True
    )    

    # =========================================================
    # TABEL FULL KOMBINASI: ALIGNMENT TEKNIKAL vs SENTIMEN (multi-faktor / pair)
    # =========================================================
    import itertools
    import re

    st.markdown('---')
    st.markdown(
        '<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:10px;">'
        'Kesesuaian Teknikal & Sentimen Berita (Full Kombinasi)'
        '</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Tabel ini membandingkan arah prediksi teknikal (Random Forest + median return) "
        "dengan sentimen berita FinBERT berdasarkan kombinasi faktor (bisa 2 faktor). Neutral tidak dipakai."
    )

    def _pct_range_str(lo: float, hi: float, decimals: int = 4) -> str:
        def fmt(x: float) -> str:
            return f"{x:+.{decimals}f}%".replace(".", ",")
        return f"{fmt(lo)} s/d {fmt(hi)}"

    def _score_range_str(lo: float, hi: float, decimals: int = 4) -> str:
        def fmt(x: float) -> str:
            return f"{x:+.{decimals}f}".replace(".", ",")
        return f"{fmt(lo)} s/d {fmt(hi)}"

    def _band_from_value(v: float, widen: float = 0.6) -> tuple[float, float]:
        v = float(v)
        a = v * (1 - widen)
        b = v * (1 + widen)
        return (a, b) if a <= b else (b, a)

    def _norm_text(x: str) -> str:
        x = str(x or "")
        x = x.lower()
        x = re.sub(r"\s+", " ", x).strip()
        return x

    # --- ambil data berita dari cache ---
    df_news = get_news_cached(st.session_state.data_version).copy()

    if df_news.empty:
        st.info("Belum ada data berita. Buka halaman Berita lalu jalankan pengambilan berita.")
    else:
        start_news = pd.Timestamp("2026-01-01")
        end_news = pd.Timestamp(pred["last_date"]).normalize()
        range_label = f"({start_news.date()} s/d {end_news.date()})"

        df_news["tanggal_dt"] = pd.to_datetime(df_news.get("tanggal", None), errors="coerce")
        df_news = df_news[(df_news["tanggal_dt"] >= start_news) & (df_news["tanggal_dt"] <= end_news)].copy()

        if "relevan" in df_news.columns:
            df_rel = df_news[df_news["relevan"] == True].copy()
            if df_rel.empty:
                df_rel = df_news.copy()
        else:
            df_rel = df_news.copy()

        if "sentiment_score" not in df_rel.columns:
            st.warning("Kolom 'sentiment_score' belum ada di berita.tsv.")
        else:
            # ---------------------------
            # 1) Teknis: rentang prediksi teknikal dari median return
            # ---------------------------
            feat_df = get_feature_df_cached(st.session_state.data_version)
            med = get_median_returns(feat_df)
            r_up = float(med.get("median_up", 0.0))
            r_dn = float(med.get("median_down", 0.0))

            up_lo, up_hi = _band_from_value(r_up * 100.0, widen=0.6)
            dn_lo, dn_hi = _band_from_value(r_dn * 100.0, widen=0.6)

            tech_range_up = _pct_range_str(up_lo, up_hi, decimals=4)
            tech_range_dn = _pct_range_str(dn_lo, dn_hi, decimals=4)

            # ---------------------------
            # 2) Deteksi MULTI FAKTOR per berita dari judul+isi pakai FACTOR_KEYWORDS kamu
            # ---------------------------
            
            # precompile keyword regex biar lebih cepat
            factor_patterns: dict[str, list[re.Pattern]] = {}
            for fac, kws in FACTOR_KEYWORDS.items():
                pats = []
                for kw in kws:
                    kw_norm = _norm_text(kw)
                    if not kw_norm:
                        continue
                    # match sebagai substring aman (lebih longgar)
                    pats.append(re.compile(re.escape(kw_norm)))
                factor_patterns[fac] = pats

            def detect_factors(title: str, body: str) -> list[str]:
                text = _norm_text(str(title) + " " + str(body))
                found = []
                for fac, pats in factor_patterns.items():
                    for p in pats:
                        if p.search(text):
                            found.append(fac)
                            break
                return found

            df_rel = df_rel.copy()
            df_rel["judul"] = df_rel.get("judul", "")
            df_rel["isi"] = df_rel.get("isi", "")
            df_rel["sentiment_score"] = pd.to_numeric(df_rel["sentiment_score"], errors="coerce")

            df_rel["factors_list"] = df_rel.apply(
                lambda r: detect_factors(r.get("judul", ""), r.get("isi", "")),
                axis=1
            )

            # Buang berita yang tidak kena faktor manapun (biar kombinasi meaningful)
            df_rel = df_rel[df_rel["factors_list"].map(lambda x: isinstance(x, list) and len(x) > 0)].copy()
            df_rel = df_rel.dropna(subset=["sentiment_score"])

            if df_rel.empty:
                st.warning("Tidak ada berita yang terdeteksi punya faktor dari FACTOR_KEYWORDS (atau sentiment_score kosong).")
            else:
                # ---------------------------
                # 3) Buat combo faktor:
                #    - kalau berita punya >=2 faktor: ambil semua pasangan (pair)
                #    - kalau cuma 1 faktor: tetap masuk sebagai single (opsional)
                # ---------------------------
                include_single_factor = True  # kalau mau wajib double saja, ubah False

                combo_rows = []
                for _, r in df_rel.iterrows():
                    facs = sorted(set(r["factors_list"]))
                    score = float(r["sentiment_score"])
                    if len(facs) >= 2:
                        for a, b in itertools.combinations(facs, 2):
                            combo_rows.append({"faktor_combo": f"{a}, {b}", "sentiment_score": score})
                    elif include_single_factor:
                        combo_rows.append({"faktor_combo": facs[0], "sentiment_score": score})

                df_combo = pd.DataFrame(combo_rows)
                if df_combo.empty:
                    st.warning("Tidak ada kombinasi faktor yang bisa dibentuk dari data berita.")
                else:
                    # ---------------------------
                    # 4) Ambil min-max sentimen REAL per combo & per sign
                    # ---------------------------
                    def sent_range_for_combo(combo: str, sign: str) -> tuple[float, float] | None:
                        dfx = df_combo[df_combo["faktor_combo"] == combo].copy()
                        if sign == "Positif":
                            dfx = dfx[dfx["sentiment_score"] > 0]
                        else:
                            dfx = dfx[dfx["sentiment_score"] < 0]
                        if dfx.empty:
                            return None
                        lo = float(dfx["sentiment_score"].min())
                        hi = float(dfx["sentiment_score"].max())
                        return (lo, hi) if lo <= hi else (hi, lo)

                    combos = sorted(df_combo["faktor_combo"].unique().tolist())

                    rows = []
                    for combo in combos:
                        for sign in ["Positif", "Negatif"]:
                            rng = sent_range_for_combo(combo, sign)

                            # kalau untuk combo ini tidak ada skor positif/negatif, skip aja (biar gak palsu)
                            if rng is None:
                                continue

                            sent_lo, sent_hi = rng
                            sent_range = _score_range_str(sent_lo, sent_hi, decimals=4)
                            impact = "Bullish" if sign == "Positif" else "Bearish"

                            for arah_tek in ["Naik", "Turun"]:
                                pred_tech_range = tech_range_up if arah_tek == "Naik" else tech_range_dn
                                searah = (
                                    (arah_tek == "Naik" and impact == "Bullish") or
                                    (arah_tek == "Turun" and impact == "Bearish")
                                )
                                alignment = "Searah" if searah else "Berlawanan"

                                rows.append({
                                    "Alignment": alignment,
                                    "Arah Teknikal": arah_tek,
                                    "Prediksi Teknikal (Kontinu)": pred_tech_range,
                                    "Sentimen Score": sent_range,
                                    "Faktor Dominan": combo,   # bisa 1 atau 2 faktor
                                    "Dampak Emas": impact,
                                })

                    df_align = pd.DataFrame(rows)
                    if df_align.empty:
                        st.warning("Kombinasi faktor ada, tapi tidak ada rentang sentimen (+/-) yang memenuhi.")
                    else:
                        # Searah dulu
                        df_align["_rank"] = df_align["Alignment"].map({"Searah": 0, "Berlawanan": 1}).fillna(9).astype(int)
                        df_align = df_align.sort_values(
                            ["_rank", "Faktor Dominan", "Arah Teknikal", "Dampak Emas"]
                        ).drop(columns=["_rank"])
                        
                        # =========================================================
                        # GABUNG ROW YANG SAMA → faktor jadi 1 sel (dipisah koma)
                        # =========================================================
                        group_cols = [
                            "Alignment",
                            "Arah Teknikal",
                            "Prediksi Teknikal (Kontinu)",
                            "Sentimen Score",
                            "Dampak Emas",
                        ]

                        def _join_unique_factors(series: pd.Series) -> str:
                            # split "Banking, Dolar" → flatten → unique → urut alfabet
                            items: list[str] = []
                            for x in series.dropna().astype(str):
                                for part in x.split(","):
                                    p = part.strip()
                                    if p:
                                        items.append(p)
                            uniq = sorted(set(items))
                            return ", ".join(uniq)

                        df_align = (
                            df_align
                            .groupby(group_cols, dropna=False, as_index=False)
                            .agg({"Faktor Dominan": _join_unique_factors})
                        )

                        # urutkan: Searah dulu, lalu Berlawanan (biar rapi)
                        df_align["_rank"] = df_align["Alignment"].map({"Searah": 0, "Berlawanan": 1}).fillna(9).astype(int)
                        df_align = df_align.sort_values(["_rank", "Arah Teknikal", "Dampak Emas", "Faktor Dominan"]).drop(columns=["_rank"])
                        
                        # =========================================================
                        # PISAHKAN TABEL: SEARAH vs BERLAWANAN
                        # =========================================================
                        df_searah = df_align[df_align["Alignment"] == "Searah"].copy()
                        df_berlawanan = df_align[df_align["Alignment"] == "Berlawanan"].copy()

                        st.markdown(
                            '<div style="font-size:28px; font-weight:900; margin-top:10px;">'
                            'Kondisi Searah (Teknikal & Sentimen Konsisten)'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.caption(
                            "Situasi di mana arah prediksi teknikal (Random Forest + median return) "
                            "sejalan dengan dampak emas hasil analisis sentimen FinBERT."
                        )

                        st.dataframe(
                            df_searah[
                                [
                                    "Alignment",
                                    "Arah Teknikal",
                                    "Prediksi Teknikal (Kontinu)",
                                    "Sentimen Score",
                                    "Faktor Dominan",
                                    "Dampak Emas",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown(
                            '<div style="font-size:28px; font-weight:900; margin-top:30px;">'
                            'Kondisi Berlawanan (Teknikal vs Sentimen Konflik)'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.caption(
                            "Situasi di mana sinyal teknikal dan sentimen berita memberikan arah yang berbeda. "
                            "Kondisi ini penting untuk analisis risiko dan ketidakpastian pasar."
                        )

                        st.dataframe(
                            df_berlawanan[
                                [
                                    "Alignment",
                                    "Arah Teknikal",
                                    "Prediksi Teknikal (Kontinu)",
                                    "Sentimen Score",
                                    "Faktor Dominan",
                                    "Dampak Emas",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )
