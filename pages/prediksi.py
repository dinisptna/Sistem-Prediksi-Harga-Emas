from __future__ import annotations

import streamlit as st
import pandas as pd

from core.style import load_css
from core.state import ensure_state_defaults

from services.predictor import (
    compute_feature_frame,
    load_model,
    get_median_returns,
    predict_direction,
)
from services.market_data import fetch_gold_ohlcv
from services.history_store import (
    ensure_history_csv,
    load_history,
    add_record,
    update_record,
    delete_record,
)

# ---------- helpers ----------
def format_id_number(x: float, decimals: int = 3) -> str:
    s = f"{x:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def keep_up_to_last_close_day(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    return df[df.index.normalize() < today].copy()

def next_n_business_days(anchor_date: pd.Timestamp, n: int) -> pd.Timestamp:
    d = pd.Timestamp(anchor_date)
    cnt = 0
    while cnt < n:
        d += pd.Timedelta(days=1)
        if d.dayofweek < 5:
            cnt += 1
    return d

def compute_multi_horizon_prices(
    harga_input: float,
    pred_dir: int,
    med: dict,
    anchor_date: pd.Timestamp,
) -> dict:
    """
    H+1, H+5, H+22 hari kerja.
    Pakai r median berdasarkan pred_dir.
    Compounding: Pn = P0*(1+r)^n
    """
    r = float(med["median_up"] if pred_dir == 1 else med["median_down"])
    p0 = float(harga_input)

    def price_n(n: int) -> float:
        return float(p0 * ((1 + r) ** n))

    p1 = price_n(1)
    p5 = price_n(5)
    p22 = price_n(22)

    # pct vs input
    pct1 = (p1 / p0 - 1) * 100 if p0 != 0 else 0.0
    pct5 = (p5 / p0 - 1) * 100 if p0 != 0 else 0.0
    pct22 = (p22 / p0 - 1) * 100 if p0 != 0 else 0.0

    d1 = next_n_business_days(anchor_date, 1)
    d5 = next_n_business_days(anchor_date, 5)
    d22 = next_n_business_days(anchor_date, 22)

    return {
        "r_used": r,
        "tanggal_besok": d1,
        "harga_besok": p1,
        "pct_besok": pct1,
        "tanggal_minggu": d5,
        "harga_minggu": p5,
        "pct_minggu": pct5,
        "tanggal_bulan": d22,
        "harga_bulan": p22,
        "pct_bulan": pct22,
    }

@st.cache_data(show_spinner=False)
def _get_gold_data_for_model() -> pd.DataFrame:
    """
    Data yfinance untuk membuat feature terakhir (basis H-1),
    dan untuk menghitung median return.
    """
    df = fetch_gold_ohlcv(start="2021-01-19", end=None, interval="1d")
    df = keep_up_to_last_close_day(df)
    return df

@st.cache_data(show_spinner=False)
def _get_feat_for_model() -> pd.DataFrame:
    df = _get_gold_data_for_model()
    feat = compute_feature_frame(df)
    return feat


def _badge(pct: float) -> str:
    up = pct >= 0
    arrow = "🔼" if up else "🔽"
    color = "#16a34a" if up else "#dc2626"
    return f"<span style='color:{color}; font-weight:900;'>{arrow} {pct:+.2f}%</span>"


def prediksi_page():
    ensure_state_defaults()
    load_css()
    ensure_history_csv()  # auto create data/prediksi_history.csv

    # Header
    st.markdown(
        '<div style="font-size:16px; opacity:.7; margin-top:6px;">Created by Dini Septiana & Ayu Andani</div>',
        unsafe_allow_html=True
    )
    st.image("assets/Prediksi.jpg", use_container_width=True)

    # ambil data model sekali
    feat = _get_feat_for_model()
    model = load_model("models/rf_gold_direction_model.pkl")
    med = get_median_returns(feat)

    # -------- Form input prediksi --------
    st.markdown('<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:30px; margin-top:10px;">Input Prediksi (Create)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        tanggal_input = st.date_input("**Tanggal**", value=pd.Timestamp.today().date())
    with c2:
        harga_input = st.number_input("**Harga Emas (USD)**", min_value=0.0, value=0.0, step=0.1)

    do_predict = st.button("Prediksi", use_container_width=True)

    # tempat simpan hasil prediksi terakhir di session (biar bisa dipakai saat simpan/history)
    if "last_pred_result" not in st.session_state:
        st.session_state["last_pred_result"] = None

    if do_predict:
        # latest row untuk model prediction (basis H-1 / last available features)
        latest = feat.iloc[-1]
        pred_dir = predict_direction(model, latest)

        anchor = pd.Timestamp(tanggal_input)
        results = compute_multi_horizon_prices(
            harga_input=float(harga_input),
            pred_dir=pred_dir,
            med=med,
            anchor_date=anchor,
        )

        st.session_state["last_pred_result"] = {
            "tanggal_input": str(anchor.date()),
            "harga_input": float(harga_input),
            "pred_dir": int(pred_dir),
            **results,
        }

        # simpan ke CSV langsung setelah prediksi
        rec = {
            "tanggal_input": str(anchor.date()),
            "harga_input": float(harga_input),
            "tanggal_besok": str(results["tanggal_besok"].date()),
            "harga_besok": float(results["harga_besok"]),
            "pct_besok": float(results["pct_besok"]),
            "tanggal_minggu": str(results["tanggal_minggu"].date()),
            "harga_minggu": float(results["harga_minggu"]),
            "pct_minggu": float(results["pct_minggu"]),
            "tanggal_bulan": str(results["tanggal_bulan"].date()),
            "harga_bulan": float(results["harga_bulan"]),
            "pct_bulan": float(results["pct_bulan"]),
        }
        new_id = add_record(rec)
        st.success(f"✅ Prediksi tersimpan ke riwayat. (id: {new_id})")

    # -------- tampilkan hasil prediksi terakhir --------
    if st.session_state["last_pred_result"] is not None:
        r = st.session_state["last_pred_result"]

        st.markdown("### Hasil Prediksi")
        # Besok
        st.markdown(
            f"**Prediksi Besok** → ({r['tanggal_besok'].date()}) = "
            f"**{format_id_number(r['harga_besok'], 3)} USD** "
            f"{_badge(r['pct_besok'])}",
            unsafe_allow_html=True
        )
        # Minggu
        st.markdown(
            f"**Prediksi 1 Minggu** → ({r['tanggal_minggu'].date()}) = "
            f"**{format_id_number(r['harga_minggu'], 3)} USD** "
            f"{_badge(r['pct_minggu'])}",
            unsafe_allow_html=True
        )
        # Bulan
        st.markdown(
            f"**Prediksi 1 Bulan** → ({r['tanggal_bulan'].date()}) = "
            f"**{format_id_number(r['harga_bulan'], 3)} USD** "
            f"{_badge(r['pct_bulan'])}",
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="font-size:16px; opacity:.85;">
            dimana hasil ini berdasarkan prediksi secara teknikal<br>
            diluar dengan hasil kondisi/berita<br>
            yg dapat mempengaruhi harga secara mendadak<br>
            jika ingin melihat berita dengan analisa sentimen pada <b>Halaman Berita</b>.
            </div>
            """,
            unsafe_allow_html=True
        )
    # -------- Riwayat + CRUD --------
    st.markdown("---")
    st.markdown('<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:30px;">Riwayat Prediksi (Read)</div>', unsafe_allow_html=True)

    df_hist = load_history()

    if df_hist.empty:
        st.info("Belum ada riwayat prediksi.")
        return

    # tampilan: No + id (pendek) setelah No
    df_show = df_hist.copy().reset_index(drop=True)
    df_show.insert(0, "No", range(1, len(df_show) + 1))
    # pastikan id di kolom ke-2 (setelah No)
    cols_order = ["No", "id"] + [c for c in df_show.columns if c not in ["No", "id"]]
    df_show = df_show[cols_order]

    st.dataframe(
        df_show[[
            "No", "id",
            "tanggal_input", "harga_input",
            "tanggal_besok", "harga_besok",
            "tanggal_minggu", "harga_minggu",
            "tanggal_bulan", "harga_bulan",
        ]],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")

    # header: judul kiri + search kanan
    h1, h2 = st.columns([1.2, 0.8], vertical_alignment="center")
    with h1:
        st.markdown('<div style="font-size:35px; font-weight:900; line-height:1; margin-bottom:40px; margin-top:30px;">Aksi per Baris<br>(Update dan Delete)</div>', unsafe_allow_html=True)
    with h2:
        q = st.text_input(
            "Telusuri",
            placeholder="Cari id / tanggal / harga...",
            label_visibility="collapsed",
            key="search_prediksi",
        )

    # ---- NOTIFIKASI (DI BAWAH AKSI PER BARIS) ----
    if "action_msg" in st.session_state:
        if st.session_state.get("action_type") == "success":
            st.success(st.session_state["action_msg"])
        elif st.session_state.get("action_type") == "warning":
            st.warning(st.session_state["action_msg"])
        elif st.session_state.get("action_type") == "error":
            st.error(st.session_state["action_msg"])

        # auto-clear supaya sekali tampil
        del st.session_state["action_msg"]
        del st.session_state["action_type"]

    df_work = df_hist.copy()

    # filter kalau ada query
    if q and q.strip():
        qq = q.strip().lower()

        def _match(row):
            return (
                qq in str(row.get("id", "")).lower()
                or qq in str(row.get("tanggal_input", "")).lower()
                or qq in str(row.get("harga_input", "")).lower()
                or qq in str(row.get("tanggal_besok", "")).lower()
                or qq in str(row.get("tanggal_minggu", "")).lower()
                or qq in str(row.get("tanggal_bulan", "")).lower()
            )

        df_work = df_work[df_work.apply(_match, axis=1)]

    if df_work.empty:
        st.info("Tidak ada data yang cocok dengan pencarian.")
    else:
        for i, row in df_work.reset_index(drop=True).iterrows():
            rid = str(row["id"])

            with st.container(border=True):
                top = st.columns([2.2, 1, 1], vertical_alignment="center")

                with top[0]:
                    st.markdown(
                        f"**{i+1}. ID:** `{rid}`  \n"
                        f"**Tanggal:** {row['tanggal_input']}  \n"
                        f"**Harga:** {float(row['harga_input']):,.3f} USD"
                    )

                with top[1]:
                    do_edit = st.button("Update", key=f"btn_update_{rid}", use_container_width=True)

                with top[2]:
                    if st.button("Delete", key=f"btn_delete_{rid}", use_container_width=True):
                        ok = delete_record(rid)
                        if ok:
                            st.session_state["action_msg"] = f"Prediksi ID {rid} berhasil dihapus"
                            st.session_state["action_type"] = "success"
                        else:
                            st.session_state["action_msg"] = f"Gagal menghapus ID {rid}"
                            st.session_state["action_type"] = "error"
                        st.rerun()
                # Form update
                if do_edit:
                    st.session_state["edit_id"] = rid

                if st.session_state.get("edit_id") == rid:
                    st.markdown("**Edit Tanggal & Harga (Re-Predict otomatis)**")

                    c1u, c2u, c3u = st.columns([1, 1, 1])
                    with c1u:
                        new_date = st.date_input(
                            "Tanggal (Update)",
                            value=pd.to_datetime(row["tanggal_input"]).date(),
                            key=f"date_{rid}"
                        )
                    with c2u:
                        new_price = st.number_input(
                            "Harga Emas (USD) (Update)",
                            min_value=0.0,
                            value=float(row["harga_input"]) if pd.notna(row["harga_input"]) else 0.0,
                            step=0.1,
                            key=f"price_{rid}"
                        )
                    with c3u:
                        st.write("")
                        st.write("")
                        cancel = st.button("Batal", key=f"cancel_{rid}", use_container_width=True)

                    if cancel:
                        st.session_state["edit_id"] = None
                        st.rerun()

                    if st.button("Simpan Update", key=f"save_{rid}", type="primary", use_container_width=True):
                        latest = feat.iloc[-1]
                        pred_dir = predict_direction(model, latest)

                        anchor = pd.Timestamp(new_date)
                        results = compute_multi_horizon_prices(
                            harga_input=float(new_price),
                            pred_dir=pred_dir,
                            med=med,
                            anchor_date=anchor,
                        )

                        updates = {
                            "tanggal_input": str(anchor.date()),
                            "harga_input": float(new_price),
                            "tanggal_besok": str(results["tanggal_besok"].date()),
                            "harga_besok": float(results["harga_besok"]),
                            "pct_besok": float(results["pct_besok"]),
                            "tanggal_minggu": str(results["tanggal_minggu"].date()),
                            "harga_minggu": float(results["harga_minggu"]),
                            "pct_minggu": float(results["pct_minggu"]),
                            "tanggal_bulan": str(results["tanggal_bulan"].date()),
                            "harga_bulan": float(results["harga_bulan"]),
                            "pct_bulan": float(results["pct_bulan"]),
                        }

                        ok = update_record(rid, updates)
                        if ok:
                            st.session_state["action_msg"] = f"Prediksi ID {rid} berhasil diupdate"
                            st.session_state["action_type"] = "success"
                            st.session_state["edit_id"] = None
                        else:
                            st.session_state["action_msg"] = f"Gagal update ID {rid}"
                            st.session_state["action_type"] = "error"

                        st.rerun()