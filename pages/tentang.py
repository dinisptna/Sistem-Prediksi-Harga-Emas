from __future__ import annotations

import streamlit as st

from core.style import load_css
from core.state import ensure_state_defaults


def _h1(title: str):
    st.markdown(
        f'<div style="font-size:42px; font-weight:900; line-height:1.05; margin-top:8px; margin-bottom:10px;">{title}</div>',
        unsafe_allow_html=True,
    )


def _h2(title: str):
    st.markdown(
        f'<div style="font-size:30px; font-weight:900; line-height:1.1; margin-top:26px; margin-bottom:10px;">{title}</div>',
        unsafe_allow_html=True,
    )


def _p(html: str):
    st.markdown(
        f'<div style="font-size:16px; line-height:1.75; opacity:.9; margin-top:6px;">{html}</div>',
        unsafe_allow_html=True,
    )


def _badge(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-flex; align-items:center;
            padding:6px 12px; border-radius:999px;
            background:rgba(59,130,246,0.12);
            color:#1d4ed8;
            font-weight:900; font-size:12px;
            margin-right:8px;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )


def _step_card(step_no: str, title: str, desc_html: str):
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(0,0,0,.12);
            border-radius:16px;
            padding:16px;
            background:rgba(255,255,255,.9);
            margin-top:12px;
        ">
          <div style="display:flex; align-items:baseline; gap:10px;">
            <div style="font-size:14px; font-weight:900; opacity:.8;">{step_no}</div>
            <div style="font-size:18px; font-weight:900; line-height:1.1;">{title}</div>
          </div>
          <div style="font-size:15px; line-height:1.75; opacity:.9; margin-top:10px;">
            {desc_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tentang_page():
    ensure_state_defaults()
    load_css()

    # Header (creator)
    st.markdown(
        '<div style="font-size:16px; opacity:.7; margin-top:6px;">Created by Dini Septiana & Ayu Andani</div>',
        unsafe_allow_html=True
    )

    _h1("Analisis Sistem")

    _p(
        """
        Halaman ini menjelaskan mekanisme sistem prediksi harga emas secara end-to-end.
        Sistem dirancang dengan pendekatan <b>hybrid</b>, yaitu menggabungkan:
        <b>analisis teknikal</b> (berbasis data harga historis) dan
        <b>analisis sentimen berita</b> (berbasis teks berita global menggunakan FinBERT).
        Hasil akhir digunakan untuk membantu interpretasi arah dan potensi pergerakan harga emas.</br>
        </br>
        Prediksi harga emas pada sistem ini diperoleh melalui pendekatan <b>hybrid</b>,<br> 
        yaitu mengombinasikan <b>analisis teknikal</b> dan <b>analisis sentimen berita</b>.</br>
        Pada sisi teknikal, arah pergerakan harga diprediksi menggunakan <b>model Random Forest</b><br> 
        berbasis data historis bersumber yfinance GC=F,</br> 
        kemudian dikonversi menjadi <b>nilai harga kontinu</b> melalui estimasi return median historis.</br>
        </br>
        Sementara itu pada sisi berita ekonomi global, sistem menganalisis <b>menggunakan FinBERT</b></br> 
        untuk memperoleh sentiment score yang merepresentasikan persepsi pasar.</br>
        Sentimen tersebut kemudian dikaitkan dengan faktor dominan yang memengaruhi harga emas,</br> 
        seperti suku bunga atau geopolitik, untuk menentukan <b>dampak emas (Bullish/Bearish)</b>.</br>
        </br>
        """
    )

    st.markdown("---")

    _h2('<div style="margin-top:20px; margin-bottom:30px;">Ringkasan Komponen Utama</div>')
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        _badge("Teknikal")
        _p(
            """
            Data harga emas (GC=F) diolah menjadi fitur teknikal (return, moving average ratio,
            volatilitas, momentum). Model <b>Random Forest</b> memprediksi arah (naik/turun).
            """
        )
    with c2:
        _badge("Nilai Kontinu")
        _p(
            """
            Setelah arah diprediksi, sistem memakai <b>median return historis</b> untuk memproyeksikan
            harga (lebih robust terhadap outlier dibanding mean).
            """
        )
    with c3:
        _badge("Berita + Sentimen")
        _p(
            """
            Berita global diambil via RSS dan disimpan ke <code>data/berita.tsv</code>.
            Sentimen dianalisis menggunakan <b>FinBERT</b> lalu dipetakan ke faktor dominan dan dampak emas.
            """
        )

    st.markdown("---")

    _h2("Alur Kerja Sistem (End-to-End)")

    _step_card(
        "1) Pengumpulan Data Harga",
        "Ambil data OHLCV emas (GC=F) dan batasi hingga close terakhir",
        """
        Sumber data harga menggunakan <b>yfinance</b> (GC=F). Data difilter agar hanya memuat data
        yang sudah memiliki <i>close</i> (tidak termasuk hari berjalan), sehingga konsisten untuk training/prediksi.
        """
    )

    _step_card(
        "2) Pembuatan Fitur Teknikal",
        "Transformasi harga menjadi fitur yang relevan untuk model",
        """
        Data harga diubah menjadi fitur seperti:
        <ul style="margin:6px 0 0 18px;">
          <li><b>daily_return</b> (persentase perubahan harian)</li>
          <li><b>price_ma5_ratio</b> dan <b>price_ma20_ratio</b> (rasio harga terhadap MA)</li>
          <li><b>volatility_10</b> (std return rolling 10 hari)</li>
          <li><b>momentum_5_pct</b> (perubahan 5 hari)</li>
        </ul>
        """
    )

    _step_card(
        "3) Prediksi Arah (Klasifikasi)",
        "Random Forest memprediksi naik/turun untuk hari berikutnya",
        """
        Model <b>Random Forest</b> mempelajari pola historis fitur teknikal untuk menghasilkan output
        <b>arah</b> (1 = naik, 0 = turun). Keunggulan RF adalah mampu menangkap hubungan non-linear
        dan tahan terhadap noise pada fitur pasar.
        """
    )

    _step_card(
        "4) Proyeksi Harga (Kontinu) via Median Return",
        "Konversi arah menjadi proyeksi harga menggunakan median return historis",
        """
        Karena model menghasilkan arah (diskrit), sistem mengubahnya menjadi nilai harga kontinu dengan:
        <br><br>
        <b>Return median naik</b> diambil dari histori saat target=1, dan <b>return median turun</b> dari histori saat target=0.
        Proyeksi harga dihitung dengan pendekatan kompound:
        <br><br>
        <code>P(n) = P0 × (1 + r)<sup>n</sup></code>
        <br><br>
        Untuk horizon:
        <ul style="margin:6px 0 0 18px;">
          <li><b>H+1</b> (besok) ≈ 1 hari kerja</li>
          <li><b>H+5</b> (1 minggu) ≈ 5 hari kerja</li>
          <li><b>H+22</b> (1 bulan) ≈ 22 hari kerja</li>
        </ul>
        """
    )

    _step_card(
        "5) Pengumpulan Berita (RSS) + Penyimpanan Lokal",
        "Ambil berita global dan simpan agar tidak fetch ulang",
        """
        Sistem mengambil berita global dari banyak sumber RSS lalu menyimpan hasilnya ke file
        <code>data/berita.tsv</code>. Dengan cara ini, aplikasi tidak perlu mengambil ulang semua berita
        pada setiap refresh (lebih cepat dan stabil). Tombol update akan melakukan <i>append</i> berita baru
        sesuai rentang tanggal yang dibutuhkan.
        """
    )

    _step_card(
        "6) Preprocessing + Sentimen per Berita (FinBERT)",
        "Hitung skor sentimen -1..1 untuk setiap berita",
        """
        Untuk setiap berita, teks (judul + ringkasan/isi yang tersedia dari RSS) dianalisis menggunakan
        <b>FinBERT</b> sehingga menghasilkan <b>sentiment score</b> numerik.
        Skor ini kemudian digunakan dalam agregasi harian dan ringkasan KPI.
        """
    )

    _step_card(
        "7) Klasifikasi Faktor Dominan (Rule/Keyword-Based)",
        "Deteksi topik yang paling dominan memengaruhi emas",
        """
        Sistem memetakan berita ke faktor seperti <b>Suku Bunga</b>, <b>Inflasi</b>, <b>Geopolitik</b>, <b>Resesi</b>,
        <b>Dolar AS</b>, <b>Pasar Saham/Risiko</b>, dan lainnya.
        Faktor dominan dihitung dari frekuensi faktor pada berita yang dianggap relevan terhadap emas.
        """
    )

    _step_card(
        "8) Dampak Emas + Probabilitas",
        "Gabungkan sentimen dan faktor untuk dampak Bullish/Bearish",
        """
        Dampak emas ditentukan dari agregasi sentimen pada berita relevan:
        <b>score ≥ 0 → Bullish</b>, <b>score &lt; 0 → Bearish</b>.
        Probabilitas naik/turun dihitung dari pemetaan score ke rentang probabilitas (dikontrol agar tidak ekstrem),
        kemudian ditampilkan pada KPI ringkasan.
        """
    )

    st.markdown("---")

    _h2("Cara Membaca Hasil (Teknikal vs Berita)")
    _p(
        """
        <b>Jika Prediksi Teknikal dan Dampak Berita searah</b>, maka terjadi <i>konfirmasi</i> antar pendekatan
        sehingga interpretasi prediksi lebih kuat.
        <br><br>
        <b>Jika berbeda arah</b>, hal ini dapat mengindikasikan adanya fase transisi pasar, berita yang bersifat
        mendadak, atau sentimen yang mendahului pergerakan harga. Pada kondisi ini, prediksi sebaiknya dibaca
        sebagai sinyal yang perlu dipantau bersama perkembangan berita terbaru.
        """
    )

    st.markdown("---")

    _h2("Catatan Keterbatasan")
    _p(
        """
        Sistem ini berbasis data historis dan ringkasan berita RSS. Perubahan mendadak akibat rilis data makro,
        keputusan bank sentral, atau eskalasi geopolitik dapat membuat pergerakan harga melenceng dari pola historis.
        Oleh karena itu, hasil prediksi bersifat <b>indikatif</b> dan sebaiknya digunakan sebagai dukungan analisis,
        bukan satu-satunya dasar keputusan.
        """
    )
