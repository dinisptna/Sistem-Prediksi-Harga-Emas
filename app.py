import streamlit as st

from pages.beranda import home_page
from pages.prediksi import prediksi_page
from pages.berita import berita_page
from pages.tentang import tentang_page

st.set_page_config(page_title="Sistem Prediksi Harga Emas", layout="wide")

# inget page terakhir
if "last_page" not in st.session_state:
    st.session_state["last_page"] = "Beranda"

def _wrap(page_fn, page_name: str):
    def runner():
        st.session_state["last_page"] = page_name
        page_fn()
    # PENTING: bikin nama unik biar url pathname gak tabrakan
    runner.__name__ = f"page_{page_name.lower()}"
    runner.__qualname__ = runner.__name__
    return runner

pages = [
    st.Page(
        _wrap(home_page, "Beranda"),
        title="Beranda",
        icon=":material/home:",
        default=(st.session_state["last_page"] == "Beranda"),
    ),
    st.Page(
        _wrap(prediksi_page, "Prediksi"),
        title="Prediksi",
        icon=":material/show_chart:",
        default=(st.session_state["last_page"] == "Prediksi"),
    ),
    st.Page(
        _wrap(berita_page, "Berita"),
        title="Berita",
        icon=":material/newspaper:",
        default=(st.session_state["last_page"] == "Berita"),
    ),
    st.Page(
        _wrap(tentang_page, "Tentang"),
        title="Tentang",
        icon=":material/psychology:",
        default=(st.session_state["last_page"] == "Tentang"),
    ),
]

current_page = st.navigation(pages, position="top")
current_page.run()
