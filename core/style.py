from pathlib import Path
import streamlit as st

def load_css(css_path: str = "assets/style.css") -> None:
    # bikin path aman walau dipanggil dari pages/
    root = Path(__file__).resolve().parents[1]
    path = root / css_path
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS tidak ditemukan: {path}")

def hdiv() -> None:
    st.markdown("---")
    
