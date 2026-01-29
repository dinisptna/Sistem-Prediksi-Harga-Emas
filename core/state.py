import streamlit as st

def ensure_state_defaults() -> None:
    if "data_version" not in st.session_state:
        st.session_state.data_version = 0

def bump_data_version() -> None:
    st.session_state.data_version += 1
