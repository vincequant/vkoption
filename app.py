import streamlit as st

st.set_page_config(
    page_title="VK Option",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.switch_page("pages/portfolio.py")
