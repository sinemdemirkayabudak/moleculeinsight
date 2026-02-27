import streamlit as st

from app.ui import render_app

if __name__ == "__main__":
    st.set_page_config(page_title="MoleculeInsight", page_icon="🧬", layout="wide")
    render_app()
