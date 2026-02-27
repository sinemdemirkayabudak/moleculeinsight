import streamlit as st


def render_metric_card(title, value, emoji):

    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:12px;
            background:#1e1e1e;
            text-align:center;
        ">

        <div style="
            font-size:15px;
            opacity:0.8;
            margin-bottom:6px;
        ">
            {emoji} {title}
        </div>

        <div style="
            font-size:22px;
            font-weight:bold;
        ">
            {value}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )
