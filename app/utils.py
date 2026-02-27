from typing import Any

import requests
import streamlit as st

from app.config import logger


def safe_execute(func, *args) -> Any | None:
    try:
        return func(*args)
    except Exception as e:
        logger.exception(f"Error executing {func.__name__}")  # Full traceback to log
        st.error(f"An error occurred: {str(e)}")  # User-friendly message
        return None


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_response_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Log full final URL (including params)
        logger.info(f"Successfully fetched: {response.url}")

        return response.json()

    except requests.exceptions.RequestException as e:
        logger.warning(f"API request failed for {url} with params={params}: {e}")
        st.error(f"API request failed: {e}")
        return None
