import time
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


def get_response_json(
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: int = 60,
) -> dict[str, Any] | None:
    """Fetch JSON from URL with retry logic for transient errors and rate limiting.

    Handles both server errors (5xx), rate limiting (429), and timeouts with appropriate backoff.

    Parameters:
        url: API endpoint URL
        params: Query parameters
        max_retries: Number of retry attempts for transient errors
        retry_delay: Initial delay between retries (exponential backoff)
        timeout: Request timeout in seconds (default 60, increased from 30 for slow ChEMBL queries)

    Returns:
        Parsed JSON response or None on failure
    """
    base_delay = retry_delay

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()

            logger.info(f"Successfully fetched: {response.url}")
            return response.json()

        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2**attempt)
                logger.warning(
                    f"API request timeout on attempt {attempt + 1}. Retrying in {wait_time:.1f}s..."
                )
                # Silently retry without showing warning to reduce clutter
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"API request timed out after {max_retries} attempts: {e}")
                st.error(
                    "⏱️ ChEMBL API request timed out. The server is not responding quickly. "
                    "Please try again in a few moments."
                )
                return None

        except requests.exceptions.HTTPError as e:
            status_code = response.status_code

            # Rate limit: 429 Too Many Requests
            if status_code == 429:
                if attempt < max_retries - 1:
                    # ChEMBL rate limit: use longer backoff
                    wait_time = 5 * (2**attempt)  # 5s, 10s, 20s - much longer for rate limits
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = max(wait_time, int(retry_after))
                        except ValueError:
                            pass

                    logger.warning(
                        f"ChEMBL API rate limited (429). "
                        f"Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}..."
                    )
                    # Silently retry without showing warning to reduce clutter
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("ChEMBL API rate limit exceeded after max retries")
                    st.error(
                        "🚫 ChEMBL API rate limit exceeded. "
                        "Please wait a few minutes before trying again. "
                        "This is a limitation of the free ChEMBL API."
                    )
                    break

            # Server errors: 5xx - retry with backoff
            elif 500 <= status_code < 600:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Server error {status_code} on attempt {attempt + 1}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"API HTTP error for {url}: {e}")
                    # Don't raise - let calling code handle gracefully
                    break
            else:
                # Client errors (4xx) - don't retry
                logger.warning(f"API HTTP error for {url}: {e}")
                break

        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed for {url}: {e}")
            break

    # Don't display errors here - let calling code handle them
    # This prevents duplicate error messages

    return None
