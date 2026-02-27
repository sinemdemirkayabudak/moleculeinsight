"""Tests for utility functions."""

from unittest.mock import MagicMock, patch

import requests

from app.utils import get_response_json, safe_execute


class TestSafeExecute:
    """Test safe execution wrapper."""

    def test_successful_execution(self):
        """Test that successful call returns result."""

        def add(a, b):
            return a + b

        result = safe_execute(add, 2, 3)
        assert result == 5

    def test_function_with_no_args(self):
        """Test execution of function with no arguments."""

        def get_message():
            return "Success"

        result = safe_execute(get_message)
        assert result == "Success"

    def test_exception_returns_none(self):
        """Test that exceptions return None."""

        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func)
        assert result is None

    @patch("streamlit.error")
    def test_exception_shows_error_message(self, mock_error):
        """Test that exceptions trigger st.error()."""

        def failing_func():
            raise ValueError("Test error")

        safe_execute(failing_func)
        assert mock_error.called


class TestGetResponseJson:
    """Test API response retrieval."""

    @patch("requests.get")
    def test_successful_json_response(self, mock_get):
        """Test successful JSON response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = get_response_json("http://example.com/api")
        assert result == {"key": "value"}

    @patch("streamlit.error")
    @patch("requests.get")
    def test_bad_status_code_returns_none(self, mock_get, mock_error):
        """Test that bad status returns None."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        result = get_response_json("http://example.com/404")
        assert result is None

    @patch("streamlit.error")
    @patch("requests.get")
    def test_timeout_returns_none(self, mock_get, mock_error):
        """Test that timeout returns None."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = get_response_json("http://slow-api.com")
        assert result is None

    @patch("streamlit.error")
    @patch("requests.get")
    def test_connection_error_returns_none(self, mock_get, mock_error):
        """Test that connection error returns None."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = get_response_json("http://invalid.com")
        assert result is None
