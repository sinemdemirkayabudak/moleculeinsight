"""Tests for utility functions - 100% coverage."""

from unittest.mock import MagicMock, call, patch

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

    @patch("app.utils.st.error")
    def test_exception_shows_error_message(self, mock_error):
        """Test that exceptions trigger st.error()."""

        def failing_func():
            raise ValueError("Test error")

        safe_execute(failing_func)
        assert mock_error.called

    @patch("app.utils.logger")
    @patch("app.utils.st.error")
    def test_exception_logs_traceback(self, mock_error, mock_logger):
        """Test that exceptions are logged with traceback."""

        def failing_func():
            raise ValueError("Test error")

        safe_execute(failing_func)

        assert mock_logger.exception.called

    def test_function_with_multiple_args(self):
        """Test with multiple arguments."""

        def multiply(a, b, c):
            return a * b * c

        result = safe_execute(multiply, 2, 3, 4)
        assert result == 24


class TestGetResponseJson:
    """Test API response retrieval - 100% coverage."""

    @patch("requests.get")
    def test_successful_json_response(self, mock_get):
        """Test successful JSON response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_response.url = "http://example.com/api"
        mock_get.return_value = mock_response

        result = get_response_json("http://example.com/api")
        assert result == {"key": "value"}

    @patch("requests.get")
    def test_successful_response_with_params(self, mock_get):
        """Test successful response with query parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": 123}
        mock_response.url = "http://example.com/api?key=value"
        mock_get.return_value = mock_response

        params = {"key": "value"}
        result = get_response_json("http://example.com/api", params=params)

        assert result == {"data": 123}
        mock_get.assert_called_once_with("http://example.com/api", params=params, timeout=60)

    @patch("app.utils.logger")
    @patch("app.utils.st.error")
    @patch("requests.get")
    def test_bad_status_code_404_returns_none(self, mock_get, mock_error, mock_logger):
        """Test that 404 status returns None without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        result = get_response_json("http://example.com/404")
        assert result is None
        assert mock_logger.warning.called

    @patch("app.utils.st.error")
    @patch("requests.get")
    def test_timeout_first_attempt_no_retry(self, mock_get, mock_error):
        """Test timeout on first attempt."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = get_response_json("http://slow-api.com", max_retries=1)
        assert result is None
        mock_error.assert_called_once()

    @patch("app.utils.st.error")
    @patch("requests.get")
    def test_connection_error_returns_none(self, mock_get, mock_error):
        """Test that connection error returns None."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = get_response_json("http://invalid.com")
        assert result is None

    @patch("time.sleep")
    @patch("requests.get")
    def test_timeout_with_retry_succeeds_on_second_attempt(self, mock_get, mock_sleep):
        """Test timeout retry succeeds on second attempt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.url = "http://example.com/api"

        # First call times out, second succeeds
        mock_get.side_effect = [requests.exceptions.Timeout(), mock_response]

        with patch("app.utils.st.error"):
            result = get_response_json("http://example.com/api", max_retries=3, retry_delay=0.5)

        assert result == {"success": True}
        assert mock_get.call_count == 2
        # First sleep should be 0.5 seconds (base_delay * 2^0)
        mock_sleep.assert_called_with(0.5)

    @patch("time.sleep")
    @patch("requests.get")
    def test_timeout_with_exponential_backoff(self, mock_get, mock_sleep):
        """Test exponential backoff for timeouts."""
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
        ]

        with patch("app.utils.st.error"):
            result = get_response_json("http://example.com/api", max_retries=3, retry_delay=1.0)

        assert result is None
        assert mock_get.call_count == 3
        # Should have 2 sleep calls: 1s and 2s (exponential backoff)
        assert mock_sleep.call_count == 2
        sleep_calls = [call(1.0), call(2.0)]
        mock_sleep.assert_has_calls(sleep_calls)

    @patch("time.sleep")
    @patch("requests.get")
    def test_rate_limit_429_retries_with_longer_backoff(self, mock_get, mock_sleep):
        """Test rate limit (429) with longer backoff."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {}
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Rate limited"
        )

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"data": "ok"}
        mock_response_success.url = "http://api.example.com"

        mock_get.side_effect = [mock_response_error, mock_response_success]

        with patch("app.utils.st.error"):
            result = get_response_json("http://api.example.com", max_retries=3)

        assert result == {"data": "ok"}
        # Rate limit backoff should be 5s for first retry (5 * 2^0)
        mock_sleep.assert_called_once_with(5)

    @patch("time.sleep")
    @patch("requests.get")
    def test_rate_limit_429_with_retry_after_header(self, mock_get, mock_sleep):
        """Test 429 respects Retry-After header."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {"Retry-After": "10"}
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Rate limited"
        )

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"ok": True}
        mock_response_success.url = "http://api.example.com"

        mock_get.side_effect = [mock_response_error, mock_response_success]

        with patch("app.utils.st.error"):
            result = get_response_json("http://api.example.com", max_retries=3)

        # Should use max of backoff (5s) and Retry-After (10s), so 10s
        mock_sleep.assert_called_once_with(10)
        assert result == {"ok": True}

    @patch("time.sleep")
    @patch("app.utils.logger")
    @patch("requests.get")
    def test_rate_limit_429_max_retries_exceeded(self, mock_get, mock_logger, mock_sleep):
        """Test 429 when max retries exceeded."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {}
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Rate limited"
        )

        mock_get.return_value = mock_response_error

        with patch("app.utils.st.error") as mock_error:
            result = get_response_json("http://api.example.com", max_retries=2)

        assert result is None
        # Should call get exactly max_retries times
        assert mock_get.call_count == 2
        mock_error.assert_called_once()

    @patch("time.sleep")
    @patch("requests.get")
    def test_server_error_5xx_retries(self, mock_get, mock_sleep):
        """Test 5xx server errors trigger retry."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 503
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Service unavailable"
        )

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"status": "ok"}
        mock_response_success.url = "http://api.example.com"

        mock_get.side_effect = [mock_response_error, mock_response_success]

        with patch("app.utils.st.error"):
            result = get_response_json("http://api.example.com", max_retries=3, retry_delay=1.0)

        assert result == {"status": "ok"}
        # Should retry with exponential backoff (1.0 * 2^0 = 1.0)
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep")
    @patch("app.utils.logger")
    @patch("requests.get")
    def test_server_error_500_max_retries_exceeded(self, mock_get, mock_logger, mock_sleep):
        """Test 500 when max retries exceeded."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 500
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Internal server error"
        )

        mock_get.return_value = mock_response_error

        with patch("app.utils.st.error"):
            result = get_response_json("http://api.example.com", max_retries=2, retry_delay=0.1)

        assert result is None
        assert mock_get.call_count == 2

    @patch("app.utils.st.error")
    @patch("requests.get")
    def test_generic_request_exception_no_retry(self, mock_get, mock_error):
        """Test generic RequestException without retry."""
        mock_get.side_effect = requests.exceptions.RequestException("Generic error")

        result = get_response_json("http://api.example.com")
        assert result is None

    @patch("requests.get")
    def test_custom_timeout_parameter(self, mock_get):
        """Test custom timeout parameter is passed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.url = "http://api.example.com"
        mock_get.return_value = mock_response

        get_response_json("http://api.example.com", timeout=120)

        mock_get.assert_called_once_with("http://api.example.com", params=None, timeout=120)

    @patch("requests.get")
    def test_custom_retry_delay_parameter(self, mock_get):
        """Test custom retry_delay parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.url = "http://api.example.com"
        mock_get.return_value = mock_response

        with patch("time.sleep") as mock_sleep:
            get_response_json("http://api.example.com", retry_delay=2.5)

        mock_sleep.assert_not_called()  # Success, no sleeps needed

    @patch("time.sleep")
    @patch("requests.get")
    def test_multiple_params_passed_correctly(self, mock_get, mock_sleep):
        """Test that all parameters work together."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.url = "http://api.example.com"
        mock_get.return_value = mock_response

        params = {"key": "value", "id": "123"}
        result = get_response_json(
            "http://api.example.com", params=params, max_retries=5, retry_delay=0.5, timeout=90
        )

        assert result == {"result": "ok"}
        mock_get.assert_called_once_with("http://api.example.com", params=params, timeout=90)

    @patch("time.sleep")
    @patch("requests.get")
    def test_rate_limit_429_with_invalid_retry_after_header(self, mock_get, mock_sleep):
        """Test 429 with invalid Retry-After header (non-numeric)."""
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {"Retry-After": "invalid"}
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Rate limited"
        )

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"ok": True}
        mock_response_success.url = "http://api.example.com"

        mock_get.side_effect = [mock_response_error, mock_response_success]

        with patch("app.utils.st.error"):
            result = get_response_json("http://api.example.com", max_retries=3)

        # Should use default backoff since Retry-After is invalid
        mock_sleep.assert_called_once_with(5)
        assert result == {"ok": True}

    @patch("requests.get")
    def test_max_retries_zero_returns_none(self, mock_get):
        """Test that max_retries=0 returns None without attempting request."""
        result = get_response_json("http://example.com", max_retries=0)

        assert result is None
        # Should never attempt the request if max_retries is 0
        mock_get.assert_not_called()
