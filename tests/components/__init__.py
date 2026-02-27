"""Tests for UI card components."""

from unittest.mock import patch


class TestRenderMetricCard:
    """Test metric card rendering functionality."""

    @patch("streamlit.markdown")
    def test_render_metric_card_with_valid_inputs(self, mock_markdown):
        """Test rendering metric card with valid inputs."""
        from app.components.cards import render_metric_card

        # Call the function
        render_metric_card("CPU Usage", "45%", "🖥️")

        # Verify streamlit.markdown was called
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        # Verify the content contains expected values
        assert "CPU Usage" in call_args
        assert "45%" in call_args
        assert "🖥️" in call_args

    @patch("streamlit.markdown")
    def test_render_metric_card_with_numeric_value(self, mock_markdown):
        """Test rendering metric card with numeric value."""
        from app.components.cards import render_metric_card

        render_metric_card("Temperature", 72.5, "🌡️")

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        assert "Temperature" in call_args
        assert "72.5" in call_args

    @patch("streamlit.markdown")
    def test_render_metric_card_markdown_contains_styling(self, mock_markdown):
        """Test that rendered metric card includes styling."""
        from app.components.cards import render_metric_card

        render_metric_card("Test", "Value", "📊")

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        # Verify HTML/CSS styling is included
        assert "padding" in call_args or "style=" in call_args
        assert "unsafe_allow_html=True" in str(mock_markdown.call_args)

    @patch("streamlit.markdown")
    def test_render_metric_card_with_special_characters(self, mock_markdown):
        """Test rendering metric card with special characters in values."""
        from app.components.cards import render_metric_card

        render_metric_card("Value", "$1,234.56", "💰")

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        assert "$1,234.56" in call_args
