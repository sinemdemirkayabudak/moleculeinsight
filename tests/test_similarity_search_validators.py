"""Tests for similarity_search validators module."""

import pandas as pd
import pytest

from app.similarity_search.validators import (
    validate_dataframe,
    validate_parameters,
)


class TestValidateParameters:
    """Test parameter validation for similarity search."""

    def test_valid_parameters(self):
        """Test valid parameter combinations."""
        validate_parameters(radius=2, top_n=20)
        validate_parameters(radius=0, top_n=1)
        validate_parameters(radius=10, top_n=1000)  # No upper limit on radius

    def test_invalid_radius_too_low(self):
        """Test radius below minimum."""
        with pytest.raises(ValueError, match="Radius must be"):
            validate_parameters(radius=-1, top_n=20)

    def test_invalid_top_n_zero(self):
        """Test top_n of zero."""
        with pytest.raises(ValueError, match="top_n must be|must be positive"):
            validate_parameters(radius=2, top_n=0)

    def test_invalid_top_n_negative(self):
        """Test negative top_n."""
        with pytest.raises(ValueError, match="top_n must be|must be positive"):
            validate_parameters(radius=2, top_n=-5)


class TestValidateDataframe:
    """Test dataframe validation."""

    def test_valid_dataframe(self):
        """Test valid dataframe with required columns."""
        df = pd.DataFrame({"smiles": ["CC", "CCO"], "name": ["ethane", "ethanol"]})
        validate_dataframe(df, "test", ["smiles", "name"])

    def test_empty_dataframe(self):
        """Test empty dataframe raises error."""
        df = pd.DataFrame(columns=["smiles", "name"])  # ty:ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df, "test", ["smiles", "name"])

    def test_missing_column(self):
        """Test missing required column."""
        df = pd.DataFrame({"smiles": ["CC"]})
        with pytest.raises(ValueError, match="name"):
            validate_dataframe(df, "test", ["smiles", "name"])

    def test_non_dataframe_input(self):
        """Test non-dataframe input."""
        with pytest.raises((ValueError, AttributeError)):
            validate_dataframe({"smiles": ["CC"]}, "test", ["smiles"])

    def test_case_sensitive_columns(self):
        """Test that column names are case-sensitive."""
        df = pd.DataFrame({"SMILES": ["CC"], "name": ["ethane"]})
        with pytest.raises(ValueError):
            validate_dataframe(df, "test", ["smiles", "name"])
