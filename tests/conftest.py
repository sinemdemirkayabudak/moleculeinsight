"""Pytest configuration and fixtures for MoleculeInsight tests."""

import pytest
from rdkit import Chem


def pytest_configure(config):
    """Patch streamlit.cache_data before anything is imported."""

    # This runs before any tests or imports, so we can patch globally
    def cache_data_passthrough(**kwargs):
        """Pass-through decorator that doesn't cache (for testing)."""

        def decorator(func):
            return func

        return decorator

    # Patch at module level before imports
    import streamlit

    streamlit.cache_data = cache_data_passthrough  # ty:ignore[invalid-assignment]


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Clear Streamlit cache before each test to prevent cross-test contamination."""
    # Clear Streamlit's cache data before the test runs
    try:
        from streamlit.runtime.caching import clear_cache  # ty:ignore[unresolved-import]

        clear_cache()
    except ImportError:
        # For different Streamlit versions, try alternative approach
        try:
            import streamlit as st

            # Access the internal cache and clear it
            if hasattr(st, "_cache"):
                st._cache.clear()  # ty:ignore[unresolved-attribute]
        except Exception:
            # If clearing fails, continue with test - not critical
            pass

    yield

    # Clean up after test if needed
    try:
        from streamlit.runtime.caching import clear_cache  # ty:ignore[unresolved-import]

        clear_cache()
    except ImportError:
        pass


@pytest.fixture
def benzene_molecule():
    """Create a benzene molecule for testing."""
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture
def ethanol_molecule():
    """Create an ethanol molecule for testing."""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def aspirin_molecule():
    """Create an aspirin molecule for testing."""
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
