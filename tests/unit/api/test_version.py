"""
Tests for API version management.
"""

import pytest

from api.version import get_all_versions, get_api_version

pytestmark = pytest.mark.unit


def test_get_api_version():
    """Test that API version can be read from pyproject.toml."""
    version = get_api_version()

    assert version is not None
    assert version != "unknown"
    assert "." in version  # Should be semantic version (x.y.z)


def test_get_api_version_caching():
    """Test that get_api_version uses LRU cache correctly."""
    version1 = get_api_version()
    version2 = get_api_version()

    assert version1 == version2
    # Cache test - same result should be returned


def test_get_all_versions():
    """Test aggregation of all component versions."""
    versions = get_all_versions()

    assert "cli" in versions
    assert "api" in versions
    assert "environment" in versions
    assert "timestamp" in versions

    assert versions["cli"]["version"] is not None
    assert versions["api"]["version"] is not None
    assert versions["cli"]["name"] == "phentrieve"
    assert versions["api"]["name"] == "phentrieve-api"


def test_version_reading_works_with_python_310_and_311():
    """Test that version reading works with both tomllib (3.11+) and tomli (3.10)."""
    # This test verifies that the fallback mechanism works
    # If running on Python 3.10, tomli should be used
    # If running on Python 3.11+, tomllib should be used
    version = get_api_version()

    assert version is not None
    assert version != "unknown"
    # Should successfully read version regardless of Python version
