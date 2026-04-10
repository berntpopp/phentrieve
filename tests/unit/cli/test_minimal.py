"""Minimal test file to check pytest collection speed."""

import pytest

pytestmark = pytest.mark.unit


def test_minimal():
    """Minimal test."""
    assert True
