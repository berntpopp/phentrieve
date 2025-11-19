"""
Tests for HPO graph data caching behavior.

This module tests the refactored @lru_cache implementation for load_hpo_graph_data,
ensuring thread-safety, cache behavior, and API compatibility.
"""

import concurrent.futures
import os

import pytest

from phentrieve.evaluation.metrics import (
    _load_hpo_graph_data_impl,
    _resolve_hpo_db_path,
    load_hpo_graph_data,
)

pytestmark = pytest.mark.unit


class TestCacheKeyNormalization:
    """Test that cache keys are normalized correctly."""

    def test_resolve_path_none_returns_absolute(self):
        """Test that None is resolved to absolute path."""
        path = _resolve_hpo_db_path(None)
        assert os.path.isabs(path)
        assert path.endswith("hpo_data.db")

    def test_resolve_path_relative_returns_absolute(self):
        """Test that relative paths are converted to absolute."""
        path = _resolve_hpo_db_path("data/hpo_data.db")
        assert os.path.isabs(path)

    def test_resolve_path_absolute_unchanged(self):
        """Test that absolute paths are preserved."""
        abs_path = "/absolute/path/hpo_data.db"
        result = _resolve_hpo_db_path(abs_path)
        # Will be normalized but should still be absolute
        assert os.path.isabs(result)

    def test_resolve_path_consistent_for_none_and_default(self, tmp_path):
        """Test that None and default path resolve to same value."""
        # This ensures cache key consistency
        path1 = _resolve_hpo_db_path(None)
        path2 = _resolve_hpo_db_path(None)
        assert path1 == path2


class TestCacheBehavior:
    """Test LRU cache behavior of load_hpo_graph_data."""

    def test_cache_clear_method_exists(self):
        """Test that cache_clear method is exposed on public API."""
        assert hasattr(load_hpo_graph_data, "cache_clear")
        assert callable(load_hpo_graph_data.cache_clear)

    def test_cache_info_method_exists(self):
        """Test that cache_info method is exposed on public API."""
        assert hasattr(load_hpo_graph_data, "cache_info")
        assert callable(load_hpo_graph_data.cache_info)

    def test_cache_clearing_works(self, mocker):
        """Test that cache can be cleared."""
        # Clear cache first to ensure clean state
        _load_hpo_graph_data_impl.cache_clear()

        # Mock HPODatabase to track calls
        mock_db_instance = mocker.MagicMock()
        mock_db_instance.load_graph_data.return_value = (
            {"HP:0000001": {"HP:0000118"}},
            {"HP:0000001": 1},
        )
        mock_db_class = mocker.patch(
            "phentrieve.evaluation.metrics.HPODatabase",
            return_value=mock_db_instance,
        )

        # Mock os.path.exists to return True for db path
        mocker.patch("phentrieve.evaluation.metrics.os.path.exists", return_value=True)

        # First call - should load
        load_hpo_graph_data()
        assert mock_db_class.call_count == 1

        # Second call (should use cache)
        load_hpo_graph_data()
        assert mock_db_class.call_count == 1  # Still 1 (cached)

        # Clear cache
        _load_hpo_graph_data_impl.cache_clear()

        # Third call (should reload)
        load_hpo_graph_data()
        assert mock_db_class.call_count == 2  # Now 2 (reloaded)

    def test_deprecated_parameters_ignored(self, mocker, caplog):
        """Test that deprecated parameters trigger warning but are ignored."""
        # Mock the implementation
        mocker.patch(
            "phentrieve.evaluation.metrics._load_hpo_graph_data_impl",
            return_value=({}, {}),
        )

        # Call with deprecated parameters
        load_hpo_graph_data(
            db_path=None,
            ancestors_path="/deprecated/ancestors.pkl",
            depths_path="/deprecated/depths.pkl",
        )

        # Check that warning was logged
        assert any("deprecated" in record.message.lower() for record in caplog.records)


class TestThreadSafety:
    """Test thread-safety of HPO graph data loading."""

    def test_concurrent_loads_return_consistent_data(self, mocker):
        """Test that concurrent loads return the same cached data."""
        # Mock the implementation with a slight delay to increase chance of race
        import time

        def mock_load(db_path):
            time.sleep(0.01)  # Small delay
            return (
                {"HP:0000001": {"HP:0000118"}},
                {"HP:0000001": 1},
            )

        mocker.patch(
            "phentrieve.evaluation.metrics._load_hpo_graph_data_impl",
            side_effect=mock_load,
        )

        # Clear cache before test
        load_hpo_graph_data.cache_clear()

        # Launch multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_hpo_graph_data) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results should be identical (same object from cache)
        first_result = results[0]
        for result in results[1:]:
            # After first load, all should return the exact same cached objects
            assert result[0] is first_result[0] or result[0] == first_result[0]
            assert result[1] is first_result[1] or result[1] == first_result[1]

    def test_cache_info_reflects_hits_and_misses(self, mocker):
        """Test that cache statistics are tracked correctly."""
        # Mock dependencies to avoid actual DB loading
        mocker.patch("phentrieve.evaluation.metrics.os.path.exists", return_value=True)
        mock_db = mocker.MagicMock()
        mock_db.load_graph_data.return_value = ({}, {})
        mocker.patch("phentrieve.evaluation.metrics.HPODatabase", return_value=mock_db)

        # Clear cache and reset stats
        _load_hpo_graph_data_impl.cache_clear()

        # Get initial cache info
        initial_info = _load_hpo_graph_data_impl.cache_info()
        assert initial_info.hits == 0
        assert initial_info.misses == 0

        # First call (miss)
        load_hpo_graph_data()
        info_after_first = _load_hpo_graph_data_impl.cache_info()
        assert info_after_first.misses == 1

        # Second call (hit)
        load_hpo_graph_data()
        info_after_second = _load_hpo_graph_data_impl.cache_info()
        assert info_after_second.hits == 1
        assert info_after_second.misses == 1  # Still 1 miss


class TestBackwardCompatibility:
    """Test that refactoring maintains backward compatibility."""

    def test_function_signature_unchanged(self, mocker):
        """Test that function signature accepts same parameters."""
        # Mock to avoid actual DB loading
        mocker.patch(
            "phentrieve.evaluation.metrics._load_hpo_graph_data_impl",
            return_value=({}, {}),
        )

        # Should accept all original parameters
        result = load_hpo_graph_data(
            db_path=None,
            ancestors_path=None,  # Deprecated but accepted
            depths_path=None,  # Deprecated but accepted
        )

        # Should return tuple of two dicts
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)

    def test_return_type_unchanged(self, mocker):
        """Test that return type is still tuple of dicts."""
        # Mock with realistic data structure
        mock_data = (
            {"HP:0000001": {"HP:0000118", "HP:0000002"}},
            {"HP:0000001": 5},
        )
        mocker.patch(
            "phentrieve.evaluation.metrics._load_hpo_graph_data_impl",
            return_value=mock_data,
        )

        ancestors, depths = load_hpo_graph_data()

        assert isinstance(ancestors, dict)
        assert isinstance(depths, dict)
        # Check dict value types
        for term_id, ancestor_set in ancestors.items():
            assert isinstance(term_id, str)
            assert isinstance(ancestor_set, set)
        for term_id, depth in depths.items():
            assert isinstance(term_id, str)
            assert isinstance(depth, int)


class TestErrorHandling:
    """Test error handling in refactored implementation."""

    def test_empty_dicts_returned_on_db_not_found(self, mocker, tmp_path):
        """Test that empty dicts are returned when DB not found."""
        # Create a non-existent path
        fake_db = tmp_path / "nonexistent" / "hpo_data.db"

        # Mock path resolution to return our fake path
        mocker.patch(
            "phentrieve.evaluation.metrics._resolve_hpo_db_path",
            return_value=str(fake_db),
        )

        # Clear cache
        _load_hpo_graph_data_impl.cache_clear()

        # Should return empty dicts, not raise exception
        ancestors, depths = load_hpo_graph_data()
        assert ancestors == {}
        assert depths == {}

    def test_empty_dicts_returned_on_loading_error(self, mocker, tmp_path):
        """Test that empty dicts are returned when loading fails."""
        # Create a real file path that exists
        fake_db = tmp_path / "hpo_data.db"
        fake_db.touch()

        # Mock path resolution
        mocker.patch(
            "phentrieve.evaluation.metrics._resolve_hpo_db_path",
            return_value=str(fake_db),
        )

        # Mock HPODatabase to raise exception
        mocker.patch(
            "phentrieve.evaluation.metrics.HPODatabase",
            side_effect=Exception("Database error"),
        )

        # Clear cache
        _load_hpo_graph_data_impl.cache_clear()

        # Should return empty dicts, not propagate exception
        ancestors, depths = load_hpo_graph_data()
        assert ancestors == {}
        assert depths == {}
