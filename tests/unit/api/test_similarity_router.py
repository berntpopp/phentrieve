"""Unit tests for similarity_router helper functions.

Following established testing pattern:
- Test helper functions directly (not endpoints)
- Mock only external dependencies
- Use Arrange-Act-Assert pattern
- Focus on edge cases and error handling
"""

import pytest

from api.routers.similarity_router import _get_hpo_label_map_api

pytestmark = pytest.mark.unit


class TestGetHPOLabelMapAPI:
    """Test _get_hpo_label_map_api helper function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear LRU cache before each test to prevent test pollution."""
        _get_hpo_label_map_api.cache_clear()
        yield
        _get_hpo_label_map_api.cache_clear()

    def test_returns_dict_mapping_ids_to_labels(self, mocker):
        """Test returns dictionary with HPO ID to label mapping."""
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure"},
                {"id": "HP:0000002", "label": "Tremor"},
                {"id": "HP:0000003", "label": "Ataxia"},
            ],
        )

        # Act
        result = _get_hpo_label_map_api()

        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["HP:0000001"] == "Seizure"
        assert result["HP:0000002"] == "Tremor"
        assert result["HP:0000003"] == "Ataxia"

    def test_handles_empty_hpo_terms_data(self, mocker):
        """Test handles empty HPO terms data gracefully."""
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[],
        )

        # Act
        result = _get_hpo_label_map_api()

        # Assert
        assert result == {}

    def test_handles_none_hpo_terms_data(self, mocker):
        """Test handles None HPO terms data gracefully."""
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=None,
        )

        # Act
        result = _get_hpo_label_map_api()

        # Assert
        assert result == {}

    def test_skips_terms_missing_id_or_label(self, mocker):
        """Test skips terms that don't have both id and label."""
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure"},  # Valid
                {"id": "HP:0000002"},  # Missing label
                {"label": "Tremor"},  # Missing id
                {"id": "", "label": "Ataxia"},  # Empty id
                {"id": "HP:0000003", "label": ""},  # Empty label
                {"id": "HP:0000004", "label": "Dystonia"},  # Valid
            ],
        )

        # Act
        result = _get_hpo_label_map_api()

        # Assert
        assert len(result) == 2  # Only 2 valid terms
        assert result["HP:0000001"] == "Seizure"
        assert result["HP:0000004"] == "Dystonia"
        assert "HP:0000002" not in result
        assert "HP:0000003" not in result

    def test_result_is_cached_via_lru_cache(self, mocker):
        """Test function uses LRU cache (should only load once)."""
        # Arrange
        mock_load = mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure"},
            ],
        )

        # Act - Call multiple times (cache already cleared by fixture)
        result1 = _get_hpo_label_map_api()
        result2 = _get_hpo_label_map_api()
        result3 = _get_hpo_label_map_api()

        # Assert - load_hpo_terms should only be called once due to caching
        assert result1 == result2 == result3
        assert mock_load.call_count == 1  # Only called once!

    def test_handles_malformed_term_data_partially(self, mocker):
        """Test handles some malformed term data but not all types.

        NOTE: Current implementation will crash on non-dict items (e.g., strings, None).
        This test documents that behavior for valid dictionary-type items only.
        """
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure"},  # Valid
                {},  # Empty dict - handled OK
                {"other_field": "value"},  # Different structure - handled OK
                {"id": "HP:0000002", "label": "Tremor"},  # Valid
            ],
        )

        # Act
        result = _get_hpo_label_map_api()

        # Assert - Should only extract valid terms with both id and label
        assert len(result) == 2
        assert result["HP:0000001"] == "Seizure"
        assert result["HP:0000002"] == "Tremor"

    def test_preserves_label_with_special_characters(self, mocker):
        """Test preserves labels with special characters correctly."""
        # Arrange
        mocker.patch(
            "api.routers.similarity_router.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure (epileptic)"},
                {"id": "HP:0000002", "label": "Tremor - resting"},
                {"id": "HP:0000003", "label": "Ataxia/Dystonia"},
                {"id": "HP:0000004", "label": "Complex symptom [rare]"},
            ],
        )

        # Act (cache already cleared by fixture)
        result = _get_hpo_label_map_api()

        # Assert - Special characters preserved
        assert result["HP:0000001"] == "Seizure (epileptic)"
        assert result["HP:0000002"] == "Tremor - resting"
        assert result["HP:0000003"] == "Ataxia/Dystonia"
        assert result["HP:0000004"] == "Complex symptom [rare]"
