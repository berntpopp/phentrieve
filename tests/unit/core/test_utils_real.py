"""Real unit tests for utils module (actual code execution)."""

import json
from unittest.mock import mock_open, patch

import pytest

from phentrieve.utils import (
    calculate_similarity,
    generate_collection_name,
    get_embedding_dimension,
    get_model_slug,
    load_translation_text,
    normalize_id,
)

pytestmark = pytest.mark.unit


class TestGetEmbeddingDimension:
    """Test get_embedding_dimension function with real logic execution."""

    def test_known_model_distiluse(self):
        """Test dimension for known model: distiluse."""
        result = get_embedding_dimension(
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        assert result == 512

    def test_known_model_bge_m3(self):
        """Test dimension for known model: BGE-M3."""
        result = get_embedding_dimension("BAAI/bge-m3")
        assert result == 1024

    def test_known_model_labse(self):
        """Test dimension for known model: LaBSE."""
        result = get_embedding_dimension("sentence-transformers/LaBSE")
        assert result == 768

    def test_known_model_minilm(self):
        """Test dimension for known model: MiniLM."""
        result = get_embedding_dimension(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert result == 384

    def test_unknown_model_default_dimension(self):
        """Test default dimension (768) for unknown models."""
        result = get_embedding_dimension("unknown/model-name")
        assert result == 768

    def test_empty_string_model(self):
        """Test default dimension for empty string."""
        result = get_embedding_dimension("")
        assert result == 768


class TestGetModelSlug:
    """Test get_model_slug function with real logic execution."""

    def test_none_model_name(self):
        """Test None input returns default slug."""
        result = get_model_slug(None)
        assert result == "biolord_2023_m"

    def test_model_with_slash(self):
        """Test extracting slug from model path with slash."""
        result = get_model_slug("sentence-transformers/all-MiniLM-L6-v2")
        assert result == "all_minilm_l6_v2"

    def test_model_without_slash(self):
        """Test slug generation for model without slash."""
        result = get_model_slug("bert-base-uncased")
        assert result == "bert_base_uncased"

    def test_special_characters_replacement(self):
        """Test that special characters are replaced with underscores."""
        result = get_model_slug("model@name#123")
        assert result == "model_name_123"

    def test_duplicate_underscores_removed(self):
        """Test that duplicate underscores are removed."""
        result = get_model_slug("model___name")
        assert result == "model_name"

    def test_leading_trailing_underscores_removed(self):
        """Test that leading and trailing underscores are removed."""
        result = get_model_slug("_model-name_")
        assert result == "model_name"

    def test_lowercase_conversion(self):
        """Test that slug is converted to lowercase."""
        result = get_model_slug("MyModelName")
        assert result == "mymodelname"

    def test_complex_model_name(self):
        """Test complex real-world model name."""
        result = get_model_slug("FremyCompany/BioLORD-2023-M")
        assert result == "biolord_2023_m"


class TestGenerateCollectionName:
    """Test generate_collection_name function."""

    def test_collection_name_with_simple_model(self):
        """Test collection name generation with simple model."""
        result = generate_collection_name("test-model")
        assert result == "phentrieve_test_model"

    def test_collection_name_with_slash_model(self):
        """Test collection name generation with model containing slash."""
        result = generate_collection_name("org/model-name")
        assert result == "phentrieve_model_name"

    def test_collection_name_prefix(self):
        """Test that collection name has correct prefix."""
        result = generate_collection_name("any-model")
        assert result.startswith("phentrieve_")


class TestCalculateSimilarity:
    """Test calculate_similarity function with real logic execution."""

    def test_perfect_similarity_zero_distance(self):
        """Test perfect similarity (distance = 0)."""
        result = calculate_similarity(0.0)
        assert result == 1.0

    def test_no_similarity_max_distance(self):
        """Test no similarity (distance = 2)."""
        result = calculate_similarity(2.0)
        # 1 - 2 = -1, but clamped to 0.0
        assert result == 0.0

    def test_medium_similarity(self):
        """Test medium similarity."""
        result = calculate_similarity(0.5)
        assert result == 0.5

    def test_high_similarity(self):
        """Test high similarity (small distance)."""
        result = calculate_similarity(0.1)
        assert result == 0.9

    def test_low_similarity(self):
        """Test low similarity (large distance)."""
        result = calculate_similarity(0.9)
        assert abs(result - 0.1) < 1e-10  # Use approximate comparison for floats

    def test_clamping_above_one(self):
        """Test that negative distances are clamped to 1.0."""
        result = calculate_similarity(-0.5)
        # 1 - (-0.5) = 1.5, but clamped to 1.0
        assert result == 1.0

    def test_clamping_below_zero(self):
        """Test that large distances are clamped to 0.0."""
        result = calculate_similarity(3.0)
        # 1 - 3 = -2, but clamped to 0.0
        assert result == 0.0


class TestNormalizeId:
    """Test normalize_id function with real logic execution."""

    def test_uri_format_normalization(self):
        """Test normalization of URI format HPO ID."""
        uri = "http://purl.obolibrary.org/obo/HP_0000001"
        result = normalize_id(uri)
        assert result == "HP:0000001"

    def test_already_normalized_format(self):
        """Test that already normalized IDs are unchanged."""
        normalized = "HP:0000123"
        result = normalize_id(normalized)
        assert result == normalized

    def test_uri_with_different_id(self):
        """Test URI normalization with different HPO ID."""
        uri = "http://purl.obolibrary.org/obo/HP_1234567"
        result = normalize_id(uri)
        assert result == "HP:1234567"

    def test_non_hpo_id_passthrough(self):
        """Test that non-HPO IDs are returned as-is."""
        other_id = "MONDO:0000001"
        result = normalize_id(other_id)
        assert result == other_id

    def test_cached_normalization(self):
        """Test that normalization is cached (same input returns same output)."""
        uri = "http://purl.obolibrary.org/obo/HP_0000001"
        result1 = normalize_id(uri)
        result2 = normalize_id(uri)
        assert result1 == result2


class TestLoadTranslationText:
    """Test load_translation_text function."""

    def test_successful_load_with_synonyms(self):
        """Test successful loading of translation with synonyms."""
        # Arrange
        mock_data = {
            "lbl": "Translated Label",
            "meta": {
                "synonyms": [
                    {"val": "Synonym 1"},
                    {"val": "Synonym 2"},
                    {"val": "Synonym 3"},
                ]
            },
        }
        mock_json = json.dumps(mock_data)

        # Act
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=mock_json)),
        ):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result == "Translated Label. Synonyms: Synonym 1; Synonym 2; Synonym 3"

    def test_successful_load_without_synonyms(self):
        """Test successful loading of translation without synonyms."""
        # Arrange
        mock_data = {"lbl": "Translated Label Only"}
        mock_json = json.dumps(mock_data)

        # Act
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=mock_json)),
        ):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result == "Translated Label Only"

    def test_file_not_found(self):
        """Test when translation file doesn't exist."""
        # Act
        with patch("os.path.exists", return_value=False):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result is None

    def test_missing_label_field(self):
        """Test when label field is missing from translation."""
        # Arrange
        mock_data = {"meta": {"synonyms": [{"val": "Synonym 1"}]}}
        mock_json = json.dumps(mock_data)

        # Act
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=mock_json)),
        ):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result is None

    def test_empty_label_field(self):
        """Test when label field is empty."""
        # Arrange
        mock_data = {"lbl": ""}
        mock_json = json.dumps(mock_data)

        # Act
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=mock_json)),
        ):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result is None

    def test_json_decode_error(self):
        """Test error handling for invalid JSON."""
        # Act
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid json{")),
        ):
            result = load_translation_text("HP:0004241", "/fake/translation/dir")

        # Assert
        assert result is None

    def test_hpo_id_format_conversion(self):
        """Test that HPO ID format is converted correctly for filename."""
        # Arrange
        mock_data = {"lbl": "Test Label"}
        mock_json = json.dumps(mock_data)

        # Act
        with (
            patch("os.path.exists", return_value=True) as mock_exists,
            patch("builtins.open", mock_open(read_data=mock_json)),
        ):
            load_translation_text("HP:0004241", "/fake/dir")

            # Verify the correct path was checked
            expected_path = "/fake/dir/HP_0004241.json"
            mock_exists.assert_called_with(expected_path)
