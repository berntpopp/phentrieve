"""Unit tests for bundle manifest module.

Tests for BundleManifest and related functions from bundle_manifest.py.
Follows best practices with clear Arrange-Act-Assert structure.

Issue #117: Pre-built data distribution system.
"""

import pytest

from phentrieve.data_processing.bundle_manifest import (
    MANIFEST_VERSION,
    MODEL_SLUGS,
    SLUG_TO_MODEL,
    BundleManifest,
    EmbeddingModelInfo,
    compute_directory_checksum,
    compute_file_checksum,
    get_model_name,
    get_model_slug,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for get_model_slug()
# =============================================================================


class TestGetModelSlug:
    """Tests for get_model_slug() function."""

    def test_returns_known_slug_for_biolord(self):
        """Test that BioLORD returns 'biolord' slug."""
        # Act
        slug = get_model_slug("FremyCompany/BioLORD-2023-M")

        # Assert
        assert slug == "biolord"

    def test_returns_known_slug_for_bge_m3(self):
        """Test that BGE-M3 returns 'bge-m3' slug."""
        # Act
        slug = get_model_slug("BAAI/bge-m3")

        # Assert
        assert slug == "bge-m3"

    def test_sanitizes_unknown_model_name(self):
        """Test that unknown model names are sanitized."""
        # Act
        slug = get_model_slug("custom/My-Model-v1")

        # Assert
        assert slug == "custom_my_model_v1"

    def test_handles_model_without_slash(self):
        """Test handling of model names without organization prefix."""
        # Act
        slug = get_model_slug("LocalModel")

        # Assert
        assert slug == "localmodel"


class TestGetModelName:
    """Tests for get_model_name() function."""

    def test_returns_model_for_known_slug(self):
        """Test that known slugs return full model names."""
        # Act
        model = get_model_name("biolord")

        # Assert
        assert model == "FremyCompany/BioLORD-2023-M"

    def test_returns_none_for_unknown_slug(self):
        """Test that unknown slugs return None."""
        # Act
        model = get_model_name("unknown-slug")

        # Assert
        assert model is None


# =============================================================================
# Tests for EmbeddingModelInfo
# =============================================================================


class TestEmbeddingModelInfo:
    """Tests for EmbeddingModelInfo dataclass."""

    def test_creates_from_model_name(self):
        """Test creation from model name."""
        # Act
        info = EmbeddingModelInfo.from_model_name(
            "FremyCompany/BioLORD-2023-M", dimension=768
        )

        # Assert
        assert info.name == "FremyCompany/BioLORD-2023-M"
        assert info.slug == "biolord"
        assert info.dimension == 768
        assert info.distance_metric == "cosine"

    def test_accepts_custom_distance_metric(self):
        """Test that custom distance metric is accepted."""
        # Act
        info = EmbeddingModelInfo.from_model_name(
            "BAAI/bge-m3", dimension=1024, distance_metric="l2"
        )

        # Assert
        assert info.distance_metric == "l2"


# =============================================================================
# Tests for BundleManifest
# =============================================================================


class TestBundleManifest:
    """Tests for BundleManifest dataclass."""

    def test_creates_with_defaults(self):
        """Test manifest creation with default values."""
        # Act
        manifest = BundleManifest()

        # Assert
        assert manifest.manifest_version == MANIFEST_VERSION
        assert manifest.bundle_format == "tar.gz"
        assert manifest.active_terms == 0
        assert manifest.model is None
        assert manifest.created_at  # Should have a timestamp

    def test_creates_with_hpo_data(self):
        """Test manifest creation with HPO metadata."""
        # Arrange
        manifest = BundleManifest(
            hpo_version="v2025-03-03",
            active_terms=17000,
            obsolete_terms=2000,
        )

        # Assert
        assert manifest.hpo_version == "v2025-03-03"
        assert manifest.active_terms == 17000
        assert manifest.obsolete_terms == 2000
        assert manifest.total_terms == 0  # Not auto-calculated in __init__

    def test_to_dict_without_model(self):
        """Test converting manifest without model to dict."""
        # Arrange
        manifest = BundleManifest(hpo_version="v2025-03-03", active_terms=17000)

        # Act
        data = manifest.to_dict()

        # Assert
        assert data["hpo_version"] == "v2025-03-03"
        assert data["active_terms"] == 17000
        assert data["model"] is None

    def test_to_dict_with_model(self):
        """Test converting manifest with model to dict."""
        # Arrange
        model = EmbeddingModelInfo.from_model_name(
            "FremyCompany/BioLORD-2023-M", dimension=768
        )
        manifest = BundleManifest(hpo_version="v2025-03-03", model=model)

        # Act
        data = manifest.to_dict()

        # Assert
        assert data["model"]["name"] == "FremyCompany/BioLORD-2023-M"
        assert data["model"]["slug"] == "biolord"
        assert data["model"]["dimension"] == 768

    def test_to_json_and_from_json(self):
        """Test JSON serialization roundtrip."""
        # Arrange
        model = EmbeddingModelInfo.from_model_name(
            "FremyCompany/BioLORD-2023-M", dimension=768
        )
        original = BundleManifest(
            hpo_version="v2025-03-03",
            active_terms=17000,
            obsolete_terms=2000,
            model=model,
        )

        # Act
        json_str = original.to_json()
        restored = BundleManifest.from_json(json_str)

        # Assert
        assert restored.hpo_version == original.hpo_version
        assert restored.active_terms == original.active_terms
        assert restored.model.name == original.model.name
        assert restored.model.dimension == original.model.dimension

    def test_save_and_load(self, tmp_path):
        """Test saving and loading manifest from file."""
        # Arrange
        manifest = BundleManifest(hpo_version="v2025-03-03", active_terms=17000)
        manifest_path = tmp_path / "manifest.json"

        # Act
        manifest.save(manifest_path)
        loaded = BundleManifest.load(manifest_path)

        # Assert
        assert loaded.hpo_version == manifest.hpo_version
        assert loaded.active_terms == manifest.active_terms

    def test_get_bundle_filename_raises_without_model(self):
        """Test bundle filename raises ValueError when model is missing."""
        # Arrange
        manifest = BundleManifest(hpo_version="v2025-03-03")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            manifest.get_bundle_filename()

        assert "requires a model" in str(exc_info.value)

    def test_get_bundle_filename_with_model(self):
        """Test bundle filename generation with model."""
        # Arrange
        model = EmbeddingModelInfo.from_model_name(
            "FremyCompany/BioLORD-2023-M", dimension=768
        )
        manifest = BundleManifest(hpo_version="v2025-03-03", model=model)

        # Act
        filename = manifest.get_bundle_filename()

        # Assert
        assert filename == "phentrieve-data-v2025-03-03-biolord.tar.gz"

    def test_add_file_checksum(self, tmp_path):
        """Test adding file checksum to manifest."""
        # Arrange
        manifest = BundleManifest()
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Act
        checksum = manifest.add_file_checksum(test_file)

        # Assert
        assert checksum in manifest.checksums.values()
        assert "test.txt" in manifest.checksums

    def test_verify_checksum_success(self, tmp_path):
        """Test successful checksum verification."""
        # Arrange
        manifest = BundleManifest()
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        manifest.add_file_checksum(test_file)

        # Act
        result = manifest.verify_checksum(test_file)

        # Assert
        assert result is True

    def test_verify_checksum_failure(self, tmp_path):
        """Test checksum verification failure."""
        # Arrange
        manifest = BundleManifest()
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        manifest.add_file_checksum(test_file)

        # Modify file
        test_file.write_text("modified content")

        # Act
        result = manifest.verify_checksum(test_file)

        # Assert
        assert result is False


# =============================================================================
# Tests for checksum functions
# =============================================================================


class TestChecksumFunctions:
    """Tests for checksum computation functions."""

    def test_compute_file_checksum(self, tmp_path):
        """Test file checksum computation."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Act
        checksum = compute_file_checksum(test_file)

        # Assert
        assert len(checksum) == 64  # SHA-256 hex digest length
        assert checksum.isalnum()

    def test_compute_file_checksum_deterministic(self, tmp_path):
        """Test that checksum is deterministic."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Act
        checksum1 = compute_file_checksum(test_file)
        checksum2 = compute_file_checksum(test_file)

        # Assert
        assert checksum1 == checksum2

    def test_compute_directory_checksum(self, tmp_path):
        """Test directory checksum computation."""
        # Arrange
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")

        # Act
        checksum = compute_directory_checksum(subdir)

        # Assert
        assert len(checksum) == 64
        assert checksum.isalnum()

    def test_compute_directory_checksum_deterministic(self, tmp_path):
        """Test that directory checksum is deterministic."""
        # Arrange
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")

        # Act
        checksum1 = compute_directory_checksum(subdir)
        checksum2 = compute_directory_checksum(subdir)

        # Assert
        assert checksum1 == checksum2

    def test_compute_directory_checksum_with_exclusion(self, tmp_path):
        """Test directory checksum with file exclusion."""
        # Arrange
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file1.txt").write_text("content1")
        (subdir / "excluded.txt").write_text("excluded")

        # Act
        checksum_without_exclude = compute_directory_checksum(subdir)
        checksum_with_exclude = compute_directory_checksum(
            subdir, exclude={"excluded.txt"}
        )

        # Assert
        assert checksum_without_exclude != checksum_with_exclude


# =============================================================================
# Tests for MODEL_SLUGS mapping
# =============================================================================


class TestModelSlugsMapping:
    """Tests for MODEL_SLUGS and SLUG_TO_MODEL mappings."""

    def test_all_benchmark_models_have_slugs(self):
        """Test that all benchmark models have slug mappings."""
        benchmark_models = [
            "FremyCompany/BioLORD-2023-M",
            "BAAI/bge-m3",
            "sentence-transformers/LaBSE",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ]

        for model in benchmark_models:
            assert model in MODEL_SLUGS, f"Missing slug for {model}"

    def test_reverse_mapping_is_complete(self):
        """Test that SLUG_TO_MODEL is complete reverse of MODEL_SLUGS."""
        # Assert
        assert len(MODEL_SLUGS) == len(SLUG_TO_MODEL)

        for model, slug in MODEL_SLUGS.items():
            assert SLUG_TO_MODEL[slug] == model

    def test_slugs_are_url_safe(self):
        """Test that all slugs are URL-safe."""
        import re

        url_safe_pattern = re.compile(r"^[a-z0-9_-]+$")

        for slug in MODEL_SLUGS.values():
            assert url_safe_pattern.match(slug), f"Slug '{slug}' is not URL-safe"
