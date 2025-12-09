"""Unit tests for bundle packager module.

Tests for bundle creation and extraction functions from bundle_packager.py.
Follows best practices with clear Arrange-Act-Assert structure.

Issue #117: Pre-built data distribution system.
"""

import tarfile
from unittest.mock import MagicMock, patch

import pytest

from phentrieve.data_processing.bundle_manifest import BundleManifest
from phentrieve.data_processing.bundle_packager import (
    _get_index_dimension,
    _populate_manifest_from_db,
    _verify_bundle_checksums,
    create_bundle,
    extract_bundle,
    list_available_bundles,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for create_bundle()
# =============================================================================


class TestCreateBundle:
    """Tests for create_bundle() function."""

    def test_raises_error_when_database_missing(self, tmp_path):
        """Test that FileNotFoundError is raised when database is missing."""
        # Arrange
        output_dir = tmp_path / "output"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            create_bundle(
                output_dir=output_dir,
                model_name=None,
                data_dir=data_dir,
            )

        assert "hpo_data.db" in str(exc_info.value)

    def test_raises_error_when_index_missing(self, tmp_path):
        """Test that ValueError is raised when model index doesn't exist."""
        # Arrange
        output_dir = tmp_path / "output"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create mock database
        db_path = data_dir / "hpo_data.db"
        db_path.touch()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            with patch(
                "phentrieve.data_processing.bundle_packager._populate_manifest_from_db"
            ):
                create_bundle(
                    output_dir=output_dir,
                    model_name="FremyCompany/BioLORD-2023-M",
                    data_dir=data_dir,
                )

        assert "ChromaDB index not found" in str(exc_info.value)


# =============================================================================
# Tests for extract_bundle()
# =============================================================================


class TestExtractBundle:
    """Tests for extract_bundle() function."""

    def test_extracts_bundle_with_manifest(self, tmp_path):
        """Test bundle extraction with manifest verification."""
        # Arrange
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()

        # Create mock files
        manifest = BundleManifest(hpo_version="v2025-03-03", active_terms=17000)
        manifest_path = bundle_dir / "manifest.json"
        manifest.save(manifest_path)

        db_file = bundle_dir / "hpo_data.db"
        db_file.write_text("mock database")
        manifest.add_file_checksum(db_file)
        manifest.save(manifest_path)  # Save with checksum

        # Create tarball
        bundle_path = tmp_path / "test-bundle.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(manifest_path, arcname="manifest.json")
            tar.add(db_file, arcname="hpo_data.db")

        # Extract directory
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Act
        result = extract_bundle(
            bundle_path=bundle_path,
            target_dir=extract_dir,
            verify_checksums=True,
        )

        # Assert
        assert result.hpo_version == "v2025-03-03"
        assert result.active_terms == 17000
        assert (extract_dir / "manifest.json").exists()
        assert (extract_dir / "hpo_data.db").exists()

    def test_raises_error_when_manifest_missing(self, tmp_path):
        """Test that ValueError is raised when manifest is missing."""
        # Arrange
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()

        # Create tarball without manifest
        db_file = bundle_dir / "hpo_data.db"
        db_file.write_text("mock database")

        bundle_path = tmp_path / "test-bundle.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(db_file, arcname="hpo_data.db")

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            extract_bundle(bundle_path=bundle_path, target_dir=extract_dir)

        assert "manifest.json" in str(exc_info.value)


# =============================================================================
# Tests for _verify_bundle_checksums()
# =============================================================================


class TestVerifyBundleChecksums:
    """Tests for _verify_bundle_checksums() function."""

    def test_passes_when_checksums_match(self, tmp_path):
        """Test verification passes when checksums match."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        manifest = BundleManifest()
        manifest.add_file_checksum(test_file)

        # Act & Assert - should not raise
        _verify_bundle_checksums(manifest, tmp_path)

    def test_raises_when_checksum_mismatch(self, tmp_path):
        """Test verification raises when checksum doesn't match."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        manifest = BundleManifest()
        manifest.add_file_checksum(test_file)

        # Modify file after checksumming
        test_file.write_text("modified content")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            _verify_bundle_checksums(manifest, tmp_path)

        assert "test.txt" in str(exc_info.value)

    def test_raises_when_file_missing(self, tmp_path):
        """Test verification raises when file is missing."""
        # Arrange
        manifest = BundleManifest()
        manifest.checksums["missing.txt"] = "fake_checksum"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            _verify_bundle_checksums(manifest, tmp_path)

        assert "missing" in str(exc_info.value)


# =============================================================================
# Tests for _get_index_dimension()
# =============================================================================


class TestGetIndexDimension:
    """Tests for _get_index_dimension() function."""

    def test_returns_default_on_error(self, tmp_path):
        """Test that default dimension is returned on error."""
        # Arrange
        index_dir = tmp_path / "nonexistent"

        # Act
        dimension = _get_index_dimension(index_dir)

        # Assert
        assert dimension == 768  # Default

    def test_returns_default_when_no_collections(self, tmp_path):
        """Test that default is returned when no collections exist."""
        # Arrange
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_client_class.return_value = mock_client

            # Act
            dimension = _get_index_dimension(index_dir)

        # Assert
        assert dimension == 768


# =============================================================================
# Tests for list_available_bundles()
# =============================================================================


class TestListAvailableBundles:
    """Tests for list_available_bundles() function."""

    def test_returns_ready_for_models_with_indexes(self, tmp_path):
        """Test that models with indexes are marked as ready."""
        # Arrange
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create database
        (data_dir / "hpo_data.db").touch()

        # Create indexes directory
        index_dir = data_dir / "indexes"
        index_dir.mkdir(parents=True)

        # Mock _verify_collection_exists to return True for BioLORD collection
        with patch(
            "phentrieve.data_processing.bundle_packager._verify_collection_exists",
            side_effect=lambda path, name: name == "phentrieve_biolord_2023_m",
        ):
            # Act
            bundles = list_available_bundles(data_dir=data_dir)

        # Assert
        biolord = next(b for b in bundles if b["model_slug"] == "biolord")
        assert biolord["status"] == "ready"

    def test_returns_missing_index_for_models_without_indexes(self, tmp_path):
        """Test that models without indexes are marked as missing_index."""
        # Arrange
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create database
        (data_dir / "hpo_data.db").touch()

        # Mock _verify_collection_exists to return False for all collections
        with patch(
            "phentrieve.data_processing.bundle_packager._verify_collection_exists",
            return_value=False,
        ):
            # Act
            bundles = list_available_bundles(data_dir=data_dir)

        # Assert
        biolord = next(b for b in bundles if b["model_slug"] == "biolord")
        assert biolord["status"] == "missing_index"


# =============================================================================
# Tests for _populate_manifest_from_db()
# =============================================================================


class TestPopulateManifestFromDb:
    """Tests for _populate_manifest_from_db() function."""

    def test_populates_manifest_with_metadata(self, tmp_path):
        """Test that manifest is populated with database metadata."""
        # Arrange
        manifest = BundleManifest()
        db_path = tmp_path / "test.db"

        # Mock HPODatabase
        with patch(
            "phentrieve.data_processing.hpo_database.HPODatabase"
        ) as MockHPODatabase:
            mock_db = MagicMock()
            mock_db.get_metadata.side_effect = lambda key: {
                "hpo_version": "v2025-03-03",
                "hpo_download_date": "2025-03-03",
                "hpo_source_url": "https://example.com/hp.json",
                "active_terms_count": "17000",
                "obsolete_terms_filtered": "2000",
            }.get(key)
            MockHPODatabase.return_value = mock_db

            # Act
            _populate_manifest_from_db(manifest, db_path)

        # Assert
        assert manifest.hpo_version == "v2025-03-03"
        assert manifest.active_terms == 17000
        assert manifest.obsolete_terms == 2000
        assert manifest.total_terms == 19000

    def test_handles_missing_metadata(self, tmp_path):
        """Test handling when metadata is missing."""
        # Arrange
        manifest = BundleManifest(hpo_version="default")
        db_path = tmp_path / "test.db"

        # Mock HPODatabase with missing values
        with patch(
            "phentrieve.data_processing.hpo_database.HPODatabase"
        ) as MockHPODatabase:
            mock_db = MagicMock()
            mock_db.get_metadata.return_value = None
            MockHPODatabase.return_value = mock_db

            # Act
            _populate_manifest_from_db(manifest, db_path)

        # Assert
        assert manifest.hpo_version == "default"  # Unchanged
        assert manifest.active_terms == 0
        assert manifest.obsolete_terms == 0
