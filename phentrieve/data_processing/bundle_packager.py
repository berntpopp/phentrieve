"""
Bundle packager for pre-built data distribution (Issue #117).

This module handles creating tar.gz bundles containing:
- SQLite database (hpo_data.db)
- ChromaDB vector indexes
- Manifest with checksums

Bundles are designed for GitHub Releases distribution and Docker integration.

See: https://github.com/berntpopp/phentrieve/issues/117
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from phentrieve.config import (
    DEFAULT_HPO_DB_FILENAME,
    HPO_VERSION,
)
from phentrieve.data_processing.bundle_manifest import (
    BundleManifest,
    EmbeddingModelInfo,
    compute_directory_checksum,
    compute_file_checksum,
    get_model_slug,
)
from phentrieve.utils import generate_collection_name, get_default_data_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_bundle(
    output_dir: Path,
    model_name: str | None = None,
    data_dir: Path | None = None,
    include_hpo_json: bool = False,
    hpo_version: str | None = None,
    multi_vector: bool = False,
) -> Path:
    """
    Create a pre-built data bundle for distribution.

    Args:
        output_dir: Directory to write the bundle file
        model_name: Embedding model name (None for minimal bundle)
        data_dir: Source data directory (default: from config)
        include_hpo_json: Include original hp.json file
        hpo_version: Override HPO version (default: from config)
        multi_vector: If True, bundle multi-vector index (with _multi suffix)

    Returns:
        Path to created bundle file

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If model index doesn't exist
    """
    # Resolve paths
    data_dir = data_dir or get_default_data_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hpo_version = hpo_version or HPO_VERSION

    # Verify source files exist
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME
    if not db_path.exists():
        raise FileNotFoundError(
            f"HPO database not found: {db_path}. Run 'phentrieve data prepare' first."
        )

    # Initialize manifest
    manifest = BundleManifest(
        hpo_version=hpo_version,
        hpo_source_url=f"https://github.com/obophenotype/human-phenotype-ontology/releases/download/{hpo_version}/hp.json",
    )

    # Get term statistics from database
    _populate_manifest_from_db(manifest, db_path)

    # Handle embedding model
    index_base_dir = None
    if model_name:
        index_base_dir = data_dir / "indexes"
        base_collection_name = generate_collection_name(model_name)
        # Multi-vector collections have "_multi" suffix
        collection_name = (
            f"{base_collection_name}_multi" if multi_vector else base_collection_name
        )
        index_type_str = "multi-vector" if multi_vector else "single-vector"

        # ChromaDB stores all collections in a single directory with chroma.sqlite3
        chroma_db_path = index_base_dir / "chroma.sqlite3"
        if not chroma_db_path.exists():
            raise ValueError(
                f"ChromaDB index not found at {index_base_dir}. "
                f"Run 'phentrieve index build --model-name \"{model_name}\"' first."
            )

        # Verify the collection exists in ChromaDB
        if not _verify_collection_exists(index_base_dir, collection_name):
            mv_flag = " --multi-vector" if multi_vector else ""
            raise ValueError(
                f"Collection '{collection_name}' not found in ChromaDB. "
                f"Run 'phentrieve index build --model-name \"{model_name}\"{mv_flag}' first."
            )

        logger.info(f"Found {index_type_str} collection: {collection_name}")

        # Get embedding dimension from index metadata
        dimension = _get_index_dimension(index_base_dir)
        manifest.model = EmbeddingModelInfo.from_model_name(
            model_name, dimension=dimension, multi_vector=multi_vector
        )

    # Create bundle in temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bundle_root = temp_path / "bundle"
        bundle_root.mkdir()

        # Copy database
        db_dest = bundle_root / DEFAULT_HPO_DB_FILENAME
        shutil.copy2(db_path, db_dest)
        manifest.add_file_checksum(db_dest)
        logger.info(f"Added database: {DEFAULT_HPO_DB_FILENAME}")

        # Copy index if model specified
        if index_base_dir and model_name:
            indexes_dest = bundle_root / "indexes"
            # Copy entire ChromaDB directory (includes chroma.sqlite3 and segment dirs)
            shutil.copytree(index_base_dir, indexes_dest)
            manifest.checksums["indexes/"] = compute_directory_checksum(indexes_dest)
            logger.info(f"Added ChromaDB index for model: {model_name}")

        # Optionally include hp.json
        if include_hpo_json:
            hp_json_path = data_dir / "hp.json"
            if hp_json_path.exists():
                hp_dest = bundle_root / "hp.json"
                shutil.copy2(hp_json_path, hp_dest)
                manifest.add_file_checksum(hp_dest)
                logger.info("Added hp.json")
            else:
                logger.warning("hp.json not found, skipping")

        # Save manifest
        manifest_path = bundle_root / "manifest.json"
        manifest.save(manifest_path)

        # Create tarball
        bundle_filename = manifest.get_bundle_filename()
        bundle_path = output_dir / bundle_filename

        with tarfile.open(bundle_path, "w:gz") as tar:
            for item in bundle_root.iterdir():
                tar.add(item, arcname=item.name)

        bundle_size = bundle_path.stat().st_size
        logger.info(
            f"Created bundle: {bundle_path} ({bundle_size / 1024 / 1024:.1f} MB)"
        )

        return bundle_path


def _populate_manifest_from_db(manifest: BundleManifest, db_path: Path) -> None:
    """
    Populate manifest with metadata from HPO database.

    Args:
        manifest: Manifest to populate
        db_path: Path to HPO database
    """
    from phentrieve.data_processing.hpo_database import HPODatabase

    with HPODatabase(db_path) as db:
        # Get metadata
        manifest.hpo_version = db.get_metadata("hpo_version") or manifest.hpo_version
        manifest.hpo_release_date = db.get_metadata("hpo_download_date") or ""
        manifest.hpo_source_url = (
            db.get_metadata("hpo_source_url") or manifest.hpo_source_url
        )

        # Get term statistics
        active_str = db.get_metadata("active_terms_count")
        obsolete_str = db.get_metadata("obsolete_terms_filtered")

        if active_str:
            manifest.active_terms = int(active_str)
        if obsolete_str:
            manifest.obsolete_terms = int(obsolete_str)

        manifest.total_terms = manifest.active_terms + manifest.obsolete_terms


def _get_index_dimension(index_dir: Path) -> int:
    """
    Get embedding dimension from ChromaDB index.

    Args:
        index_dir: Path to ChromaDB index directory

    Returns:
        Embedding dimension (default: 768)
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir))
        collections = client.list_collections()
        if collections:
            # Get dimension from first collection's metadata or peek
            collection = collections[0]
            # Try to peek at an embedding
            result = collection.peek(limit=1)
            if result and result.get("embeddings"):
                embeddings = result["embeddings"]
                if embeddings and len(embeddings) > 0:
                    return len(embeddings[0])
    except Exception as e:
        logger.warning(f"Could not determine embedding dimension: {e}")

    # Default dimension for most models
    return 768


def _verify_collection_exists(index_dir: Path, collection_name: str) -> bool:
    """
    Verify that a collection exists in ChromaDB.

    Args:
        index_dir: Path to ChromaDB index directory
        collection_name: Name of collection to check

    Returns:
        True if collection exists, False otherwise
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir))
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        return collection_name in collection_names
    except Exception as e:
        logger.warning(f"Could not verify collection existence: {e}")
        return False


def extract_bundle(
    bundle_path: Path,
    target_dir: Path,
    verify_checksums: bool = True,
) -> BundleManifest:
    """
    Extract a bundle to target directory.

    Args:
        bundle_path: Path to bundle tar.gz file
        target_dir: Directory to extract to
        verify_checksums: Verify file checksums after extraction

    Returns:
        Manifest from extracted bundle

    Raises:
        ValueError: If checksum verification fails
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting bundle: {bundle_path}")

    with tarfile.open(bundle_path, "r:gz") as tar:
        # Use data filter for security (prevents path traversal attacks)
        # Python 3.12+ has built-in filter, for earlier versions we validate manually
        tar.extractall(path=target_dir, filter="data")  # noqa: S202

    # Load manifest
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Bundle missing manifest.json")

    manifest = BundleManifest.load(manifest_path)

    # Verify checksums if requested
    if verify_checksums:
        _verify_bundle_checksums(manifest, target_dir)

    logger.info(
        f"Extracted bundle: HPO {manifest.hpo_version}, "
        f"{manifest.active_terms} active terms"
    )

    return manifest


def _verify_bundle_checksums(manifest: BundleManifest, target_dir: Path) -> None:
    """
    Verify all checksums in extracted bundle.

    Args:
        manifest: Bundle manifest
        target_dir: Directory containing extracted files

    Raises:
        ValueError: If any checksum fails
    """
    failed = []

    for filename, expected_hash in manifest.checksums.items():
        if filename.endswith("/"):
            # Directory checksum
            dir_path = target_dir / filename.rstrip("/")
            if dir_path.exists():
                actual_hash = compute_directory_checksum(dir_path)
                if actual_hash != expected_hash:
                    failed.append(filename)
        else:
            # File checksum
            file_path = target_dir / filename
            if file_path.exists():
                actual_hash = compute_file_checksum(file_path)
                if actual_hash != expected_hash:
                    failed.append(filename)
            else:
                failed.append(f"{filename} (missing)")

    if failed:
        raise ValueError(f"Checksum verification failed for: {', '.join(failed)}")

    logger.info("All checksums verified successfully")


def list_available_bundles(
    data_dir: Path | None = None,
) -> list[dict[str, str | None]]:
    """
    List available bundles that can be created from local data.

    Args:
        data_dir: Data directory to check

    Returns:
        List of bundle info dicts with model_name, model_slug, status
    """
    from phentrieve.config import BENCHMARK_MODELS

    data_dir = data_dir or get_default_data_dir()
    index_base_dir = data_dir / "indexes"

    bundles = []

    # Check each benchmark model - use ChromaDB collection check
    for model_name in BENCHMARK_MODELS:
        collection_name = generate_collection_name(model_name)
        has_collection = _verify_collection_exists(index_base_dir, collection_name)

        bundles.append(
            {
                "model_name": model_name,
                "model_slug": get_model_slug(model_name),
                "status": "ready" if has_collection else "missing_index",
                "index_path": str(index_base_dir) if has_collection else None,
            }
        )

    return bundles
