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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from phentrieve.config import (
    DEFAULT_HPO_DB_FILENAME,
    HPO_VERSION,
    VectorStoreConfig,
)
from phentrieve.data_processing.bundle_manifest import (
    BundleManifest,
    EmbeddingModelInfo,
    compute_directory_checksum,
    compute_file_checksum,
    get_model_slug,
)
from phentrieve.data_processing.release_contract import DataReleaseSpec
from phentrieve.utils import generate_collection_name, get_default_data_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _collection_name(collection: object) -> str:
    """Return a Chroma collection name across ChromaDB 0.6 and 1.x APIs."""
    if isinstance(collection, str):
        return collection
    return str(cast(Any, collection).name)


def _collection_from_list_entry(client: object, collection: object) -> object:
    """Resolve ChromaDB 0.6 name-list entries to collection objects."""
    if isinstance(collection, str):
        return cast(Any, client).get_collection(collection)
    return collection


def create_bundle(
    output_dir: Path,
    model_name: str | None = None,
    data_dir: Path | None = None,
    include_hpo_json: bool = False,
    hpo_version: str | None = None,
    multi_vector: bool = False,
    release_spec: DataReleaseSpec | None = None,
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
        release_spec: Immutable release contract required for a published data build.

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
    if release_spec is not None:
        _apply_release_spec_provenance(manifest, release_spec)

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

        model_revision = _expected_model_revision(
            model_name=model_name,
            release_spec=release_spec,
        )
        model_trust_remote_code = _expected_model_trust_remote_code(
            model_name=model_name,
            release_spec=release_spec,
        )
        with _open_collection(index_base_dir, collection_name) as collection:
            expected_document_count = _expected_document_count(
                collection=collection,
                manifest=manifest,
                multi_vector=multi_vector,
                release_spec=release_spec,
            )
            dimension = _validate_collection_provenance(
                collection=collection,
                manifest=manifest,
                model_name=model_name,
                index_type="multi_vector" if multi_vector else "single_vector",
                expected_document_count=expected_document_count,
                expected_model_revision=model_revision,
            )
            resolved_model_revision = str(
                cast(Any, collection).metadata.get("model_revision", "")
            )
        logger.info(f"Found validated {index_type_str} collection: {collection_name}")

        manifest.model = EmbeddingModelInfo.from_model_name(
            model_name,
            dimension=dimension,
            multi_vector=multi_vector,
            revision=resolved_model_revision,
            trust_remote_code=model_trust_remote_code,
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
        manifest.hpo_release_date = db.get_metadata("hpo_release_date") or ""
        manifest.hpo_source_url = (
            db.get_metadata("hpo_source_url") or manifest.hpo_source_url
        )
        manifest.hpo_source_sha256 = db.get_metadata("hpo_source_sha256") or ""

        # Get term statistics
        active_str = db.get_metadata("active_terms_count")
        obsolete_str = db.get_metadata("obsolete_terms_filtered")

        if active_str:
            manifest.active_terms = int(active_str)
        if obsolete_str:
            manifest.obsolete_terms = int(obsolete_str)

        manifest.total_terms = manifest.active_terms + manifest.obsolete_terms


def _apply_release_spec_provenance(
    manifest: BundleManifest, release_spec: DataReleaseSpec
) -> None:
    """Require database provenance to match an immutable data-release specification."""
    for field_name, actual, expected in (
        ("HPO version", manifest.hpo_version, release_spec.hpo_version),
        ("HPO release date", manifest.hpo_release_date, release_spec.hpo_release_date),
        ("HPO source URL", manifest.hpo_source_url, release_spec.hpo_source_url),
        ("HPO source SHA-256", manifest.hpo_source_sha256, release_spec.hpo_sha256),
        ("active term count", manifest.active_terms, release_spec.active_terms),
    ):
        if actual != expected:
            raise ValueError(
                f"Database {field_name} does not match release spec: "
                f"expected {expected!r}, got {actual!r}"
            )

    manifest.source_commit = release_spec.source_commit
    manifest.lockfile_sha256 = release_spec.lockfile_sha256
    manifest.phentrieve_version = release_spec.phentrieve_version


def _expected_model_revision(
    model_name: str, release_spec: DataReleaseSpec | None
) -> str | None:
    """Return the required model revision for a release build."""
    if release_spec is None:
        return None
    for model in release_spec.models:
        if model.name == model_name:
            return model.revision
    raise ValueError(f"Model {model_name!r} is not declared in the release spec")


def _expected_model_trust_remote_code(
    model_name: str,
    release_spec: DataReleaseSpec | None,
) -> bool:
    """Return the custom-code policy pinned for a release model."""
    if release_spec is None:
        return False
    for model in release_spec.models:
        if model.name == model_name:
            return model.trust_remote_code
    raise ValueError(
        f"Model {model_name!r} is not present in the release specification"
    )


def _expected_document_count(
    collection: object,
    manifest: BundleManifest,
    multi_vector: bool,
    release_spec: DataReleaseSpec | None,
) -> int:
    """Resolve the expected count from the release spec or validated collection metadata."""
    index_type = "multi_vector" if multi_vector else "single_vector"
    if release_spec is not None:
        return release_spec.expected_document_count(index_type)
    if not multi_vector:
        return manifest.active_terms

    metadata = cast(Any, collection).metadata or {}
    raw_count = metadata.get("expected_document_count")
    if not isinstance(raw_count, int) or raw_count <= 0:
        raise ValueError(
            "Multi-vector collection is missing a valid expected document count"
        )
    return raw_count


@contextmanager
def _open_collection(index_dir: Path, collection_name: str) -> Iterator[object]:
    """Keep the owning Chroma client alive while using its collection."""
    import chromadb

    vector_store_config = VectorStoreConfig(
        path=str(index_dir),
        collection_name=collection_name,
    )
    client: Any = chromadb.PersistentClient(
        path=str(index_dir),
        settings=vector_store_config.to_chromadb_settings(),
    )
    try:
        yield cast(object, client.get_collection(collection_name))
    finally:
        client.close()


def _validate_collection_provenance(
    collection: object,
    manifest: BundleManifest,
    model_name: str,
    index_type: str,
    expected_document_count: int,
    expected_model_revision: str | None = None,
) -> int:
    """Reject collections that cannot be proven to match the bundle inputs."""
    metadata = cast(Any, collection).metadata or {}
    expected_metadata = {
        "hpo_version": manifest.hpo_version,
        "hpo_source_sha256": manifest.hpo_source_sha256,
        "model": model_name,
        "index_type": index_type,
        "expected_document_count": expected_document_count,
    }
    labels = {
        "hpo_version": "HPO version",
        "hpo_source_sha256": "HPO source SHA-256",
        "model": "model",
        "index_type": "index type",
        "expected_document_count": "expected document count",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Collection {labels[key]} does not match bundle input: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    if expected_model_revision is not None and (
        metadata.get("model_revision") != expected_model_revision
    ):
        raise ValueError(
            "Collection model revision does not match release spec: "
            f"expected {expected_model_revision!r}, got {metadata.get('model_revision')!r}"
        )

    actual_document_count = cast(Any, collection).count()
    if actual_document_count != expected_document_count:
        raise ValueError(
            "Collection document count does not match bundle input: "
            f"expected {expected_document_count}, got {actual_document_count}"
        )

    dimension = metadata.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(f"Collection has invalid embedding dimension: {dimension!r}")
    return dimension


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

        vector_store_config = VectorStoreConfig(
            path=str(index_dir),
            collection_name="_bundle_dimension_probe",
        )
        client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=vector_store_config.to_chromadb_settings(),
        )
        collections = client.list_collections()
        if collections:
            # Get dimension from first collection's metadata or peek
            collection = cast(Any, _collection_from_list_entry(client, collections[0]))
            # Try to peek at an embedding
            result = collection.peek(limit=1)
            embeddings = result.get("embeddings") if result else None
            if embeddings is not None and len(embeddings) > 0:
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

        vector_store_config = VectorStoreConfig(
            path=str(index_dir),
            collection_name=collection_name,
        )
        client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=vector_store_config.to_chromadb_settings(),
        )
        collections = client.list_collections()
        collection_names = [_collection_name(c) for c in collections]
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
