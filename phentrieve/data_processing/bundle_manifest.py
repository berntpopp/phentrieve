"""
Bundle manifest for pre-built data distribution (Issue #117).

This module defines the manifest format for pre-built HPO data bundles,
enabling version tracking, integrity verification, and reproducible builds.

The manifest is stored as manifest.json in bundle root and contains:
- HPO version and source metadata
- Embedding model information
- Term statistics (active vs obsolete)
- File checksums for integrity verification

See: https://github.com/berntpopp/phentrieve/issues/117
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Current manifest schema version
MANIFEST_VERSION = "1.0.0"

# Model slug mapping for shorter filenames
MODEL_SLUGS: dict[str, str] = {
    "FremyCompany/BioLORD-2023-M": "biolord",
    "jinaai/jina-embeddings-v2-base-de": "jina-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer": "tsystems-ende",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet-multi",
    "sentence-transformers/distiluse-base-multilingual-cased-v2": "distiluse-multi",
    "BAAI/bge-m3": "bge-m3",
    "Alibaba-NLP/gte-multilingual-base": "gte-multi",
    "sentence-transformers/LaBSE": "labse",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "minilm-multi",
}

# Reverse mapping for slug to model name
SLUG_TO_MODEL: dict[str, str] = {v: k for k, v in MODEL_SLUGS.items()}


def get_model_slug(model_name: str) -> str:
    """
    Get the short slug for a model name.

    Args:
        model_name: Full model name (e.g., "FremyCompany/BioLORD-2023-M")

    Returns:
        Short slug (e.g., "biolord") or sanitized name if not in mapping
    """
    if model_name in MODEL_SLUGS:
        return MODEL_SLUGS[model_name]
    # Fallback: sanitize the model name
    return model_name.replace("/", "_").replace("-", "_").lower()


def get_model_name(slug: str) -> str | None:
    """
    Get the full model name from a slug.

    Args:
        slug: Short slug (e.g., "biolord")

    Returns:
        Full model name or None if not found
    """
    return SLUG_TO_MODEL.get(slug)


@dataclass
class EmbeddingModelInfo:
    """
    Embedding model metadata for bundle.

    Stores information about the embedding model used to generate
    the vector index, enabling reproducibility.
    """

    name: str  # e.g., "FremyCompany/BioLORD-2023-M"
    slug: str  # e.g., "biolord"
    dimension: int  # e.g., 768
    distance_metric: str = "cosine"  # e.g., "cosine", "l2", "ip"

    @classmethod
    def from_model_name(
        cls, model_name: str, dimension: int, distance_metric: str = "cosine"
    ) -> EmbeddingModelInfo:
        """Create from model name with auto-generated slug."""
        return cls(
            name=model_name,
            slug=get_model_slug(model_name),
            dimension=dimension,
            distance_metric=distance_metric,
        )


@dataclass
class BundleManifest:
    """
    Manifest for pre-built data bundles.

    Stored as manifest.json in bundle root. Contains all metadata
    needed to verify bundle integrity and compatibility.

    Attributes:
        manifest_version: Schema version for forward compatibility
        bundle_format: Archive format (always "tar.gz")
        hpo_version: HPO ontology version (e.g., "v2025-03-03")
        hpo_release_date: Date of HPO release
        hpo_source_url: URL where HPO JSON was downloaded from
        total_terms: Total HPO terms before filtering
        active_terms: Terms after filtering obsolete
        obsolete_terms: Number of filtered obsolete terms
        model: Embedding model metadata (optional for minimal bundles)
        created_at: ISO 8601 timestamp of bundle creation
        created_by: Build system identifier
        phentrieve_version: Package version used for building
        checksums: SHA-256 checksums for all bundle files
    """

    # Versioning
    manifest_version: str = MANIFEST_VERSION
    bundle_format: str = "tar.gz"

    # HPO Data
    hpo_version: str = ""
    hpo_release_date: str = ""
    hpo_source_url: str = ""

    # Term Statistics
    total_terms: int = 0
    active_terms: int = 0
    obsolete_terms: int = 0

    # Embedding Model (required for bundles)
    model: EmbeddingModelInfo | None = None

    # Bundle Metadata
    created_at: str = ""
    created_by: str = "phentrieve"
    phentrieve_version: str = ""

    # Checksums (SHA-256)
    checksums: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for optional fields."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.phentrieve_version:
            try:
                from importlib.metadata import version

                self.phentrieve_version = version("phentrieve")
            except Exception:
                self.phentrieve_version = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        data = asdict(self)
        # Handle nested dataclass
        if self.model is not None:
            data["model"] = asdict(self.model)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info(f"Saved bundle manifest to {path}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
        """Create manifest from dictionary."""
        # Handle nested model dataclass
        model_data = data.pop("model", None)
        model = EmbeddingModelInfo(**model_data) if model_data else None
        return cls(model=model, **data)

    @classmethod
    def from_json(cls, json_str: str) -> BundleManifest:
        """Deserialize manifest from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path) -> BundleManifest:
        """Load manifest from file."""
        json_str = path.read_text(encoding="utf-8")
        manifest = cls.from_json(json_str)
        logger.info(f"Loaded bundle manifest from {path}")
        return manifest

    def get_bundle_filename(self) -> str:
        """
        Generate bundle filename from manifest metadata.

        Format: phentrieve-data-{hpo_version}-{model_slug}.tar.gz

        Examples:
            - phentrieve-data-v2025-03-03-biolord.tar.gz
            - phentrieve-data-v2025-03-03-bge-m3.tar.gz
        """
        if self.model is None:
            raise ValueError("Bundle requires a model to generate filename")
        return f"phentrieve-data-{self.hpo_version}-{self.model.slug}.tar.gz"

    def verify_checksum(self, file_path: Path, expected_key: str | None = None) -> bool:
        """
        Verify a file's checksum against stored value.

        Args:
            file_path: Path to file to verify
            expected_key: Key in checksums dict (defaults to file name)

        Returns:
            True if checksum matches, False otherwise
        """
        key = expected_key or file_path.name
        if key not in self.checksums:
            logger.warning(f"No checksum stored for {key}")
            return False

        actual_hash = compute_file_checksum(file_path)
        expected_hash = self.checksums[key]

        if actual_hash != expected_hash:
            logger.error(
                f"Checksum mismatch for {key}: expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )
            return False

        logger.debug(f"Checksum verified for {key}")
        return True

    def add_file_checksum(self, file_path: Path, key: str | None = None) -> str:
        """
        Compute and store checksum for a file.

        Args:
            file_path: Path to file
            key: Key to use in checksums dict (defaults to file name)

        Returns:
            Computed SHA-256 checksum
        """
        checksum = compute_file_checksum(file_path)
        self.checksums[key or file_path.name] = checksum
        return checksum


def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex-encoded checksum string
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_checksum(
    dir_path: Path, algorithm: str = "sha256", exclude: set[str] | None = None
) -> str:
    """
    Compute combined checksum of all files in a directory.

    Args:
        dir_path: Path to directory
        algorithm: Hash algorithm (default: sha256)
        exclude: Set of filenames to exclude

    Returns:
        Hex-encoded combined checksum
    """
    exclude = exclude or set()
    hasher = hashlib.new(algorithm)

    # Sort files for deterministic order
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.name not in exclude:
            # Include relative path in hash for structure awareness
            rel_path = file_path.relative_to(dir_path)
            hasher.update(str(rel_path).encode())
            hasher.update(compute_file_checksum(file_path, algorithm).encode())

    return hasher.hexdigest()
