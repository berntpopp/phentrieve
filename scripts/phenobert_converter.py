"""
PhenoBERT corpus converter for Phentrieve.

Converts PhenoBERT datasets (GSC+, ID-68, GeneReviews) to Phentrieve JSON format.
Simple, focused implementation following alpha software principles.

Includes provenance tracking for reproducibility:
- Source version (git commit SHA or release tag)
- File checksums (SHA256)
- Conversion timestamp
- Converter version
"""

import hashlib
import json
import logging
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Converter version for reproducibility
CONVERTER_VERSION = "1.0.0"


# ============================================================================
# Provenance Tracking
# ============================================================================


class ProvenanceTracker:
    """Tracks provenance information for reproducibility."""

    @staticmethod
    def get_git_version(repo_path: Path) -> Optional[dict[str, Any]]:
        """
        Extract git version information from repository.

        Args:
            repo_path: Path to git repository

        Returns:
            Dictionary with git metadata, or None if not a git repo
        """
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            logger.info("Not a git repository, skipping version detection")
            return None

        try:
            # Get commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_sha = result.stdout.strip()

            # Get commit date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],  # noqa: S607
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_date = result.stdout.strip()

            # Check if repo is dirty (uncommitted changes)
            result = subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            is_dirty = bool(result.stdout.strip())

            return {
                "commit_sha": commit_sha,
                "commit_date": commit_date,
                "is_dirty": is_dirty,
                "repository": "https://github.com/EclipseCN/PhenoBERT",
            }

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to extract git version: {e}")
            return None

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """
        Calculate SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal SHA256 checksum
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Annotation:
    """Single annotation with character offsets."""

    start: int
    end: int
    text: str
    hpo_id: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate annotation on creation."""
        if self.start < 0:
            raise ValueError(f"Invalid start offset: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Invalid end offset: {self.end} (start: {self.start})")
        if not self.hpo_id.startswith("HP:"):
            raise ValueError(f"Invalid HPO ID format: {self.hpo_id}")


@dataclass
class ConversionStats:
    """Statistics for conversion tracking with provenance metadata."""

    total_docs: int = 0
    total_annotations: int = 0
    datasets: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def add_document(self, dataset: str, num_annotations: int) -> None:
        """Record document conversion."""
        self.total_docs += 1
        self.total_annotations += num_annotations

        if dataset not in self.datasets:
            self.datasets[dataset] = {
                "docs": 0,
                "annotations": 0,
            }

        self.datasets[dataset]["docs"] += 1
        self.datasets[dataset]["annotations"] += num_annotations

    def set_provenance(
        self,
        source_version: Optional[dict[str, Any]],
        converter_version: str,
        conversion_date: str,
    ) -> None:
        """
        Set provenance metadata for reproducibility.

        Args:
            source_version: Git version info from source repository
            converter_version: Version of this converter
            conversion_date: ISO-8601 timestamp of conversion
        """
        self.provenance = {
            "converter_version": converter_version,
            "conversion_date": conversion_date,
            "source": {
                "repository": "https://github.com/EclipseCN/PhenoBERT",
                "version": source_version or {"note": "Not a git repository"},
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Export stats as dictionary with provenance."""
        return {
            "provenance": self.provenance,
            "summary": {
                "total_documents": self.total_docs,
                "total_annotations": self.total_annotations,
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "datasets": self.datasets,
        }


# ============================================================================
# Component 1: DatasetLoader
# ============================================================================


class DatasetLoader:
    """Discovers and loads corpus and annotation file pairs."""

    # Dataset directory mappings
    DATASET_DIRS = {
        "GSC+": ("GSC+/corpus", "GSC+/ann"),
        "ID-68": ("ID-68/corpus", "ID-68/ann"),
        "GeneReviews": ("GeneReviews/corpus", "GeneReviews/ann"),
    }

    def __init__(self, phenobert_data_dir: Path):
        """Initialize loader with PhenoBERT data directory."""
        self.data_dir = Path(phenobert_data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"PhenoBERT data directory not found: {self.data_dir}"
            )

    def discover_datasets(self) -> list[str]:
        """
        Discover available datasets in the data directory.

        Returns:
            List of dataset names (e.g., ["GSC+", "ID-68", "GeneReviews"])
        """
        available = []

        for dataset in self.DATASET_DIRS:
            corpus_dir, ann_dir = self.DATASET_DIRS[dataset]
            corpus_path = self.data_dir / corpus_dir
            ann_path = self.data_dir / ann_dir

            if corpus_path.exists() and ann_path.exists():
                available.append(dataset)
                logger.debug(f"Found dataset: {dataset}")
            else:
                logger.warning(
                    f"Dataset {dataset} incomplete (missing corpus or annotations)"
                )

        return available

    def load_file_pairs(self, dataset: str) -> Iterator[tuple[Path, Path]]:
        """
        Load corpus and annotation file pairs for a dataset.

        Args:
            dataset: Dataset name (e.g., "GSC+")

        Yields:
            Tuples of (corpus_file, annotation_file)
        """
        if dataset not in self.DATASET_DIRS:
            raise ValueError(f"Unknown dataset: {dataset}")

        corpus_subdir, ann_subdir = self.DATASET_DIRS[dataset]
        corpus_dir = self.data_dir / corpus_subdir
        ann_dir = self.data_dir / ann_subdir

        # Get all corpus files
        corpus_files = sorted(corpus_dir.glob("*.txt"))

        logger.info(f"Found {len(corpus_files)} corpus files in {dataset}")

        for corpus_file in corpus_files:
            # Match annotation file by stem (filename without extension)
            ann_file = ann_dir / f"{corpus_file.stem}.ann"

            if ann_file.exists():
                yield (corpus_file, ann_file)
            else:
                logger.warning(f"No annotation file for {corpus_file.name}")

    def read_text(self, file_path: Path) -> str:
        """
        Read text file with automatic encoding detection.

        Args:
            file_path: Path to text file

        Returns:
            File contents as string
        """
        # Try UTF-8 first (most common)
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logger.debug(f"UTF-8 failed for {file_path.name}, trying Latin-1")

        # Fallback to Latin-1 (always succeeds)
        try:
            with open(file_path, encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            raise OSError(f"Failed to read {file_path}: {e}")


# ============================================================================
# Component 2: AnnotationParser
# ============================================================================


class AnnotationParser:
    """
    Unified parser with automatic format detection.

    Handles both raw and processed PhenoBERT annotation formats.
    """

    # Regex patterns for format detection and parsing
    PATTERN_RAW = re.compile(r"\[(\d+)::(\d+)\]\s+(HP_\d+)\s+\|\s+(.+)")
    PATTERN_PROCESSED = re.compile(r"^(\d+)\t(\d+)\t([^\t]+)\t(HP:\d+)(?:\t([\d.]+))?$")

    def parse(self, ann_file: Path) -> list[Annotation]:
        """
        Parse annotation file with automatic format detection.

        Args:
            ann_file: Path to annotation file

        Returns:
            List of Annotation objects
        """
        format_type = self._detect_format(ann_file)

        if format_type == "raw":
            return self._parse_raw(ann_file)
        elif format_type == "processed":
            return self._parse_processed(ann_file)
        else:
            raise ValueError(f"Unknown annotation format in {ann_file}")

    def _detect_format(self, ann_file: Path) -> str:
        """
        Detect annotation format from first line.

        Args:
            ann_file: Path to annotation file

        Returns:
            "raw" or "processed"
        """
        with open(ann_file, encoding="utf-8") as f:
            # Skip empty lines
            for line in f:
                line = line.strip()
                if line:
                    # Check for raw format: starts with '['
                    if line.startswith("["):
                        return "raw"
                    # Check for processed format: contains tabs
                    elif "\t" in line:
                        return "processed"
                    else:
                        break

        raise ValueError(f"Could not detect annotation format in {ann_file}")

    def _parse_raw(self, ann_file: Path) -> list[Annotation]:
        """
        Parse raw format: [27::42] HP_0000110 | renal dysplasia

        Args:
            ann_file: Path to annotation file

        Returns:
            List of Annotation objects
        """
        return self._parse_with_pattern(ann_file, self.PATTERN_RAW, is_raw=True)

    def _parse_processed(self, ann_file: Path) -> list[Annotation]:
        """
        Parse processed format: 9\t17\theadache\tHP:0002315\t1.0

        Args:
            ann_file: Path to annotation file

        Returns:
            List of Annotation objects
        """
        return self._parse_with_pattern(ann_file, self.PATTERN_PROCESSED, is_raw=False)

    def _parse_with_pattern(
        self,
        ann_file: Path,
        pattern: re.Pattern[str],
        is_raw: bool,
    ) -> list[Annotation]:
        """
        Shared parsing logic (DRY principle).

        File reading, error handling, and HPO normalization done once.

        Args:
            ann_file: Path to annotation file
            pattern: Compiled regex pattern
            is_raw: True if raw format, False if processed

        Returns:
            List of Annotation objects
        """
        annotations = []

        with open(ann_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match = pattern.match(line)
                if not match:
                    logger.warning(
                        f"Invalid format at {ann_file.name}:{line_num}: {line[:50]}"
                    )
                    continue

                try:
                    confidence_value: Optional[float]
                    if is_raw:
                        # Raw format: [start::end] HPO_ID | text
                        start, end, hpo_id, text = match.groups()
                        confidence_value = None
                    else:
                        # Processed format: start\tend\ttext\tHPO:ID\tconfidence
                        start, end, text, hpo_id, confidence_str = match.groups()
                        confidence_value = float(confidence_str) if confidence_str else None

                    # Normalize HPO ID (HP_NNNN → HP:NNNN)
                    hpo_id = hpo_id.replace("_", ":")

                    annotations.append(
                        Annotation(
                            start=int(start),
                            end=int(end),
                            text=text,
                            hpo_id=hpo_id,
                            confidence=confidence_value,
                        )
                    )

                except (ValueError, IndexError) as e:
                    logger.warning(f"Parse error at {ann_file.name}:{line_num}: {e}")
                    continue

        logger.debug(f"Parsed {len(annotations)} annotations from {ann_file.name}")
        return annotations


# ============================================================================
# Component 3: HPOLookup
# ============================================================================


class HPOLookup:
    """Simple HPO label lookup from Phentrieve's data."""

    def __init__(self, hpo_data_path: Path):
        """
        Initialize lookup with HPO data.

        Args:
            hpo_data_path: Path to hpo_core_data directory
        """
        self.cache: dict[str, str] = {}
        hpo_file = Path(hpo_data_path) / "hpo_terms.tsv"

        if not hpo_file.exists():
            raise FileNotFoundError(f"HPO terms file not found: {hpo_file}")

        self._load_hpo_data(hpo_file)

    def _load_hpo_data(self, hpo_file: Path) -> None:
        """
        Load HPO terms into memory cache.

        Args:
            hpo_file: Path to hpo_terms.tsv
        """
        with open(hpo_file, encoding="utf-8") as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    hpo_id = parts[0]
                    label = parts[1]
                    self.cache[hpo_id] = label

        logger.info(f"Loaded {len(self.cache)} HPO terms")

    def get_label(self, hpo_id: str) -> Optional[str]:
        """
        Get label for HPO ID.

        Args:
            hpo_id: HPO identifier (e.g., "HP:0001263")

        Returns:
            HPO term label, or None if not found
        """
        return self.cache.get(hpo_id)


# ============================================================================
# Component 4: PhenoBERTConverter
# ============================================================================


class PhenoBERTConverter:
    """
    Main converter with dependency injection.

    Focused responsibilities:
    - Convert documents to JSON
    - Validate spans
    - Enrich with HPO labels
    """

    def __init__(
        self,
        hpo_lookup: HPOLookup,
        dataset_loader: DatasetLoader,
        annotation_parser: AnnotationParser,
    ):
        """
        Initialize converter with dependencies.

        Args:
            hpo_lookup: HPO label lookup service
            dataset_loader: Dataset loading service
            annotation_parser: Annotation parsing service
        """
        self.hpo_lookup = hpo_lookup
        self.dataset_loader = dataset_loader
        self.annotation_parser = annotation_parser

    def convert_document(
        self,
        corpus_file: Path,
        ann_file: Path,
        dataset_name: str,
    ) -> dict[str, Any]:
        """
        Convert single document to Phentrieve JSON format.

        Args:
            corpus_file: Path to corpus text file
            ann_file: Path to annotation file
            dataset_name: Name of dataset (for doc_id generation)

        Returns:
            Document in Phentrieve JSON format
        """
        # 1. Load text
        full_text = self.dataset_loader.read_text(corpus_file)

        # 2. Parse annotations
        annotations = self.annotation_parser.parse(ann_file)

        # 3. Validate spans (single validation pass)
        valid_annotations = self._validate_spans(
            full_text, annotations, corpus_file.name
        )

        # 4. Build JSON
        return self._build_json(
            doc_id=f"{dataset_name}_{corpus_file.stem}",
            full_text=full_text,
            annotations=valid_annotations,
            dataset_name=dataset_name,
        )

    def _validate_spans(
        self,
        text: str,
        annotations: list[Annotation],
        filename: str,
    ) -> list[Annotation]:
        """
        Single validation pass - checks offsets and text match.

        Args:
            text: Full document text
            annotations: List of annotations to validate
            filename: Filename for logging

        Returns:
            List of valid annotations
        """
        valid = []

        for ann in annotations:
            # Check bounds
            if ann.start < 0 or ann.end > len(text):
                logger.warning(
                    f"{filename}: Out of bounds [{ann.start}:{ann.end}] for {ann.hpo_id}"
                )
                continue

            # Check text match
            actual = text[ann.start : ann.end]
            if actual != ann.text:
                # Try normalized comparison (whitespace differences)
                if actual.strip() == ann.text.strip():
                    logger.debug(
                        f"{filename}: Whitespace mismatch at [{ann.start}:{ann.end}], "
                        f"accepting anyway"
                    )
                else:
                    logger.warning(
                        f"{filename}: Text mismatch at [{ann.start}:{ann.end}] "
                        f"for {ann.hpo_id}: expected '{ann.text}', got '{actual}'"
                    )
                    continue

            valid.append(ann)

        return valid

    def _build_json(
        self,
        doc_id: str,
        full_text: str,
        annotations: list[Annotation],
        dataset_name: str,
    ) -> dict[str, Any]:
        """
        Build Phentrieve JSON structure.

        Args:
            doc_id: Document identifier
            full_text: Full document text
            annotations: List of validated annotations
            dataset_name: Dataset name for metadata

        Returns:
            Document in Phentrieve JSON format
        """
        # Group annotations by HPO ID
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann.hpo_id].append(ann)

        # Build annotations array
        json_annotations = []
        for hpo_id, spans in grouped.items():
            # Get HPO label
            label = self.hpo_lookup.get_label(hpo_id)
            if label is None:
                logger.warning(f"HPO label not found for {hpo_id}, using 'UNKNOWN'")
                label = "UNKNOWN"

            # Build evidence spans
            evidence_spans = []
            for span in spans:
                span_dict = {
                    "start_char": span.start,
                    "end_char": span.end,
                    "text_snippet": span.text,
                }
                # Only include confidence if present
                if span.confidence is not None:
                    span_dict["confidence"] = span.confidence

                evidence_spans.append(span_dict)

            json_annotations.append(
                {
                    "hpo_id": hpo_id,
                    "label": label,
                    "assertion_status": "affirmed",
                    "evidence_spans": evidence_spans,
                }
            )

        # Build complete document
        return {
            "doc_id": doc_id,
            "language": "en",
            "source": "phenobert",
            "full_text": full_text,
            "metadata": {
                "dataset": dataset_name,
                "text_length_chars": len(full_text),
                "num_annotations": len(json_annotations),
                "conversion_date": datetime.now().isoformat(),
            },
            "annotations": json_annotations,
        }


# ============================================================================
# Component 5: OutputWriter
# ============================================================================


class OutputWriter:
    """Writes converted documents and generates reports with provenance."""

    def __init__(self, output_dir: Path, source_data_dir: Path):
        """
        Initialize writer with output directory.

        Args:
            output_dir: Base output directory
            source_data_dir: Source PhenoBERT data directory (for provenance)
        """
        self.output_dir = Path(output_dir)
        self.stats = ConversionStats()

        # Extract provenance information
        self._initialize_provenance(source_data_dir)

    def _initialize_provenance(self, source_data_dir: Path) -> None:
        """
        Initialize provenance tracking from source directory.

        Args:
            source_data_dir: Path to PhenoBERT data directory
        """
        # Try to get parent directory (should be repo root)
        repo_root = source_data_dir.parent

        # Extract git version if available
        source_version = ProvenanceTracker.get_git_version(repo_root)

        if source_version:
            logger.info(f"Source version: {source_version['commit_sha'][:8]}")
            if source_version["is_dirty"]:
                logger.warning("Source repository has uncommitted changes!")
        else:
            logger.info("Source data not from git repository")

        # Set provenance in stats
        self.stats.set_provenance(
            source_version=source_version,
            converter_version=CONVERTER_VERSION,
            conversion_date=datetime.now().isoformat(),
        )

    def write_document(self, doc: dict[str, Any], dataset: str) -> None:
        """
        Write single JSON document.

        Args:
            doc: Document in Phentrieve JSON format
            dataset: Dataset name (for directory organization)
        """
        # Create dataset directory
        dataset_dir = self.output_dir / dataset / "annotations"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        output_file = dataset_dir / f"{doc['doc_id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        # Update stats
        self.stats.add_document(dataset, len(doc["annotations"]))

        logger.debug(f"Wrote {output_file}")

    def write_report(self) -> None:
        """Write conversion report with provenance metadata."""
        report_file = self.output_dir / "conversion_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Conversion complete: {self.stats.total_docs} documents")
        logger.info(f"Report saved to: {report_file}")

        # Log provenance info
        prov = self.stats.provenance
        if "source" in prov and "version" in prov["source"]:
            source_ver = prov["source"]["version"]
            if "commit_sha" in source_ver:
                logger.info(f"Source commit: {source_ver['commit_sha'][:8]}")
                logger.info(f"Source date: {source_ver['commit_date']}")
            logger.info(f"Converter version: {prov['converter_version']}")
