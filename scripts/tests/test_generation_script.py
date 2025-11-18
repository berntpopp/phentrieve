"""
Integration tests for generate_chunking_variants.py CLI script.

Tests the end-to-end functionality of the CLI including file processing,
dry-run mode, force overwrite, and error handling.
"""
# ruff: noqa: S603, S607

import json
import subprocess

import pytest


@pytest.fixture
def sample_doc(tmp_path):
    """Create sample document for testing."""
    doc = {
        "doc_id": "test_001",
        "language": "en",
        "source": "test",
        "full_text": "Patient has brachydactyly and hypotonia.",
        "metadata": {},
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {"start_char": 12, "end_char": 25, "text_snippet": "brachydactyly"}
                ],
            },
            {
                "hpo_id": "HP:0001252",
                "label": "Hypotonia",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {"start_char": 30, "end_char": 39, "text_snippet": "hypotonia"}
                ],
            },
        ],
    }

    file_path = tmp_path / "test_001.json"
    with open(file_path, "w") as f:
        json.dump(doc, f, indent=2)

    return file_path


@pytest.fixture
def sample_dir(tmp_path):
    """Create directory with multiple test documents."""
    annotations_dir = tmp_path / "dataset" / "annotations"
    annotations_dir.mkdir(parents=True)

    for i in range(3):
        doc = {
            "doc_id": f"test_{i:03d}",
            "language": "en",
            "source": "test",
            "full_text": f"Test document {i} with annotation.",
            "metadata": {},
            "annotations": [
                {
                    "hpo_id": "HP:0001156",
                    "label": "Test",
                    "assertion_status": "affirmed",
                    "evidence_spans": [{"start_char": 15 + i, "end_char": 25 + i}],
                }
            ],
        }

        file_path = annotations_dir / f"test_{i:03d}.json"
        with open(file_path, "w") as f:
            json.dump(doc, f, indent=2)

    return tmp_path


# ============================================================================
# Tests for single file processing
# ============================================================================


def test_cli_single_file_success(sample_doc):
    """Test CLI with single file processes successfully."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--expansion-ratios",
            "0.0",
            "1.0",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0

    # Verify file was modified
    with open(sample_doc) as f:
        doc = json.load(f)

    assert "chunk_variants" in doc
    assert "voronoi_v1" in doc["chunk_variants"]

    # Check structure
    chunks = doc["chunk_variants"]["voronoi_v1"]
    assert "provenance" in chunks
    assert "chunks" in chunks
    assert len(chunks["chunks"]) == 2


def test_cli_single_file_custom_strategy_name(sample_doc):
    """Test CLI with custom strategy name."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--strategy-name",
            "custom_strategy",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0

    with open(sample_doc) as f:
        doc = json.load(f)

    assert "custom_strategy" in doc["chunk_variants"]


def test_cli_single_file_custom_ratios(sample_doc):
    """Test CLI with custom expansion ratios."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--expansion-ratios",
            "0.0",
            "0.25",
            "0.75",
            "1.0",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0

    with open(sample_doc) as f:
        doc = json.load(f)

    chunks = doc["chunk_variants"]["voronoi_v1"]["chunks"][0]
    assert "0.00" in chunks["variants"]
    assert "0.25" in chunks["variants"]
    assert "0.75" in chunks["variants"]
    assert "1.00" in chunks["variants"]


# ============================================================================
# Tests for directory processing
# ============================================================================


def test_cli_directory_processing(sample_dir):
    """Test CLI with directory processes all matching files."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input-dir",
            str(sample_dir),
            "--pattern",
            "*/annotations/*.json",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    assert (
        "Successfully processed 3/3 files" in result.stderr
        or "Successfully processed 3/3 files" in result.stdout
    )

    # Verify all files were processed
    annotations_dir = sample_dir / "dataset" / "annotations"
    for i in range(3):
        file_path = annotations_dir / f"test_{i:03d}.json"
        with open(file_path) as f:
            doc = json.load(f)
        assert "chunk_variants" in doc


# ============================================================================
# Tests for dry-run mode
# ============================================================================


def test_cli_dry_run_no_changes(sample_doc):
    """Test dry-run doesn't modify files."""
    # Read original
    with open(sample_doc) as f:
        original = json.load(f)

    # Run dry-run
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    assert "[DRY RUN]" in result.stderr or "[DRY RUN]" in result.stdout

    # Verify unchanged
    with open(sample_doc) as f:
        after = json.load(f)

    assert original == after


# ============================================================================
# Tests for force overwrite
# ============================================================================


def test_cli_skip_existing_without_force(sample_doc):
    """Test CLI skips already-processed files without --force."""
    # Run once
    subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
        ],
        check=True,
        cwd="/mnt/c/development/phentrieve",
    )

    # Run again without force
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--log-level",
            "DEBUG",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    # Should skip (check debug log)
    output = result.stderr + result.stdout
    assert "Skipping" in output or "already" in output.lower()


def test_cli_force_overwrite(sample_doc):
    """Test CLI force flag overwrites existing chunks."""
    # Run once
    subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
        ],
        check=True,
        cwd="/mnt/c/development/phentrieve",
    )

    # Get first timestamp
    with open(sample_doc) as f:
        doc1 = json.load(f)
    timestamp1 = doc1["chunk_variants"]["voronoi_v1"]["provenance"]["generated_at"]

    # Wait a tiny bit to ensure different timestamp
    import time

    time.sleep(0.1)

    # Run again with force
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--force",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    assert "Processed" in result.stderr or "Processed" in result.stdout

    # Get second timestamp (should be different)
    with open(sample_doc) as f:
        doc2 = json.load(f)
    timestamp2 = doc2["chunk_variants"]["voronoi_v1"]["provenance"]["generated_at"]

    # Timestamps should differ (proving it was rewritten)
    assert timestamp1 != timestamp2


# ============================================================================
# Tests for error handling
# ============================================================================


def test_cli_nonexistent_file():
    """Test CLI with nonexistent file fails gracefully."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            "/nonexistent/file.json",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0  # Script itself doesn't fail
    output = result.stderr + result.stdout
    assert "not found" in output.lower() or "error" in output.lower()


def test_cli_nonexistent_directory():
    """Test CLI with nonexistent directory fails gracefully."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input-dir",
            "/nonexistent/directory",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0  # Script itself doesn't fail
    output = result.stderr + result.stdout
    assert "not found" in output.lower() or "error" in output.lower()


def test_cli_invalid_json(tmp_path):
    """Test CLI with invalid JSON file."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ invalid json }")

    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(bad_file),
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    # Should report error but not crash
    output = result.stderr + result.stdout
    assert "Error" in output or "error" in output


# ============================================================================
# Tests for logging levels
# ============================================================================


def test_cli_log_level_debug(sample_doc):
    """Test CLI with DEBUG log level produces verbose output."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--log-level",
            "DEBUG",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    # DEBUG should show more info
    output = result.stderr + result.stdout
    assert len(output) > 0


def test_cli_log_level_error(sample_doc):
    """Test CLI with ERROR log level produces minimal output."""
    result = subprocess.run(  # noqa: S603
        [
            "python",  # noqa: S607
            "scripts/generate_chunking_variants.py",
            "--input",
            str(sample_doc),
            "--log-level",
            "ERROR",
        ],
        capture_output=True,
        text=True,
        cwd="/mnt/c/development/phentrieve",
    )

    assert result.returncode == 0
    # Should still succeed but with less output
