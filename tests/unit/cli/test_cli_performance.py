"""Performance tests for CLI startup time.

These tests ensure that CLI commands that don't need ML models
complete quickly without loading heavy dependencies like PyTorch.
"""

import subprocess
import time

import pytest


@pytest.mark.unit
def test_cli_version_performance():
    """Ensure CLI version command is fast (<10 seconds).

    The version command should not load sentence-transformers or torch,
    which would add 18+ seconds of startup time. With lazy loading,
    this should complete in under 10 seconds even on slow filesystems (WSL2).

    Note: WSL2 filesystem overhead + Python startup + CLI framework loading
    can take 6-7s. The key metric is the test_no_heavy_imports_on_cli_load
    test which verifies no ML libraries are loaded.
    """
    start = time.time()
    result = subprocess.run(
        ["phentrieve", "--version"],  # noqa: S607 - Testing installed CLI command
        capture_output=True,
        text=True,
        timeout=15.0,  # Fail if takes >15s
    )
    elapsed = time.time() - start

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Phentrieve CLI version" in result.stdout
    assert elapsed < 10.0, (
        f"CLI version took {elapsed:.2f}s (expected <10s). "
        "This suggests heavy ML dependencies are being loaded at import time. "
        "Check that sentence-transformers imports use lazy loading."
    )


@pytest.mark.unit
def test_cli_help_performance():
    """Ensure CLI help command is fast (<10 seconds).

    The help command should not load sentence-transformers or torch.
    With lazy loading, this should complete quickly even on slow filesystems (WSL2).
    """
    start = time.time()
    result = subprocess.run(
        ["phentrieve", "--help"],  # noqa: S607 - Testing installed CLI command
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    elapsed = time.time() - start

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Phentrieve" in result.stdout
    assert "Commands" in result.stdout
    assert elapsed < 10.0, (
        f"CLI help took {elapsed:.2f}s (expected <10s). "
        "This suggests heavy ML dependencies are being loaded at import time."
    )


@pytest.mark.unit
def test_cli_subcommand_help_performance():
    """Ensure subcommand help is fast (<10 seconds)."""
    start = time.time()
    result = subprocess.run(
        ["phentrieve", "data", "--help"],  # noqa: S607 - Testing installed CLI command
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    elapsed = time.time() - start

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Manage HPO data" in result.stdout
    assert elapsed < 10.0, (
        f"Subcommand help took {elapsed:.2f}s (expected <10s). "
        "This suggests heavy ML dependencies are being loaded at import time."
    )


@pytest.mark.unit
def test_no_heavy_imports_on_cli_load():
    """Verify that importing CLI does not load sentence-transformers, torch, or chromadb.

    This test ensures lazy loading is working by checking that heavy ML/database
    libraries are not imported when the CLI module is loaded.
    """
    result = subprocess.run(
        [  # noqa: S607 - Testing Python interpreter in test environment
            "python",
            "-c",
            "import sys; from phentrieve.cli import app; "
            "heavy = [m for m in sys.modules if 'sentence_transformers' in m or 'torch' in m or 'chromadb' in m]; "
            "print('HEAVY_IMPORTS:', heavy)",
        ],
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    output = result.stdout.strip()

    # Check that no heavy imports were loaded
    assert "HEAVY_IMPORTS: []" in output, (
        f"Heavy ML/database libraries were imported at CLI load time: {output}. "
        "This defeats lazy loading. Check that sentence-transformers, torch, and chromadb "
        "imports are only done inside functions that actually use them."
    )
