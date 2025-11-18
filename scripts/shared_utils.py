"""
Shared utilities for standalone scripts.

Provides common functionality for scripts including:
- Provenance tracking for reproducibility
- Consistent logging setup
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

# Version for provenance tracking
SHARED_UTILS_VERSION = "1.0.0"


class ProvenanceTracker:
    """
    Provenance tracking for reproducibility.

    Tracks source version information for scripts and data processing.
    Extracted from phenobert_converter.py for reuse across scripts.
    """

    @staticmethod
    def get_git_version(repo_path: Path) -> Optional[dict[str, Any]]:
        """
        Extract version information from repository (git or ZIP download).

        Args:
            repo_path: Path to repository directory

        Returns:
            Dictionary with version metadata, or None if unavailable
        """
        git_dir = repo_path / ".git"

        # Try git repository first
        if git_dir.exists():
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
                    "download_method": "git_clone",
                }

            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        # Not a git repo - try to infer from directory name (ZIP download)
        return ProvenanceTracker._detect_zip_download(repo_path)

    @staticmethod
    def _detect_zip_download(repo_path: Path) -> Optional[dict[str, Any]]:
        """
        Detect version info from GitHub ZIP download.

        GitHub ZIPs are named like: RepoName-main or RepoName-{commit_sha}

        Args:
            repo_path: Path to extracted ZIP directory

        Returns:
            Dictionary with download metadata, or None if unavailable
        """
        dir_name = repo_path.name

        # Pattern: RepoName-main or RepoName-{sha}
        if "-" in dir_name:
            parts = dir_name.rsplit("-", 1)
            ref = parts[1]  # Either "main" or commit SHA

            # Check if it looks like a commit SHA (40 hex chars)
            if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
                return {
                    "commit_sha": ref,
                    "download_method": "github_zip",
                    "branch_or_tag": "unknown",
                }
            else:
                # Likely a branch name (e.g., "main")
                return {
                    "branch_or_tag": ref,
                    "download_method": "github_zip",
                    "note": f"Downloaded from '{ref}' branch - commit SHA unknown",
                }

        # Couldn't determine version
        return {
            "download_method": "unknown",
            "note": f"Downloaded source, version unknown (directory: {dir_name})",
        }


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging consistently across scripts.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
