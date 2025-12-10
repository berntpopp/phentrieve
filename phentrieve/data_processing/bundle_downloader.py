"""
Bundle downloader for pre-built data distribution (Issue #117).

This module handles downloading pre-built HPO data bundles from GitHub Releases,
including:
- Version discovery and listing
- Progress tracking during download
- Automatic extraction and verification
- Fallback to local data preparation if download fails

See: https://github.com/berntpopp/phentrieve/issues/117
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from phentrieve.data_processing.bundle_manifest import (
    SLUG_TO_MODEL,
    get_model_slug,
)
from phentrieve.data_processing.bundle_packager import extract_bundle
from phentrieve.utils import get_default_data_dir

if TYPE_CHECKING:
    from phentrieve.data_processing.bundle_manifest import BundleManifest

logger = logging.getLogger(__name__)

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_REPO = "berntpopp/phentrieve"
GITHUB_RELEASES_URL = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/releases"

# Download configuration
DEFAULT_TIMEOUT = 60  # seconds
DOWNLOAD_CHUNK_SIZE = 8192  # bytes


@dataclass
class ReleaseInfo:
    """Information about a GitHub release containing data bundles."""

    tag_name: str
    name: str
    published_at: str
    bundles: list[BundleAsset]
    prerelease: bool = False
    draft: bool = False


@dataclass
class BundleAsset:
    """Information about a bundle asset in a release."""

    name: str
    download_url: str
    size: int
    hpo_version: str | None = None
    model_slug: str | None = None


def list_available_releases(
    include_prereleases: bool = False,
    limit: int = 10,
) -> list[ReleaseInfo]:
    """
    List GitHub releases containing data bundles.

    Args:
        include_prereleases: Include pre-release versions
        limit: Maximum number of releases to return

    Returns:
        List of ReleaseInfo objects
    """
    url = f"{GITHUB_RELEASES_URL}?per_page={limit}"

    try:
        req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})  # noqa: S310
        with urlopen(req, timeout=DEFAULT_TIMEOUT) as response:  # noqa: S310
            releases_data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as e:
        logger.error(f"Failed to fetch releases: {e}")
        return []

    releases = []
    for release in releases_data:
        # Skip drafts
        if release.get("draft", False):
            continue

        # Skip prereleases unless requested
        if release.get("prerelease", False) and not include_prereleases:
            continue

        # Find bundle assets
        bundles = []
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.startswith("phentrieve-data-") and name.endswith(".tar.gz"):
                # Parse bundle filename
                hpo_version, model_slug = _parse_bundle_filename(name)
                bundles.append(
                    BundleAsset(
                        name=name,
                        download_url=asset.get("browser_download_url", ""),
                        size=asset.get("size", 0),
                        hpo_version=hpo_version,
                        model_slug=model_slug,
                    )
                )

        if bundles:
            releases.append(
                ReleaseInfo(
                    tag_name=release.get("tag_name", ""),
                    name=release.get("name", ""),
                    published_at=release.get("published_at", ""),
                    bundles=bundles,
                    prerelease=release.get("prerelease", False),
                    draft=release.get("draft", False),
                )
            )

    return releases


def _parse_bundle_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Parse bundle filename to extract HPO version and model slug.

    Args:
        filename: Bundle filename (e.g., "phentrieve-data-v2025-03-03-biolord.tar.gz")

    Returns:
        Tuple of (hpo_version, model_slug) or (None, None) if parsing fails
    """
    # Pattern: phentrieve-data-{hpo_version}-{model_slug}.tar.gz
    pattern = r"phentrieve-data-(v[\d-]+)-([a-z0-9_-]+)\.tar\.gz"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return match.group(1), match.group(2)
    return None, None


def find_bundle(
    model_name: str,
    hpo_version: str | None = None,
    release_tag: str | None = None,
) -> BundleAsset | None:
    """
    Find a specific bundle in available releases.

    Args:
        model_name: Full model name (e.g., "FremyCompany/BioLORD-2023-M")
                   or model slug (e.g., "biolord")
        hpo_version: Specific HPO version (default: latest)
        release_tag: Specific release tag (default: latest)

    Returns:
        BundleAsset if found, None otherwise
    """
    # Convert model name to slug
    target_slug = _normalize_model_slug(model_name)

    releases = list_available_releases(include_prereleases=False)

    if not releases:
        logger.warning("No releases found with data bundles")
        return None

    # Filter by release tag if specified
    if release_tag:
        releases = [r for r in releases if r.tag_name == release_tag]
        if not releases:
            logger.warning(f"Release {release_tag} not found")
            return None

    # Search through releases for matching bundle
    for release in releases:
        for bundle in release.bundles:
            # Check model slug match
            if bundle.model_slug != target_slug:
                continue

            # Check HPO version match if specified
            if hpo_version and bundle.hpo_version != hpo_version:
                continue

            logger.info(f"Found bundle: {bundle.name} in release {release.tag_name}")
            return bundle

    logger.warning(
        f"No bundle found for model_slug={target_slug}, "
        f"hpo_version={hpo_version}, release_tag={release_tag}"
    )
    return None


def _normalize_model_slug(model_name: str) -> str:
    """
    Normalize model name to slug.

    Args:
        model_name: Full model name or slug

    Returns:
        Normalized slug
    """
    # Check if it's already a known slug
    if model_name.lower() in SLUG_TO_MODEL:
        return model_name.lower()

    # Convert full model name to slug
    return get_model_slug(model_name)


def download_bundle(
    bundle: BundleAsset,
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Download a bundle to target directory.

    Args:
        bundle: BundleAsset to download
        target_dir: Directory to save bundle (default: temp directory)
        progress_callback: Optional callback(bytes_downloaded, total_bytes)

    Returns:
        Path to downloaded bundle file

    Raises:
        HTTPError: If download fails
        ValueError: If bundle URL is invalid
    """
    if not bundle.download_url:
        raise ValueError("Bundle has no download URL")

    target_dir = target_dir or Path(tempfile.mkdtemp())
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / bundle.name

    logger.info(f"Downloading {bundle.name} ({bundle.size / 1024 / 1024:.1f} MB)...")

    try:
        req = Request(bundle.download_url)  # noqa: S310
        with urlopen(req, timeout=DEFAULT_TIMEOUT * 10) as response:  # noqa: S310
            total_size = int(response.headers.get("content-length", bundle.size))
            downloaded = 0

            with open(target_path, "wb") as f:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(downloaded, total_size)

        logger.info(f"Downloaded: {target_path}")
        return target_path

    except (HTTPError, URLError) as e:
        logger.error(f"Download failed: {e}")
        if target_path.exists():
            target_path.unlink()
        raise


def download_and_extract_bundle(
    model_name: str,
    hpo_version: str | None = None,
    target_data_dir: Path | None = None,
    verify_checksums: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BundleManifest | None:
    """
    Download and extract a bundle to the data directory.

    This is the main entry point for downloading pre-built data.

    Args:
        model_name: Model name or slug (e.g., "biolord", "FremyCompany/BioLORD-2023-M")
        hpo_version: Specific HPO version (default: latest)
        target_data_dir: Target data directory (default: from config)
        verify_checksums: Verify checksums after extraction
        progress_callback: Optional callback(bytes_downloaded, total_bytes)

    Returns:
        BundleManifest if successful, None if bundle not found or download failed

    Example:
        >>> from phentrieve.data_processing.bundle_downloader import download_and_extract_bundle
        >>> manifest = download_and_extract_bundle(model_name="biolord")
        >>> if manifest:
        ...     print(f"Installed HPO {manifest.hpo_version} with {manifest.active_terms} terms")
    """
    # Find bundle
    bundle = find_bundle(model_name=model_name, hpo_version=hpo_version)
    if not bundle:
        return None

    target_data_dir = target_data_dir or get_default_data_dir()

    # Download to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            bundle_path = download_bundle(
                bundle,
                target_dir=temp_path,
                progress_callback=progress_callback,
            )

            # Extract bundle
            manifest = extract_bundle(
                bundle_path,
                target_dir=target_data_dir,
                verify_checksums=verify_checksums,
            )

            logger.info(
                f"Successfully installed bundle: HPO {manifest.hpo_version}, "
                f"{manifest.active_terms} active terms"
            )

            return manifest

        except Exception as e:
            logger.error(f"Failed to download and extract bundle: {e}")
            return None


def get_installed_bundle_info(data_dir: Path | None = None) -> BundleManifest | None:
    """
    Get information about the currently installed bundle.

    Args:
        data_dir: Data directory to check

    Returns:
        BundleManifest if manifest exists, None otherwise
    """
    from phentrieve.data_processing.bundle_manifest import BundleManifest

    data_dir = data_dir or get_default_data_dir()
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    try:
        return BundleManifest.load(manifest_path)
    except Exception as e:
        logger.warning(f"Failed to load manifest: {e}")
        return None


def check_for_updates(
    data_dir: Path | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """
    Check if a newer bundle is available.

    Args:
        data_dir: Data directory with installed bundle
        model_name: Model to check (uses installed model if None)

    Returns:
        Tuple of (update_available: bool, message: str)
    """
    installed = get_installed_bundle_info(data_dir)

    if not installed:
        return True, "No bundle installed"

    # Use installed model if not specified
    if model_name is None and installed.model:
        model_name = installed.model.name

    # Cannot check updates without a model
    if model_name is None:
        return False, "No model specified and no model in installed bundle"

    # Find latest available bundle
    latest = find_bundle(model_name=model_name)

    if not latest:
        return False, "No bundles available for download"

    # Compare HPO versions
    if latest.hpo_version and installed.hpo_version:
        if latest.hpo_version > installed.hpo_version:
            return (
                True,
                f"Update available: {installed.hpo_version} → {latest.hpo_version}",
            )

    return False, f"Up to date: {installed.hpo_version}"


def cleanup_downloaded_bundles(data_dir: Path | None = None) -> int:
    """
    Clean up any leftover downloaded bundle files.

    Args:
        data_dir: Data directory to clean

    Returns:
        Number of files cleaned up
    """
    data_dir = data_dir or get_default_data_dir()
    count = 0

    for item in data_dir.glob("*.tar.gz"):
        if item.name.startswith("phentrieve-data-"):
            try:
                item.unlink()
                count += 1
                logger.info(f"Removed: {item}")
            except OSError as e:
                logger.warning(f"Failed to remove {item}: {e}")

    return count
