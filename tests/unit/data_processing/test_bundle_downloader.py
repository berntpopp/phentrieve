"""Unit tests for bundle downloader module."""

import pytest

from phentrieve.data_processing.bundle_downloader import (
    BundleAsset,
    ReleaseInfo,
    _parse_bundle_filename,
    check_for_updates,
    download_and_extract_bundle,
    find_bundle,
    get_data_release_releases_url,
    get_data_release_repository,
)
from phentrieve.data_processing.bundle_manifest import (
    BundleManifest,
    EmbeddingModelInfo,
)

pytestmark = pytest.mark.unit


def test_data_release_repository_defaults_to_dedicated_repository(monkeypatch):
    monkeypatch.delenv("PHENTRIEVE_DATA_RELEASE_REPOSITORY", raising=False)

    assert get_data_release_repository() == "berntpopp/phentrieve-data"
    assert (
        get_data_release_releases_url()
        == "https://api.github.com/repos/berntpopp/phentrieve-data/releases"
    )


def test_data_release_repository_honors_explicit_legacy_override(monkeypatch):
    monkeypatch.setenv("PHENTRIEVE_DATA_RELEASE_REPOSITORY", "berntpopp/phentrieve")

    assert get_data_release_repository() == "berntpopp/phentrieve"
    assert (
        get_data_release_releases_url()
        == "https://api.github.com/repos/berntpopp/phentrieve/releases"
    )


def test_parse_bundle_filename_detects_single_and_multivector_assets():
    assert _parse_bundle_filename("phentrieve-data-v2026-02-16-minimal.tar.gz") == (
        "v2026-02-16",
        "minimal",
        False,
    )
    assert _parse_bundle_filename("phentrieve-data-v2026-02-16-biolord.tar.gz") == (
        "v2026-02-16",
        "biolord",
        False,
    )
    assert _parse_bundle_filename(
        "phentrieve-data-v2026-02-16-biolord-multivec.tar.gz"
    ) == (
        "v2026-02-16",
        "biolord",
        True,
    )


def test_find_bundle_defaults_to_single_vector_asset(mocker):
    release = ReleaseInfo(
        tag_name="data-v2026-02-16",
        name="Data v2026-02-16",
        published_at="2026-02-16T00:00:00Z",
        bundles=[
            BundleAsset(
                name="phentrieve-data-v2026-02-16-biolord-multivec.tar.gz",
                download_url="https://example.test/multivec.tar.gz",
                size=100,
                hpo_version="v2026-02-16",
                model_slug="biolord",
                multi_vector=True,
            ),
            BundleAsset(
                name="phentrieve-data-v2026-02-16-biolord.tar.gz",
                download_url="https://example.test/single.tar.gz",
                size=100,
                hpo_version="v2026-02-16",
                model_slug="biolord",
                multi_vector=False,
            ),
        ],
    )
    mocker.patch(
        "phentrieve.data_processing.bundle_downloader.list_available_releases",
        return_value=[release],
    )

    bundle = find_bundle(model_name="biolord")

    assert bundle is not None
    assert bundle.name == "phentrieve-data-v2026-02-16-biolord.tar.gz"
    assert bundle.multi_vector is False


def test_find_bundle_can_select_multivector_asset(mocker):
    release = ReleaseInfo(
        tag_name="data-v2026-02-16",
        name="Data v2026-02-16",
        published_at="2026-02-16T00:00:00Z",
        bundles=[
            BundleAsset(
                name="phentrieve-data-v2026-02-16-biolord.tar.gz",
                download_url="https://example.test/single.tar.gz",
                size=100,
                hpo_version="v2026-02-16",
                model_slug="biolord",
                multi_vector=False,
            ),
            BundleAsset(
                name="phentrieve-data-v2026-02-16-biolord-multivec.tar.gz",
                download_url="https://example.test/multivec.tar.gz",
                size=100,
                hpo_version="v2026-02-16",
                model_slug="biolord",
                multi_vector=True,
            ),
        ],
    )
    mocker.patch(
        "phentrieve.data_processing.bundle_downloader.list_available_releases",
        return_value=[release],
    )

    bundle = find_bundle(model_name="biolord", multi_vector=True)

    assert bundle is not None
    assert bundle.name == "phentrieve-data-v2026-02-16-biolord-multivec.tar.gz"
    assert bundle.multi_vector is True


def test_download_and_extract_bundle_passes_multivector_selection(mocker, tmp_path):
    bundle = BundleAsset(
        name="phentrieve-data-v2026-02-16-biolord-multivec.tar.gz",
        download_url="https://example.test/multivec.tar.gz",
        size=100,
        hpo_version="v2026-02-16",
        model_slug="biolord",
        multi_vector=True,
    )
    manifest = mocker.Mock()
    find = mocker.patch(
        "phentrieve.data_processing.bundle_downloader.find_bundle",
        return_value=bundle,
    )
    download = mocker.patch(
        "phentrieve.data_processing.bundle_downloader.download_bundle",
        return_value=tmp_path / bundle.name,
    )
    extract = mocker.patch(
        "phentrieve.data_processing.bundle_downloader.extract_bundle",
        return_value=manifest,
    )

    result = download_and_extract_bundle(
        model_name="biolord",
        hpo_version="v2026-02-16",
        target_data_dir=tmp_path,
        multi_vector=True,
    )

    assert result == manifest
    find.assert_called_once_with(
        model_name="biolord",
        hpo_version="v2026-02-16",
        multi_vector=True,
    )
    download.assert_called_once()
    extract.assert_called_once()


def test_check_for_updates_preserves_installed_multivector_selection(mocker):
    installed = BundleManifest(
        hpo_version="v2025-11-24",
        model=EmbeddingModelInfo(
            name="FremyCompany/BioLORD-2023-M",
            slug="biolord",
            dimension=768,
            multi_vector=True,
        ),
    )
    latest = BundleAsset(
        name="phentrieve-data-v2026-02-16-biolord-multivec.tar.gz",
        download_url="https://example.test/multivec.tar.gz",
        size=100,
        hpo_version="v2026-02-16",
        model_slug="biolord",
        multi_vector=True,
    )
    mocker.patch(
        "phentrieve.data_processing.bundle_downloader.get_installed_bundle_info",
        return_value=installed,
    )
    find = mocker.patch(
        "phentrieve.data_processing.bundle_downloader.find_bundle",
        return_value=latest,
    )

    update_available, message = check_for_updates()

    assert update_available is True
    assert "v2026-02-16" in message
    find.assert_called_once_with(
        model_name="FremyCompany/BioLORD-2023-M",
        multi_vector=True,
    )
