"""Tests for the offline HPO data-release verifier."""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

import pytest

from phentrieve.data_processing.bundle_manifest import (
    BundleManifest,
    EmbeddingModelInfo,
)
from phentrieve.data_processing.release_contract import (
    DataReleaseSpec,
    ModelReleaseSpec,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_data_release import expected_bundle_names, verify_release  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def spec() -> DataReleaseSpec:
    return DataReleaseSpec(
        release_tag="hpo-v2026-06-23-r1",
        hpo_version="v2026-06-23",
        hpo_release_date="2026-06-23",
        hpo_source_url="https://example.org/hp.json",
        hpo_sha256="a" * 64,
        phentrieve_version="0.26.1",
        source_commit="b" * 40,
        lockfile_sha256="c" * 64,
        models=(ModelReleaseSpec("example/model", "example", "d" * 40),),
        active_terms=2,
        multivector_documents=3,
    )


def _write_archive(
    bundle_dir: Path,
    manifest: BundleManifest,
) -> Path:
    root = bundle_dir / manifest.get_bundle_filename().removesuffix(".tar.gz")
    root.mkdir()
    database = root / "hpo_data.db"
    database.write_text("database", encoding="utf-8")
    manifest.add_file_checksum(database)
    manifest.save(root / "manifest.json")

    archive = bundle_dir / manifest.get_bundle_filename()
    with tarfile.open(archive, "w:gz") as tar:
        for child in root.iterdir():
            tar.add(child, arcname=child.name)
    return archive


def _write_release_archives(bundle_dir: Path, spec: DataReleaseSpec) -> list[Path]:
    common = {
        "hpo_version": spec.hpo_version,
        "hpo_release_date": spec.hpo_release_date,
        "hpo_source_url": spec.hpo_source_url,
        "hpo_source_sha256": spec.hpo_sha256,
        "active_terms": spec.active_terms,
        "source_commit": spec.source_commit,
        "lockfile_sha256": spec.lockfile_sha256,
        "phentrieve_version": spec.phentrieve_version,
    }
    archives = [_write_archive(bundle_dir, BundleManifest(**common))]
    model = spec.models[0]
    for multi_vector in (False, True):
        archives.append(
            _write_archive(
                bundle_dir,
                BundleManifest(
                    **common,
                    model=EmbeddingModelInfo(
                        name=model.name,
                        slug=model.slug,
                        dimension=3,
                        multi_vector=multi_vector,
                        revision=model.revision,
                        trust_remote_code=model.trust_remote_code,
                    ),
                ),
            )
        )
    (bundle_dir / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in archives
        ),
        encoding="utf-8",
    )
    return archives


def test_expected_bundle_names_cover_minimal_and_both_vector_modes(spec):
    assert expected_bundle_names(spec) == {
        "phentrieve-data-v2026-06-23-minimal.tar.gz",
        "phentrieve-data-v2026-06-23-example.tar.gz",
        "phentrieve-data-v2026-06-23-example-multivec.tar.gz",
    }


def test_verify_release_accepts_a_complete_contract(mocker, tmp_path, spec):
    _write_release_archives(tmp_path, spec)
    mocker.patch("verify_data_release._verify_archive_index")

    report = verify_release(spec, tmp_path)

    assert report["release_tag"] == spec.release_tag
    assert report["archive_count"] == 3


def test_verify_release_rejects_missing_archive(tmp_path, spec):
    _write_release_archives(tmp_path, spec)
    (tmp_path / "phentrieve-data-v2026-06-23-example-multivec.tar.gz").unlink()

    with pytest.raises(ValueError, match="Missing expected archives"):
        verify_release(spec, tmp_path)


def test_verify_release_rejects_bad_checksum(mocker, tmp_path, spec):
    _write_release_archives(tmp_path, spec)
    mocker.patch("verify_data_release._verify_archive_index")
    checksum_file = tmp_path / "SHA256SUMS"
    checksum_file.write_text(
        ("0" * 64 + "  phentrieve-data-v2026-06-23-minimal.tar.gz\n")
        + "\n".join(
            line
            for line in checksum_file.read_text(encoding="utf-8").splitlines()
            if not line.endswith("phentrieve-data-v2026-06-23-minimal.tar.gz")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_release(spec, tmp_path)
