#!/usr/bin/env python3
"""Verify the complete, immutable artifact set for an HPO data release."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from phentrieve.data_processing.bundle_manifest import BundleManifest
from phentrieve.data_processing.bundle_packager import (
    _open_collection,
    _validate_collection_provenance,
    extract_bundle,
)
from phentrieve.data_processing.release_contract import (
    DataReleaseSpec,
    ModelReleaseSpec,
)
from phentrieve.embeddings import clear_model_registry, load_embedding_model
from phentrieve.utils import generate_collection_name


def expected_bundle_names(spec: DataReleaseSpec) -> set[str]:
    """Return the exact archive names required by a release specification."""
    names = {f"phentrieve-data-{spec.hpo_version}-minimal.tar.gz"}
    for model in spec.models:
        names.add(f"phentrieve-data-{spec.hpo_version}-{model.slug}.tar.gz")
        names.add(f"phentrieve-data-{spec.hpo_version}-{model.slug}-multivec.tar.gz")
    return names


def _archive_descriptor(
    archive_name: str, spec: DataReleaseSpec
) -> tuple[ModelReleaseSpec | None, bool]:
    minimal_name = f"phentrieve-data-{spec.hpo_version}-minimal.tar.gz"
    if archive_name == minimal_name:
        return None, False
    for model in spec.models:
        if archive_name == f"phentrieve-data-{spec.hpo_version}-{model.slug}.tar.gz":
            return model, False
        if archive_name == (
            f"phentrieve-data-{spec.hpo_version}-{model.slug}-multivec.tar.gz"
        ):
            return model, True
    raise ValueError(f"Unexpected archive name: {archive_name}")


def _load_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        raise ValueError(f"Missing checksum file: {path.name}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, filename = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Invalid checksum entry: {line!r}") from error
        filename = filename.lstrip("*")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 digest for {filename!r}")
        checksums[filename] = digest
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(
    manifest: BundleManifest,
    spec: DataReleaseSpec,
    model: ModelReleaseSpec | None,
    multi_vector: bool,
) -> None:
    for field_name, actual, expected in (
        ("hpo_version", manifest.hpo_version, spec.hpo_version),
        ("hpo_release_date", manifest.hpo_release_date, spec.hpo_release_date),
        ("hpo_source_url", manifest.hpo_source_url, spec.hpo_source_url),
        ("hpo_source_sha256", manifest.hpo_source_sha256, spec.hpo_sha256),
        ("active_terms", manifest.active_terms, spec.active_terms),
        ("source_commit", manifest.source_commit, spec.source_commit),
        ("lockfile_sha256", manifest.lockfile_sha256, spec.lockfile_sha256),
        ("phentrieve_version", manifest.phentrieve_version, spec.phentrieve_version),
    ):
        if actual != expected:
            raise ValueError(
                f"Manifest {field_name} mismatch: expected {expected!r}, got {actual!r}"
            )
    if model is None:
        if manifest.model is not None:
            raise ValueError("Minimal bundle unexpectedly contains model metadata")
        return
    if manifest.model is None:
        raise ValueError("Model bundle is missing model metadata")
    if manifest.model.name != model.name:
        raise ValueError(
            f"Manifest model mismatch: expected {model.name!r}, got {manifest.model.name!r}"
        )
    if manifest.model.slug != model.slug:
        raise ValueError(
            f"Manifest model slug mismatch: expected {model.slug!r}, got {manifest.model.slug!r}"
        )
    if manifest.model.revision != model.revision:
        raise ValueError("Manifest model revision does not match release spec")
    if manifest.model.trust_remote_code != model.trust_remote_code:
        raise ValueError(
            "Manifest model custom-code policy does not match release spec"
        )
    if manifest.model.code_revision != model.code_revision:
        raise ValueError("Manifest model code revision does not match release spec")
    if manifest.model.multi_vector != multi_vector:
        raise ValueError("Manifest vector mode does not match archive name")


def _verify_archive_index(
    extracted_dir: Path,
    manifest: BundleManifest,
    spec: DataReleaseSpec,
    model: ModelReleaseSpec,
    multi_vector: bool,
    smoke_test: bool,
) -> None:
    index_dir = extracted_dir / "indexes"
    if not index_dir.exists():
        raise ValueError("Model bundle is missing indexes/")
    collection_name = generate_collection_name(model.name)
    if multi_vector:
        collection_name = f"{collection_name}_multi"
    with _open_collection(index_dir, collection_name) as collection:
        expected_count = spec.expected_document_count(
            "multi_vector" if multi_vector else "single_vector"
        )
        _validate_collection_provenance(
            collection=collection,
            manifest=manifest,
            model_name=model.name,
            index_type="multi_vector" if multi_vector else "single_vector",
            expected_document_count=expected_count,
            expected_model_revision=model.revision,
        )
        if smoke_test:
            embedding_model = load_embedding_model(
                model_name=model.name,
                revision=model.revision,
                trust_remote_code=model.trust_remote_code,
                code_revision=model.code_revision,
            )
            query_embedding = embedding_model.encode(["phenotypic abnormality"])
            result = cast(Any, collection).query(
                query_embeddings=query_embedding.tolist(),
                n_results=1,
            )
            if not result.get("ids") or not result["ids"][0]:
                raise ValueError(
                    f"Retrieval smoke test returned no result for {model.name}"
                )
            clear_model_registry()


def verify_release(
    spec: DataReleaseSpec,
    bundle_dir: Path,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Verify all release archives against one immutable data-release spec."""
    bundle_dir = Path(bundle_dir)
    expected_names = expected_bundle_names(spec)
    actual_names = {path.name for path in bundle_dir.glob("*.tar.gz")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        raise ValueError(f"Missing expected archives: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected archives: {', '.join(unexpected)}")

    checksums = _load_checksums(bundle_dir / "SHA256SUMS")
    checksum_names = set(checksums)
    if checksum_names != expected_names:
        missing_checksums = sorted(expected_names - checksum_names)
        unexpected_checksums = sorted(checksum_names - expected_names)
        raise ValueError(
            "SHA256SUMS archive set mismatch: "
            f"missing={missing_checksums}, unexpected={unexpected_checksums}"
        )

    archive_reports: list[dict[str, Any]] = []
    for archive_name in sorted(expected_names):
        archive_path = bundle_dir / archive_name
        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != checksums[archive_name]:
            raise ValueError(
                f"Checksum mismatch for {archive_name}: "
                f"expected {checksums[archive_name]}, got {actual_sha256}"
            )
        model, multi_vector = _archive_descriptor(archive_name, spec)
        with tempfile.TemporaryDirectory(prefix="phentrieve-release-") as temp_dir:
            extracted_dir = Path(temp_dir)
            manifest = extract_bundle(archive_path, extracted_dir)
            _verify_manifest(manifest, spec, model, multi_vector)
            if model is not None:
                _verify_archive_index(
                    extracted_dir,
                    manifest,
                    spec,
                    model,
                    multi_vector,
                    smoke_test,
                )
        archive_reports.append(
            {
                "name": archive_name,
                "sha256": actual_sha256,
                "model": model.name if model is not None else None,
                "multi_vector": multi_vector,
            }
        )

    return {
        "release_tag": spec.release_tag,
        "hpo_version": spec.hpo_version,
        "archive_count": len(archive_reports),
        "archives": archive_reports,
        "smoke_test": smoke_test,
        "verified_at": datetime.now(UTC).isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a complete Phentrieve HPO data release."
    )
    parser.add_argument("--spec", type=Path, required=True, help="Release spec JSON")
    parser.add_argument(
        "--bundle-dir", type=Path, required=True, help="Directory containing archives"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one embedding retrieval query for each model archive",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional verification report path (default: bundle-dir/verification-report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = verify_release(
            DataReleaseSpec.load(args.spec),
            args.bundle_dir,
            smoke_test=args.smoke_test,
        )
    except ValueError as error:
        print(f"Verification failed: {error}")
        return 1

    report_path = args.report or args.bundle_dir / "verification-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {report['archive_count']} archives; report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
