#!/usr/bin/env python3
"""Build a complete, reproducible HPO data-release artifact matrix locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from verify_data_release import verify_release

from phentrieve.data_processing.bundle_packager import create_bundle
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.data_processing.hpo_parser import prepare_hpo_data
from phentrieve.data_processing.release_contract import DataReleaseSpec
from phentrieve.embeddings import clear_model_registry
from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source_root: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise ValueError("git executable is required to verify source provenance")
    return subprocess.check_output(  # noqa: S603 - fixed local git command
        [git_executable, "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _validate_build_inputs(spec: DataReleaseSpec, source_root: Path) -> None:
    source_commit = _git_commit(source_root)
    if source_commit != spec.source_commit:
        raise ValueError(
            f"Source commit mismatch: expected {spec.source_commit}, got {source_commit}"
        )
    lockfile_sha256 = _sha256(source_root / "uv.lock")
    if lockfile_sha256 != spec.lockfile_sha256:
        raise ValueError(
            "uv.lock SHA-256 mismatch: "
            f"expected {spec.lockfile_sha256}, got {lockfile_sha256}"
        )


def _prepare_database(spec: DataReleaseSpec, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    success, error, resolved_version = prepare_hpo_data(
        force_update=True,
        hpo_file_path=data_dir / "hp.json",
        db_path=data_dir / "hpo_data.db",
        hpo_version=spec.hpo_version,
        expected_sha256=spec.hpo_sha256,
    )
    if not success:
        raise ValueError(f"HPO preparation failed: {error}")
    if resolved_version != spec.hpo_version:
        raise ValueError(
            f"Prepared unexpected HPO version: expected {spec.hpo_version}, got {resolved_version}"
        )

    with HPODatabase(data_dir / "hpo_data.db") as database:
        active_terms = database.get_metadata("active_terms_count")
        obsolete_terms = database.get_metadata("obsolete_terms_filtered")
        release_date = database.get_metadata("hpo_release_date")
        source_sha256 = database.get_metadata("hpo_source_sha256")
    if active_terms != str(spec.active_terms):
        raise ValueError(
            f"Active-term count mismatch: expected {spec.active_terms}, got {active_terms}"
        )
    if obsolete_terms is None or int(obsolete_terms) < 0:
        raise ValueError(f"Invalid obsolete-term count: {obsolete_terms!r}")
    if release_date != spec.hpo_release_date:
        raise ValueError(
            f"HPO release date mismatch: expected {spec.hpo_release_date}, got {release_date}"
        )
    if source_sha256 != spec.hpo_sha256:
        raise ValueError("Prepared HPO source digest does not match the release spec")


def _build_model_bundle(
    spec: DataReleaseSpec,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    model_revision: str,
    trust_remote_code: bool,
    code_revision: str | None,
    multi_vector: bool,
    batch_size: int,
    device: str | None,
) -> Path:
    indexes_dir = data_dir / "indexes"
    shutil.rmtree(indexes_dir, ignore_errors=True)
    index_type = "multi-vector" if multi_vector else "single-vector"
    success = orchestrate_index_building(
        model_name_arg=model_name,
        batch_size=batch_size,
        device_override=device,
        recreate=True,
        index_dir_override=str(indexes_dir),
        data_dir_override=str(data_dir),
        multi_vector=multi_vector,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        code_revision=code_revision,
    )
    if not success and device != "cpu" and batch_size > 128:
        clear_model_registry()
        shutil.rmtree(indexes_dir, ignore_errors=True)
        success = orchestrate_index_building(
            model_name_arg=model_name,
            batch_size=128,
            device_override=device,
            recreate=True,
            index_dir_override=str(indexes_dir),
            data_dir_override=str(data_dir),
            multi_vector=multi_vector,
            model_revision=model_revision,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
        )
    if not success:
        raise ValueError(f"Failed to build {index_type} index for {model_name}")

    bundle = create_bundle(
        output_dir=output_dir,
        model_name=model_name,
        data_dir=data_dir,
        multi_vector=multi_vector,
        release_spec=spec,
    )
    clear_model_registry()
    shutil.rmtree(indexes_dir, ignore_errors=True)
    return bundle


def build_release(
    spec: DataReleaseSpec,
    source_root: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, Any]:
    """Build and verify every archive described by a release spec."""
    _validate_build_inputs(spec, source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_database(spec, data_dir)

    archives = [
        create_bundle(
            output_dir=output_dir,
            model_name=None,
            data_dir=data_dir,
            release_spec=spec,
        )
    ]
    for model in spec.models:
        for multi_vector in (False, True):
            archives.append(
                _build_model_bundle(
                    spec=spec,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    model_name=model.name,
                    model_revision=model.revision,
                    trust_remote_code=model.trust_remote_code,
                    code_revision=model.code_revision,
                    multi_vector=multi_vector,
                    batch_size=batch_size,
                    device=device,
                )
            )

    checksum_lines = [
        f"{_sha256(archive)}  {archive.name}\n" for archive in sorted(archives)
    ]
    (output_dir / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    release_manifest = {
        "release": spec.to_dict(),
        "archives": [
            {
                "name": archive.name,
                "sha256": _sha256(archive),
                "size_bytes": archive.stat().st_size,
            }
            for archive in sorted(archives)
        ],
    }
    (output_dir / "release-manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification_report = verify_release(spec, output_dir)
    (output_dir / "verification-report.json").write_text(
        json.dumps(verification_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete local matrix for a Phentrieve HPO data release."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        help="Embedding device override (for example cuda or cpu). Default: auto.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_root = Path(__file__).resolve().parents[1]
    try:
        result = build_release(
            spec=DataReleaseSpec.load(args.spec),
            source_root=source_root,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
    except ValueError as error:
        print(f"Build failed: {error}")
        return 1
    print(f"Built {len(result['archives'])} archives in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
