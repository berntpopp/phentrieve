"""Storage primitives for reproducible benchmark run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

BenchmarkType = Literal["retrieval", "extraction", "llm"]

CHECKPOINT_FILENAME = "checkpoint.json"


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Filesystem locations belonging to one benchmark run."""

    results_root: Path
    benchmark_type: BenchmarkType
    dataset: str
    model: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    summary_path: Path
    terms_path: Path
    cases_path: Path
    chunks_path: Path
    checkpoint_path: Path
    legacy_dir: Path


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """One explicitly owned file in a schema-v2 run inventory."""

    path: Path
    role: str
    media_type: str


def safe_slug(value: str) -> str:
    """Return a stable lowercase path component."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def utc_run_id(now: datetime | None = None) -> str:
    """Return a microsecond-resolution UTC run identifier."""
    instant = now or datetime.now(UTC)
    return instant.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree using relative paths and contents."""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _reset_run_dir(run_dir: Path) -> None:
    """Delete a previous run's artifacts while keeping its resume checkpoint.

    Without this, an overwritten run inherits files it never produced and
    ``write_manifest`` registers them as its own, because it decides purely on
    ``Path.exists()``. The checkpoint survives because the LLM benchmark reuses
    an existing run directory precisely to resume from it; its identity is
    validated separately before any of it is trusted.
    """
    for item in run_dir.iterdir():
        if item.name == CHECKPOINT_FILENAME:
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def create_run_layout(
    results_root: Path,
    benchmark_type: BenchmarkType,
    dataset: str,
    model: str,
    *,
    run_id: str | None = None,
    exact_run_id: bool = False,
    overwrite: bool = False,
) -> RunLayout:
    """Create a unique run directory below a result root."""
    requested_run_id = (
        re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_") if run_id else utc_run_id()
    )
    requested_run_id = requested_run_id or "run"
    parent = results_root / benchmark_type / safe_slug(dataset) / safe_slug(model)
    selected_run_id = requested_run_id
    run_dir = parent / selected_run_id

    if run_dir.exists() and not overwrite:
        if exact_run_id:
            raise FileExistsError(f"Benchmark run already exists: {run_dir}")
        suffix = 2
        while run_dir.exists():
            selected_run_id = f"{requested_run_id}-{suffix}"
            run_dir = parent / selected_run_id
            suffix += 1

    if run_dir.exists() and overwrite:
        _reset_run_dir(run_dir)

    run_dir.mkdir(parents=True, exist_ok=overwrite)
    legacy_dir = run_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        results_root=results_root,
        benchmark_type=benchmark_type,
        dataset=dataset,
        model=model,
        run_id=selected_run_id,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        summary_path=run_dir / "summary.json",
        terms_path=run_dir / "terms.jsonl",
        cases_path=run_dir / "cases.jsonl",
        chunks_path=run_dir / "diagnostics" / "chunks.jsonl",
        checkpoint_path=run_dir / CHECKPOINT_FILENAME,
        legacy_dir=legacy_dir,
    )


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write formatted UTF-8 JSON."""
    _atomic_text_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write one compact JSON object per line."""
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_text_write(path, content)


def _artifact_entry(layout: RunLayout, path: Path, media_type: str) -> dict[str, str]:
    return {
        "path": path.relative_to(layout.run_dir).as_posix(),
        "media_type": media_type,
    }


def write_manifest(
    layout: RunLayout,
    metadata: Mapping[str, Any],
    *,
    extra_artifacts: Mapping[str, tuple[Path, str]] | None = None,
) -> dict[str, Any]:
    """Write a run manifest containing machine-readable artifact roles."""
    artifacts: dict[str, dict[str, str]] = {}
    candidates = (
        ("summary", layout.summary_path, "application/json"),
        ("term_results", layout.terms_path, "application/x-ndjson"),
        ("case_results", layout.cases_path, "application/x-ndjson"),
        ("chunk_diagnostics", layout.chunks_path, "application/x-ndjson"),
    )
    for role, path, media_type in candidates:
        if path.exists():
            artifacts[role] = _artifact_entry(layout, path, media_type)

    for role, (path, media_type) in (extra_artifacts or {}).items():
        if path.exists():
            artifacts[role] = _artifact_entry(layout, path, media_type)

    legacy_files = sorted(
        path for path in layout.legacy_dir.rglob("*") if path.is_file()
    )
    if legacy_files:
        artifacts["legacy_compatibility"] = {
            "path": layout.legacy_dir.relative_to(layout.run_dir).as_posix(),
            "media_type": "inode/directory",
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": layout.run_id,
        "benchmark_type": layout.benchmark_type,
        "dataset_name": layout.dataset,
        "model": layout.model,
        **dict(metadata),
        "artifacts": artifacts,
    }
    write_json(layout.manifest_path, manifest)
    return manifest


def publish_manifest_v2(
    layout: RunLayout,
    identities: Mapping[str, Any],
    inventory: Iterable[ArtifactEntry],
) -> dict[str, Any]:
    """Atomically publish an integrity-checked manifest for an existing run.

    The caller supplies the exact files produced by this execution. This keeps
    stale files in a resumed directory out of the manifest without weakening
    the established run-layout and discovery contracts.
    """
    artifacts: dict[str, dict[str, str]] = {}
    run_root = layout.run_dir.resolve()
    seen: set[Path] = set()
    items = list(inventory)
    for item in items:
        path = item.path.resolve()
        if not path.is_relative_to(run_root):
            raise ValueError("Artifact inventory paths must be inside the run directory")
        if path in seen:
            raise ValueError("Artifact inventory paths must be unique")
        if not path.is_file():
            raise ValueError(f"Artifact inventory path is not a file: {path}")
        seen.add(path)
        relative = path.relative_to(run_root).as_posix()
        key = item.role if item.role == "summary" else f"{item.role}:{relative}"
        artifacts[key] = {
            "role": item.role,
            "path": relative,
            "media_type": item.media_type,
            "sha256": sha256_file(path),
        }
    for role, compatibility_role in (
        ("prediction", "llm_predictions"),
        ("trace", "llm_traces"),
    ):
        parents = {
            item.path.resolve().parent.relative_to(run_root).as_posix()
            for item in items
            if item.role == role
        }
        if len(parents) == 1:
            artifacts[compatibility_role] = {
                "role": compatibility_role,
                "path": parents.pop(),
                "media_type": "inode/directory",
            }
    metric = next((entry for entry in artifacts.values() if entry.get("role") == "metrics"), None)
    if metric is not None:
        artifacts["metrics"] = dict(metric)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_id": layout.run_id,
        "benchmark_type": layout.benchmark_type,
        "dataset_name": layout.dataset,
        "model": layout.model,
        **dict(identities),
        "artifacts": artifacts,
    }
    write_json(layout.manifest_path, manifest)
    return manifest


def discover_artifacts(
    root: Path,
    role: str,
    *,
    benchmark_type: BenchmarkType | None = None,
) -> list[Path]:
    """Discover canonical artifacts recursively, with legacy fallback.

    ``benchmark_type`` restricts discovery to one kind of run. Callers that
    interpret an artifact against a fixed metric schema must set it: every
    benchmark type writes a ``summary`` role, so an unfiltered search of a
    shared result root also returns extraction and LLM summaries. The legacy
    fallback stays unfiltered because the old flat layout only held retrieval
    summaries.
    """
    canonical: list[Path] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                benchmark_type is not None
                and manifest.get("benchmark_type") != benchmark_type
            ):
                continue
            artifact = manifest.get("artifacts", {}).get(role)
            if not isinstance(artifact, dict):
                artifact = next(
                    (
                        entry
                        for entry in manifest.get("artifacts", {}).values()
                        if isinstance(entry, dict) and entry.get("role") == role
                    ),
                    None,
                )
            relative_path = artifact.get("path") if isinstance(artifact, dict) else None
            if not isinstance(relative_path, str):
                continue
            run_dir = manifest_path.parent.resolve()
            artifact_path = (run_dir / relative_path).resolve()
            if artifact_path.is_relative_to(run_dir) and artifact_path.is_file():
                canonical.append(artifact_path)
        except (OSError, ValueError, TypeError):
            continue
    if canonical:
        return canonical
    if role == "summary":
        return sorted(path.resolve() for path in root.rglob("*_summary.json"))
    return []
