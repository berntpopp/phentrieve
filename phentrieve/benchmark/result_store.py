"""Storage primitives for reproducible benchmark run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
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
    preexisting: bool


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


def _path_is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _assert_no_links_below(root: Path, path: Path) -> None:
    """Reject symlink or junction components at and below a storage root."""
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    if not path_absolute.is_relative_to(root_absolute):
        raise ValueError("Benchmark storage path escapes the configured results root")
    current = root_absolute
    if _path_is_link(current):
        raise ValueError(
            f"Benchmark storage path must not be a link or junction: {current}"
        )
    for part in path_absolute.relative_to(root_absolute).parts:
        current = current / part
        if _path_is_link(current):
            raise ValueError(
                f"Benchmark storage path must not be a link or junction: {current}"
            )


def _reset_run_dir(run_dir: Path, *, results_root: Path) -> None:
    """Delete a previous run's artifacts while keeping its resume checkpoint.

    Without this, an overwritten run inherits files it never produced and
    ``write_manifest`` registers them as its own, because it decides purely on
    ``Path.exists()``. The checkpoint survives because the LLM benchmark reuses
    an existing run directory precisely to resume from it; its identity is
    validated separately before any of it is trusted.
    """
    _assert_no_links_below(results_root, run_dir)
    for item in run_dir.iterdir():
        if item.name in {CHECKPOINT_FILENAME, "manifest.json", ".generations"}:
            continue
        if _path_is_link(item):
            raise ValueError(
                f"Benchmark run contents must not be links or junctions: {item}"
            )
        if item.is_dir():
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
    reset_existing: bool = True,
    materialize: bool = True,
) -> RunLayout:
    """Create a unique run directory below a result root.

    ``reset_existing=False`` lets callers validate a preserved checkpoint before
    explicitly clearing artifacts with :func:`reset_run_artifacts`.
    """
    requested_run_id = (
        re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_") if run_id else utc_run_id()
    )
    requested_run_id = requested_run_id or "run"
    parent = results_root / benchmark_type / safe_slug(dataset) / safe_slug(model)
    selected_run_id = requested_run_id
    run_dir = parent / selected_run_id
    _assert_no_links_below(results_root, run_dir)

    if run_dir.exists() and not overwrite:
        if exact_run_id:
            raise FileExistsError(f"Benchmark run already exists: {run_dir}")
        suffix = 2
        while run_dir.exists():
            selected_run_id = f"{requested_run_id}-{suffix}"
            run_dir = parent / selected_run_id
            suffix += 1

    preexisting = run_dir.exists()
    if preexisting and overwrite and reset_existing:
        _reset_run_dir(run_dir, results_root=results_root)

    if materialize:
        run_dir.mkdir(parents=True, exist_ok=overwrite)
    legacy_dir = run_dir / "legacy"
    if materialize:
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
        preexisting=preexisting,
    )


def reset_run_artifacts(layout: RunLayout) -> None:
    """Clear a validated reusable run while preserving its checkpoint."""
    if not layout.run_dir.exists():
        return
    _reset_run_dir(layout.run_dir, results_root=layout.results_root)
    layout.legacy_dir.mkdir(parents=True, exist_ok=True)


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
    reserved = {
        "schema_version",
        "run_id",
        "benchmark_type",
        "dataset_name",
        "model",
        "artifacts",
    }
    conflicts = reserved.intersection(identities)
    if conflicts:
        raise ValueError(
            f"Metadata contains reserved manifest keys: {sorted(conflicts)}"
        )
    artifacts: dict[str, dict[str, str]] = {}
    _assert_no_links_below(layout.results_root, layout.run_dir)
    run_root = layout.run_dir.resolve()
    seen: set[Path] = set()
    items = list(inventory)
    singleton_roles = {
        "summary",
        "checkpoint",
        "metrics",
        "term_results",
        "case_results",
        "chunk_diagnostics",
    }
    role_counts = Counter(item.role for item in items)
    duplicate_roles = sorted(role for role in singleton_roles if role_counts[role] > 1)
    if duplicate_roles:
        raise ValueError(f"Duplicate singleton artifact role: {duplicate_roles}")
    validated: list[tuple[ArtifactEntry, Path, str]] = []
    for item in items:
        lexical_path = item.path.absolute()
        if not lexical_path.is_relative_to(layout.run_dir.absolute()):
            raise ValueError(
                "Artifact inventory paths must be inside the run directory"
            )
        _assert_no_links_below(layout.run_dir, lexical_path)
        path = lexical_path.resolve()
        if not path.is_relative_to(run_root):
            raise ValueError(
                "Artifact inventory paths must be inside the run directory"
            )
        if path in seen:
            raise ValueError("Artifact inventory paths must be unique")
        if not path.is_file():
            raise ValueError(f"Artifact inventory path is not a file: {path}")
        seen.add(path)
        relative = lexical_path.relative_to(layout.run_dir.absolute()).as_posix()
        validated.append((item, path, relative))

    generation_id = f"{utc_run_id()}-{uuid.uuid4().hex[:8]}"
    generations_root = layout.run_dir / ".generations"
    _assert_no_links_below(layout.run_dir, generations_root)
    if _path_is_link(layout.manifest_path):
        raise ValueError("Benchmark manifest path must not be a link or junction")
    staging_dir = generations_root / f".{generation_id}.tmp"
    generation_dir = generations_root / generation_id
    try:
        staging_dir.mkdir(parents=True)
        for _item, source, relative in validated:
            destination = staging_dir.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        staging_dir.replace(generation_dir)
    except Exception:
        if (
            staging_dir.exists()
            and staging_dir.is_dir()
            and not _path_is_link(staging_dir)
        ):
            shutil.rmtree(staging_dir)
        raise

    for item, _source, source_relative in validated:
        relative = (Path(".generations") / generation_id / source_relative).as_posix()
        path = layout.run_dir / relative
        key = item.role if item.role == "summary" else f"{item.role}:{source_relative}"
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
            (layout.run_dir / artifacts[f"{item.role}:{source_relative}"]["path"])
            .parent.relative_to(run_root)
            .as_posix()
            for item, _source, source_relative in validated
            if item.role == role
        }
        if len(parents) == 1:
            artifacts[compatibility_role] = {
                "role": compatibility_role,
                "path": parents.pop(),
                "media_type": "inode/directory",
            }
    metric = next(
        (entry for entry in artifacts.values() if entry.get("role") == "metrics"), None
    )
    if metric is not None:
        artifacts["metrics"] = dict(metric)
    for role in singleton_roles - {"summary", "metrics"}:
        entry = next(
            (item for item in artifacts.values() if item.get("role") == role), None
        )
        if entry is not None:
            artifacts[role] = dict(entry)
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


def _verified_manifest_artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    artifact = artifacts.get(role)
    if not isinstance(artifact, dict):
        artifact = next(
            (
                entry
                for entry in artifacts.values()
                if isinstance(entry, dict) and entry.get("role") == role
            ),
            None,
        )
    if not isinstance(artifact, dict):
        return None
    relative_path = artifact.get("path")
    if not isinstance(relative_path, str):
        return None
    portable = PurePosixPath(relative_path)
    if (
        "\\" in relative_path
        or portable.is_absolute()
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        return None
    run_dir = manifest_path.parent.resolve()
    lexical_artifact = manifest_path.parent.joinpath(*portable.parts)
    _assert_no_links_below(manifest_path.parent, lexical_artifact)
    artifact_path = lexical_artifact.resolve()
    if not artifact_path.is_relative_to(run_dir) or not artifact_path.is_file():
        return None
    if manifest.get("schema_version") == 2:
        expected_sha256 = artifact.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256_file(artifact_path) != expected_sha256
        ):
            return None
    return artifact_path


def active_checkpoint_path(layout: RunLayout) -> Path:
    """Resolve the committed checkpoint, with fixed-root legacy fallback."""
    if not layout.manifest_path.exists():
        return layout.checkpoint_path
    _assert_no_links_below(layout.results_root, layout.manifest_path)
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Existing benchmark manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Existing benchmark manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        return layout.checkpoint_path
    if schema_version != 2:
        raise ValueError("Existing benchmark manifest has an unsupported schema")
    checkpoint = _verified_manifest_artifact_path(
        layout.manifest_path, manifest, "checkpoint"
    )
    if checkpoint is None:
        raise ValueError(
            "Existing benchmark manifest has no valid committed checkpoint"
        )
    return checkpoint


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
    root_resolved = root.resolve()
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            _assert_no_links_below(root, manifest_path)
            resolved_manifest = manifest_path.resolve()
            if not resolved_manifest.is_relative_to(root_resolved):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
            schema_version = manifest.get("schema_version")
            if schema_version not in {1, 2}:
                continue
            if (
                benchmark_type is not None
                and manifest.get("benchmark_type") != benchmark_type
            ):
                continue
            artifact_path = _verified_manifest_artifact_path(
                manifest_path, manifest, role
            )
            if artifact_path is not None:
                canonical.append(artifact_path)
        except (OSError, ValueError, TypeError):
            continue
    if canonical:
        return canonical
    if role == "summary":
        legacy: list[Path] = []
        for path in root.rglob("*_summary.json"):
            try:
                _assert_no_links_below(root, path)
                resolved = path.resolve()
                if resolved.is_relative_to(root_resolved) and resolved.is_file():
                    legacy.append(resolved)
            except (OSError, ValueError):
                continue
        return sorted(legacy)
    return []
