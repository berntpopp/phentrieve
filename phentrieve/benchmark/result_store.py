"""Storage primitives for reproducible benchmark run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

BenchmarkType = Literal["retrieval", "extraction"]


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
    legacy_dir: Path


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
        re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_")
        if run_id
        else utc_run_id()
    )
    requested_run_id = requested_run_id or "run"
    parent = (
        results_root
        / benchmark_type
        / safe_slug(dataset)
        / safe_slug(model)
    )
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
    layout: RunLayout, metadata: Mapping[str, Any]
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

    legacy_files = sorted(path for path in layout.legacy_dir.rglob("*") if path.is_file())
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


def discover_artifacts(root: Path, role: str) -> list[Path]:
    """Discover canonical artifacts recursively, with legacy fallback."""
    canonical: list[Path] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = manifest.get("artifacts", {}).get(role)
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
