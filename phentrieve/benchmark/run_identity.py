"""Stable identities for benchmark datasets and retrieval assets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import urlsplit, urlunsplit

from phentrieve.benchmark.data_loader import (
    DEFAULT_SIMPLE_ASSERTION,
    DIRECTORY_BENCHMARK_DATASETS,
    load_benchmark_data,
    parse_gold_terms,
)
from phentrieve.llm.prompts.identity import PromptBundleIdentity

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


@dataclass(frozen=True)
class DatasetIdentity:
    """Hashes identifying the source, inputs, and gold labels of a dataset."""

    source_sha256: str
    input_sha256: str
    gold_sha256: str
    document_ids_sha256: str
    projection: str
    excluded_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalAssetIdentity:
    """Identity of the installed retrieval bundle used by an evaluation."""

    asset_type: str
    embedding_model: str
    hpo_version: str
    manifest_sha256: str


@dataclass(frozen=True)
class RunFingerprints:
    """Hashes separating inference execution from scoring semantics."""

    execution_sha256: str
    scoring_sha256: str


def build_run_fingerprints(
    dataset: DatasetIdentity,
    prompt: PromptBundleIdentity,
    model: Mapping[str, JSONValue],
    asset: RetrievalAssetIdentity,
) -> RunFingerprints:
    """Build stable identities for execution inputs and scoring semantics."""
    model_payload = dict(model)
    _validate_json_value(model_payload, path="$")

    execution_payload = {
        "input_sha256": dataset.input_sha256,
        "document_ids_sha256": dataset.document_ids_sha256,
        "prompt": asdict(prompt),
        "model": model_payload,
        "asset": asdict(asset),
        "evaluation_hpo_version": asset.hpo_version,
    }
    scoring_payload = {
        "gold_sha256": dataset.gold_sha256,
        "document_ids_sha256": dataset.document_ids_sha256,
        "projection": dataset.projection,
    }
    return RunFingerprints(
        execution_sha256=_canonical_sha256(execution_payload),
        scoring_sha256=_canonical_sha256(scoring_payload),
    )


def build_dataset_identity(
    test_path: Path,
    dataset: str,
    document_ids: Sequence[str] | None = None,
) -> DatasetIdentity:
    """Build semantic identities after dataset projection and document selection."""
    payload = load_benchmark_data(test_path, dataset=dataset)
    documents = payload["documents"]
    loaded_id_list = [str(document["id"]) for document in documents]
    _reject_duplicate_ids(loaded_id_list, source="loaded")
    available_ids = set(loaded_id_list)

    if document_ids is None:
        selected_ids = available_ids
    else:
        requested_id_list = [str(document_id) for document_id in document_ids]
        _reject_duplicate_ids(requested_id_list, source="requested")
        selected_ids = set(requested_id_list)
        unknown_ids = selected_ids - available_ids
        if unknown_ids:
            missing = ", ".join(sorted(unknown_ids))
            raise ValueError(f"Unknown requested document IDs: {missing}")

    selected_documents = [
        document for document in documents if str(document["id"]) in selected_ids
    ]
    input_records: list[dict[str, str]] = [
        {"id": str(document["id"]), "text": str(document["text"])}
        for document in selected_documents
    ]
    input_records.sort(key=lambda record: record["id"])
    gold_records: list[dict[str, Any]] = [
        {
            "id": str(document["id"]),
            "hpo_ids": sorted(
                {
                    hpo_id
                    for hpo_id, assertion in parse_gold_terms(
                        document.get("gold_hpo_terms", [])
                    )
                    if assertion == DEFAULT_SIMPLE_ASSERTION
                }
            ),
        }
        for document in selected_documents
    ]
    gold_records.sort(key=lambda record: str(record["id"]))
    evaluated_ids = sorted(selected_ids)

    return DatasetIdentity(
        source_sha256=_source_sha256(test_path, dataset),
        input_sha256=_canonical_sha256(input_records),
        gold_sha256=_canonical_sha256(gold_records),
        document_ids_sha256=_canonical_sha256(evaluated_ids),
        projection="positive_hpo_present_v1",
        excluded_document_ids=tuple(sorted(available_ids - selected_ids)),
    )


def load_retrieval_asset_identity(
    data_dir: Path | None = None,
) -> RetrievalAssetIdentity:
    """Read the identity of the installed retrieval bundle."""
    from phentrieve.data_processing.bundle_downloader import (
        get_installed_bundle_info,
    )

    manifest = get_installed_bundle_info(data_dir)
    if manifest is None:
        raise ValueError("No installed retrieval bundle manifest was found.")
    if manifest.model is None:
        raise ValueError("Installed retrieval bundle manifest has no embedding model.")
    if not isinstance(manifest.hpo_version, str) or not manifest.hpo_version.strip():
        raise ValueError(
            "Installed retrieval bundle manifest has no valid HPO version provenance: "
            "expected non-empty 'hpo_version'."
        )

    manifest_path = (data_dir or _default_data_dir()) / "manifest.json"
    return RetrievalAssetIdentity(
        asset_type="multi_vector" if manifest.model.multi_vector else "single_vector",
        embedding_model=manifest.model.name,
        hpo_version=manifest.hpo_version,
        manifest_sha256=_sha256_file(manifest_path),
    )


def validate_evaluation_hpo_version(
    evaluation_hpo_version: str,
    asset: RetrievalAssetIdentity,
) -> None:
    """Reject evaluation and retrieval assets built from different HPO versions."""
    if not isinstance(asset.hpo_version, str) or not asset.hpo_version.strip():
        raise ValueError("Retrieval asset HPO version provenance must be non-empty.")
    if (
        not isinstance(evaluation_hpo_version, str)
        or not evaluation_hpo_version.strip()
    ):
        raise ValueError("Evaluation HPO version must be non-empty.")
    if evaluation_hpo_version != asset.hpo_version:
        raise ValueError(
            f"Evaluation HPO version {evaluation_hpo_version!r} does not match "
            f"retrieval asset HPO version {asset.hpo_version!r}."
        )


def sanitize_behavioral_base_url(value: str | None) -> str | None:
    """Keep endpoint behavior while removing credentials, query, and fragment."""
    if value is None or not value.strip():
        return None
    raw = value.strip()
    has_scheme = "://" in raw
    parsed = urlsplit(raw if has_scheme else f"//{raw}")
    if parsed.hostname is None:
        return None
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    if has_scheme:
        return urlunsplit((parsed.scheme.lower(), f"{host}{port}", parsed.path, "", ""))
    return f"//{host}{port}{parsed.path}"


def _canonical_sha256(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"Model configuration number at {path} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Model configuration key at {path} must be a string; "
                    f"got {type(key).__name__}"
                )
            _validate_json_value(item, path=_json_child_path(path, key))
        return
    raise TypeError(
        f"Model configuration value at {path} must be JSON-compatible; "
        f"got {type(value).__name__}"
    )


def _json_child_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _reject_duplicate_ids(document_ids: Sequence[str], *, source: str) -> None:
    duplicates = sorted(
        document_id for document_id, count in Counter(document_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate {source} document IDs: {', '.join(duplicates)}")


def _source_sha256(test_path: Path, dataset: str) -> str:
    if test_path.is_file():
        return _sha256_file(test_path)
    selected_datasets = DIRECTORY_BENCHMARK_DATASETS if dataset == "all" else (dataset,)
    selected_files = (
        file_path
        for dataset_name in selected_datasets
        for file_path in (test_path / dataset_name / "annotations").glob("*.json")
    )
    return _sha256_files(test_path, selected_files)


def _sha256_files(root: Path, files: Iterable[Path]) -> str:
    selected_files = sorted(files)
    hasher = hashlib.sha256()
    hasher.update(len(selected_files).to_bytes(8, "big"))
    for file_path in selected_files:
        relative_path = file_path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(len(relative_path).to_bytes(8, "big"))
        hasher.update(relative_path)
        hasher.update(file_path.stat().st_size.to_bytes(8, "big"))
        hasher.update(bytes.fromhex(_sha256_file(file_path)))
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _default_data_dir() -> Path:
    from phentrieve.utils import get_default_data_dir

    return get_default_data_dir()
