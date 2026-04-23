# Phenopacket V2 And Annotation Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default Phenopacket export standards-compatible, add a normalized export layer, and introduce an optional linked annotation sidecar for richer full-text and LLM provenance.

**Architecture:** Keep one strict Phenopacket v2 export path as the default artifact, and add a second optional sidecar projection fed from a normalized export record so the serializer no longer depends on ad hoc dict shapes. Use the existing Python Phenopacket bindings for conformance checks, preserve transitional descriptive strings until the sidecar exists, and validate the sidecar against a checked-in JSON Schema.

**Tech Stack:** Python 3.11, phenopackets Python bindings, protobuf JSON parsing, Pydantic, Typer, pytest, jsonschema, Ruff, mypy

---

## File Structure

- `phentrieve/phenopackets/utils.py`
  Responsible for Phenopacket export orchestration, current helper behavior, and final serialization.
- `phentrieve/phenopackets/export_models.py`
  New module for normalized export dataclasses shared by strict Phenopacket and sidecar projections.
- `phentrieve/phenopackets/sidecar.py`
  New module for sidecar JSON generation and sidecar-specific validation helpers.
- `phentrieve/phenopackets/schemas/phenotype_annotation_bundle_v1.schema.json`
  Checked-in JSON Schema for the sidecar contract.
- `phentrieve/cli/text_commands.py`
  Existing CLI Phenopacket output path for full-text/chunked results; will gain optional sidecar export plumbing.
- `phentrieve/cli/llm_commands.py`
  If present on this branch/repo state, not in current main; out of scope unless merged separately.
- `tests/unit/phenopacket_utils/test_phenopacket_utils.py`
  Existing strict Phenopacket exporter tests; needs standards and sidecar coverage.
- `tests/unit/cli/test_text_commands.py`
  CLI behavior tests for phenopacket output and optional paired artifact export.

## Task 1: Introduce Normalized Export Models

**Files:**
- Create: `phentrieve/phenopackets/export_models.py`
- Modify: `phentrieve/phenopackets/__init__.py`
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Write the failing tests for normalized export records**

```python
from phentrieve.phenopackets.export_models import (
    NormalizedPhenotypeExportRecord,
    NormalizedSpan,
)


def test_normalized_span_requires_non_negative_offsets() -> None:
    span = NormalizedSpan(start_char=10, end_char=20, text="seizures")

    assert span.start_char == 10
    assert span.end_char == 20
    assert span.text == "seizures"


def test_normalized_phenotype_export_record_keeps_optional_provenance() -> None:
    record = NormalizedPhenotypeExportRecord(
        hpo_id="HP:0001250",
        label="Seizure",
        assertion="affirmed",
        confidence=0.91,
        evidence_text="recurrent seizures",
        spans=[NormalizedSpan(start_char=10, end_char=28, text="recurrent seizures")],
        chunk_refs=[4],
        source_mode="two_phase",
        match_method="llm_mapping",
    )

    assert record.assertion == "affirmed"
    assert record.confidence == 0.91
    assert record.chunk_refs == [4]
    assert record.source_mode == "two_phase"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "normalized_span_requires_non_negative_offsets or normalized_phenotype_export_record_keeps_optional_provenance" -v
```

Expected:

```text
ImportError: cannot import name 'NormalizedPhenotypeExportRecord'
```

- [ ] **Step 3: Add minimal normalized export dataclasses**

```python
# phentrieve/phenopackets/export_models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AssertionValue = Literal[
    "affirmed", "negated", "uncertain", "family_history", "unknown"
]


@dataclass(frozen=True)
class NormalizedSpan:
    start_char: int
    end_char: int
    text: str | None = None

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")


@dataclass(frozen=True)
class NormalizedPhenotypeExportRecord:
    hpo_id: str
    label: str
    assertion: AssertionValue
    confidence: float | None = None
    evidence_text: str | None = None
    spans: list[NormalizedSpan] = field(default_factory=list)
    chunk_refs: list[int] = field(default_factory=list)
    source_mode: str | None = None
    match_method: str | None = None
```

```python
# phentrieve/phenopackets/__init__.py
from phentrieve.phenopackets.export_models import (
    NormalizedPhenotypeExportRecord,
    NormalizedSpan,
)

__all__ = ["NormalizedPhenotypeExportRecord", "NormalizedSpan"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "normalized_span_requires_non_negative_offsets or normalized_phenotype_export_record_keeps_optional_provenance" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/phenopackets/export_models.py phentrieve/phenopackets/__init__.py tests/unit/phenopacket_utils/test_phenopacket_utils.py
git commit -m "feat: add normalized phenopacket export models"
```

## Task 2: Normalize Legacy Export Inputs Before Serialization

**Files:**
- Modify: `phentrieve/phenopackets/utils.py`
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Write the failing tests for divergent input contracts**

```python
def test_normalize_aggregated_results_accepts_legacy_id_name_confidence_keys() -> None:
    records = _normalize_aggregated_results(
        [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "confidence": 0.9,
                "rank": 1,
            }
        ]
    )

    assert records[0].hpo_id == "HP:0001250"
    assert records[0].label == "Seizure"
    assert records[0].confidence == 0.9


def test_normalize_aggregated_results_accepts_llm_style_hpo_id_term_name_score_keys() -> None:
    records = _normalize_aggregated_results(
        [
            {
                "hpo_id": "HP:0001250",
                "term_name": "Seizure",
                "score": 0.8,
                "assertion": "affirmed",
                "evidence_text": "recurrent seizures",
            }
        ]
    )

    assert records[0].hpo_id == "HP:0001250"
    assert records[0].label == "Seizure"
    assert records[0].assertion == "affirmed"
    assert records[0].evidence_text == "recurrent seizures"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "normalize_aggregated_results_accepts" -v
```

Expected:

```text
NameError: name '_normalize_aggregated_results' is not defined
```

- [ ] **Step 3: Implement normalization helpers in `utils.py`**

```python
from phentrieve.phenopackets.export_models import (
    NormalizedPhenotypeExportRecord,
    NormalizedSpan,
)


def _normalize_aggregated_results(
    aggregated_results: list[dict[str, Any]],
) -> list[NormalizedPhenotypeExportRecord]:
    normalized: list[NormalizedPhenotypeExportRecord] = []
    for item in aggregated_results:
        hpo_id = item.get("id") or item.get("hpo_id")
        label = item.get("name") or item.get("term_name")
        if not hpo_id or not label:
            raise ValueError("Aggregated result is missing an HPO identifier or label")

        assertion = str(item.get("assertion") or item.get("assertion_status") or "affirmed")
        confidence = item.get("confidence")
        if confidence is None:
            confidence = item.get("score")

        normalized.append(
            NormalizedPhenotypeExportRecord(
                hpo_id=hpo_id,
                label=label,
                assertion=_normalize_assertion_value(assertion),
                confidence=float(confidence) if confidence is not None else None,
                evidence_text=item.get("evidence_text"),
            )
        )
    return normalized


def _normalize_assertion_value(value: str | None) -> str:
    normalized = (value or "unknown").strip().lower()
    aliases = {
        "present": "affirmed",
        "affirmed": "affirmed",
        "negated": "negated",
        "absent": "negated",
        "uncertain": "uncertain",
        "suspected": "uncertain",
        "family_history": "family_history",
        "family history": "family_history",
        "unknown": "unknown",
    }
    return aliases.get(normalized, "unknown")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "normalize_aggregated_results_accepts" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/phenopackets/utils.py tests/unit/phenopacket_utils/test_phenopacket_utils.py
git commit -m "refactor: normalize phenopacket export inputs"
```

## Task 3: Correct Strict Phenopacket Export Semantics

**Files:**
- Modify: `phentrieve/phenopackets/utils.py`
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Write the failing standards-focused tests**

```python
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket


def test_phenopacket_export_uses_verified_v2_schema_string() -> None:
    phenopacket_json = format_as_phenopacket_v2(
        aggregated_results=[
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
        ]
    )
    phenopacket = json.loads(phenopacket_json)

    assert phenopacket["metaData"]["phenopacketSchemaVersion"] in {"2.0", "2.0.2", "2.0.0"}


def test_phenopacket_export_round_trips_through_protobuf_parser() -> None:
    phenopacket_json = format_as_phenopacket_v2(
        aggregated_results=[
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
        ]
    )

    packet = Phenopacket()
    Parse(phenopacket_json, packet, ignore_unknown_fields=False)

    assert packet.id


def test_negated_assertion_maps_to_excluded_true() -> None:
    phenopacket_json = format_as_phenopacket_v2(
        aggregated_results=[
            {
                "hpo_id": "HP:0001324",
                "term_name": "Muscle weakness",
                "assertion": "negated",
                "score": 0.8,
            }
        ]
    )
    phenopacket = json.loads(phenopacket_json)

    assert phenopacket["phenotypicFeatures"][0]["excluded"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "verified_v2_schema_string or round_trips_through_protobuf_parser or negated_assertion_maps_to_excluded_true" -v
```

Expected:

```text
FAIL for schema version mismatch or missing negated handling in aggregated path
```

- [ ] **Step 3: Implement strict-export fixes without removing transitional descriptions**

```python
# In phentrieve/phenopackets/utils.py
VERIFIED_PHENOPACKET_SCHEMA_VERSION = "2.0"


def _build_phenotypic_feature(
    record: NormalizedPhenotypeExportRecord,
    *,
    description_parts: list[str] | None = None,
) -> PhenotypicFeature:
    feature_type = OntologyClass(id=record.hpo_id, label=record.label)
    external_reference = None
    if description_parts:
        external_reference = ExternalReference(
            id="phentrieve",
            description=" | ".join(description_parts),
        )
    evidence = [
        Evidence(
            evidence_code=OntologyClass(
                id="ECO:0007636",
                label="computational evidence used in automatic assertion",
            ),
            reference=external_reference,
        )
    ]
    excluded = record.assertion == "negated"
    return PhenotypicFeature(type=feature_type, excluded=excluded, evidence=evidence)
```

```python
# In _create_phenopacket_json
meta_data = MetaData(
    ...,
    phenopacket_schema_version=VERIFIED_PHENOPACKET_SCHEMA_VERSION,
)
```

Keep transitional `description` strings in place for now, but ensure they are fed
from normalized records instead of ad hoc field scraping.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "verified_v2_schema_string or round_trips_through_protobuf_parser or negated_assertion_maps_to_excluded_true" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/phenopackets/utils.py tests/unit/phenopacket_utils/test_phenopacket_utils.py
git commit -m "fix: align phenopacket export with strict v2 semantics"
```

## Task 4: Add Sidecar Schema And Serialization Helpers

**Files:**
- Create: `phentrieve/phenopackets/schemas/phenotype_annotation_bundle_v1.schema.json`
- Create: `phentrieve/phenopackets/sidecar.py`
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Write the failing tests for sidecar generation and schema validation**

```python
from phentrieve.phenopackets.sidecar import (
    build_annotation_sidecar,
    load_annotation_sidecar_schema,
    validate_annotation_sidecar,
)


def test_build_annotation_sidecar_uses_feature_indexes() -> None:
    records = [
        NormalizedPhenotypeExportRecord(
            hpo_id="HP:0001250",
            label="Seizure",
            assertion="affirmed",
            confidence=0.91,
            evidence_text="recurrent seizures",
            spans=[NormalizedSpan(start_char=10, end_char=28, text="recurrent seizures")],
        )
    ]

    sidecar = build_annotation_sidecar(
        phenopacket_id="packet-1",
        records=records,
        generated_by_version="0.16.0",
    )

    assert sidecar["phenopacket_id"] == "packet-1"
    assert sidecar["annotations"][0]["phenotypic_feature_index"] == 0


def test_annotation_sidecar_validates_against_checked_in_schema() -> None:
    records = [
        NormalizedPhenotypeExportRecord(
            hpo_id="HP:0001250",
            label="Seizure",
            assertion="affirmed",
        )
    ]
    sidecar = build_annotation_sidecar(
        phenopacket_id="packet-1",
        records=records,
        generated_by_version="0.16.0",
    )

    validate_annotation_sidecar(sidecar)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "annotation_sidecar" -v
```

Expected:

```text
ImportError: No module named 'phentrieve.phenopackets.sidecar'
```

- [ ] **Step 3: Add the JSON Schema and sidecar builder**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://phentrieve.org/schemas/phenotype_annotation_bundle_v1.schema.json",
  "type": "object",
  "required": ["schema_version", "artifact_type", "generated_by", "phenopacket_id", "annotations"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "artifact_type": {"const": "phenotype_annotation_bundle"},
    "generated_by": {
      "type": "object",
      "required": ["tool", "version"],
      "properties": {
        "tool": {"const": "phentrieve"},
        "version": {"type": "string"}
      }
    },
    "phenopacket_id": {"type": "string"},
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["annotation_id", "phenotypic_feature_index", "hpo_id", "label", "assertion", "spans", "provenance"],
        "properties": {
          "annotation_id": {"type": "string"},
          "phenotypic_feature_index": {"type": "integer", "minimum": 0},
          "hpo_id": {"type": "string"},
          "label": {"type": "string"},
          "assertion": {"enum": ["affirmed", "negated", "uncertain", "family_history", "unknown"]},
          "certainty": {"type": "string"},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
          "evidence_text": {"type": "string"},
          "spans": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["start_char", "end_char"],
              "properties": {
                "start_char": {"type": "integer", "minimum": 0},
                "end_char": {"type": "integer", "minimum": 0},
                "text": {"type": "string"}
              }
            }
          },
          "chunk_refs": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0}
          },
          "provenance": {"type": "object"}
        }
      }
    }
  }
}
```

```python
# phentrieve/phenopackets/sidecar.py
import json
from pathlib import Path
from typing import Any

import jsonschema

from phentrieve.phenopackets.export_models import NormalizedPhenotypeExportRecord


def load_annotation_sidecar_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).with_suffix("").parent
        / "schemas"
        / "phenotype_annotation_bundle_v1.schema.json"
    )
    return json.loads(schema_path.read_text())


def build_annotation_sidecar(
    *,
    phenopacket_id: str,
    records: list[NormalizedPhenotypeExportRecord],
    generated_by_version: str,
) -> dict[str, Any]:
    annotations = []
    for feature_index, record in enumerate(records):
        annotations.append(
            {
                "annotation_id": f"ann-{feature_index + 1:04d}",
                "phenotypic_feature_index": feature_index,
                "hpo_id": record.hpo_id,
                "label": record.label,
                "assertion": record.assertion,
                "confidence": record.confidence,
                "evidence_text": record.evidence_text,
                "spans": [
                    {
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "text": span.text,
                    }
                    for span in record.spans
                ],
                "chunk_refs": record.chunk_refs,
                "provenance": {
                    "source_mode": record.source_mode,
                    "match_method": record.match_method,
                },
            }
        )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "phenotype_annotation_bundle",
        "generated_by": {"tool": "phentrieve", "version": generated_by_version},
        "phenopacket_id": phenopacket_id,
        "annotations": annotations,
    }


def validate_annotation_sidecar(sidecar: dict[str, Any]) -> None:
    jsonschema.validate(instance=sidecar, schema=load_annotation_sidecar_schema())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "annotation_sidecar" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/phenopackets/sidecar.py phentrieve/phenopackets/schemas/phenotype_annotation_bundle_v1.schema.json tests/unit/phenopacket_utils/test_phenopacket_utils.py
git commit -m "feat: add phenopacket annotation sidecar schema"
```

## Task 5: Link Strict Phenopacket Export To Optional Sidecar

**Files:**
- Modify: `phentrieve/phenopackets/utils.py`
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Write the failing tests for paired export helpers**

```python
from phentrieve.phenopackets.utils import export_phenopacket_bundle


def test_export_phenopacket_bundle_returns_strict_packet_and_optional_sidecar() -> None:
    bundle = export_phenopacket_bundle(
        aggregated_results=[
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
        ],
        phentrieve_version="0.16.0",
        include_annotation_sidecar=True,
    )

    assert "phenopacket_json" in bundle
    assert "annotation_sidecar" in bundle
    assert bundle["annotation_sidecar"]["annotations"][0]["phenotypic_feature_index"] == 0


def test_export_phenopacket_bundle_keeps_default_single_artifact_behavior() -> None:
    bundle = export_phenopacket_bundle(
        aggregated_results=[
            {"id": "HP:0001250", "name": "Seizure", "confidence": 0.9, "rank": 1}
        ],
        include_annotation_sidecar=False,
    )

    assert "phenopacket_json" in bundle
    assert bundle["annotation_sidecar"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "export_phenopacket_bundle" -v
```

Expected:

```text
ImportError: cannot import name 'export_phenopacket_bundle'
```

- [ ] **Step 3: Add paired export orchestration without changing the default formatter API**

```python
# phentrieve/phenopackets/utils.py
from phentrieve.phenopackets.sidecar import build_annotation_sidecar


def export_phenopacket_bundle(
    *,
    aggregated_results: list[dict[str, Any]] | None = None,
    chunk_results: list[dict[str, Any]] | None = None,
    phentrieve_version: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    hpo_version: str | None = None,
    input_text: str | None = None,
    include_annotation_sidecar: bool = False,
) -> dict[str, Any]:
    normalized_records = _normalize_export_inputs(
        aggregated_results=aggregated_results,
        chunk_results=chunk_results,
    )
    phenopacket_json = _format_from_normalized_records(
        normalized_records,
        phentrieve_version=phentrieve_version,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        hpo_version=hpo_version,
        input_text=input_text,
    )
    sidecar = None
    if include_annotation_sidecar:
        packet_dict = json.loads(phenopacket_json)
        sidecar = build_annotation_sidecar(
            phenopacket_id=packet_dict["id"],
            records=normalized_records,
            generated_by_version=phentrieve_version or "unknown",
        )
    return {
        "phenopacket_json": phenopacket_json,
        "annotation_sidecar": sidecar,
    }
```

Keep `format_as_phenopacket_v2()` as the stable default wrapper returning only the
Phenopacket JSON string.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -k "export_phenopacket_bundle" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/phenopackets/utils.py tests/unit/phenopacket_utils/test_phenopacket_utils.py
git commit -m "feat: add paired phenopacket sidecar export"
```

## Task 6: Add CLI Support For Optional Sidecar Export

**Files:**
- Modify: `phentrieve/cli/text_commands.py`
- Test: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_text_process_phenopacket_output_can_request_sidecar(mocker) -> None:
    export_bundle = mocker.patch(
        "phentrieve.phenopackets.utils.export_phenopacket_bundle",
        return_value={
            "phenopacket_json": "{\"id\": \"packet-1\"}",
            "annotation_sidecar": {"schema_version": "1.0.0", "annotations": []},
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "clinical note",
            "--output-format",
            "phenopacket_v2_json",
            "--phenopacket-sidecar",
        ],
    )

    assert result.exit_code == 0
    export_bundle.assert_called_once()
    assert "packet-1" in result.output


def test_text_process_phenopacket_output_defaults_to_no_sidecar(mocker) -> None:
    export_bundle = mocker.patch(
        "phentrieve.phenopackets.utils.export_phenopacket_bundle",
        return_value={"phenopacket_json": "{\"id\": \"packet-1\"}", "annotation_sidecar": None},
    )

    result = runner.invoke(
        app,
        ["text", "process", "clinical note", "--output-format", "phenopacket_v2_json"],
    )

    assert result.exit_code == 0
    assert export_bundle.call_args.kwargs["include_annotation_sidecar"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest -n 0 tests/unit/cli/test_text_commands.py -k "phenopacket_sidecar" -v
```

Expected:

```text
FAIL because the flag does not exist and the export helper is not used
```

- [ ] **Step 3: Add minimal CLI plumbing**

```python
# phentrieve/cli/text_commands.py
phenopacket_sidecar: Annotated[
    bool,
    typer.Option(
        "--phenopacket-sidecar/--no-phenopacket-sidecar",
        help="Emit an additional linked annotation sidecar when using phenopacket output.",
    ),
] = False,
```

```python
if output_format == "phenopacket_v2_json":
    from phentrieve.phenopackets.utils import export_phenopacket_bundle

    bundle = export_phenopacket_bundle(
        chunk_results=chunk_level_results,
        phentrieve_version=__version__,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        input_text=input_text,
        include_annotation_sidecar=phenopacket_sidecar,
    )
    typer.echo(bundle["phenopacket_json"])
    if phenopacket_sidecar and bundle["annotation_sidecar"] is not None:
        typer.echo(json.dumps(bundle["annotation_sidecar"], indent=2), err=True)
    return
```

Use stderr for the optional second artifact in the minimal implementation if the
current CLI command remains stdout-oriented. If the existing command already
writes files, adapt the assertions accordingly.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/cli/test_text_commands.py -k "phenopacket_sidecar" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
git commit -m "feat: add optional phenopacket sidecar cli export"
```

## Task 7: Run Focused Validation And Full Verification

**Files:**
- Modify: none
- Test: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`
- Test: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Run focused phenopacket tests**

Run:

```bash
uv run pytest -n 0 tests/unit/phenopacket_utils/test_phenopacket_utils.py -v
uv run pytest -n 0 tests/unit/cli/test_text_commands.py -k "phenopacket" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 2: Run repo verification commands**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected:

```text
All checks pass
```

- [ ] **Step 3: Commit any final test or formatting adjustments**

```bash
git add phentrieve/phenopackets/utils.py phentrieve/phenopackets/export_models.py phentrieve/phenopackets/sidecar.py phentrieve/phenopackets/schemas/phenotype_annotation_bundle_v1.schema.json phentrieve/cli/text_commands.py tests/unit/phenopacket_utils/test_phenopacket_utils.py tests/unit/cli/test_text_commands.py
git commit -m "test: finalize phenopacket sidecar export verification"
```
