# Ontology-Aware Benchmark Command Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in ontology-aware soft and partial metrics to the benchmark commands without changing strict exact benchmark behavior.

**Architecture:** Implement deterministic ontology pair credit in a focused evaluation module, then build document-level one-to-one matching on top of it. Corpus aggregation remains in `phentrieve/evaluation/`, while `phentrieve/benchmark/extraction_*` and `phentrieve/benchmark/llm_*` only expose CLI/config options and persist the resulting metric blocks.

**Tech Stack:** Python 3.10, dataclasses, Typer, NetworkX, pytest, existing HPO SQLite graph data and `CorpusExtractionMetrics`.

---

## Command Contract

V1 is opt-in:

```bash
phentrieve benchmark extraction run tests/data/en/phenobert \
  --dataset GeneReviews \
  --ontology-aware-metrics

phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-model gemini-3.1-flash-lite-preview \
  --llm-provider gemini \
  --ontology-aware-metrics
```

Default behavior remains strict-only. The new option adds metric output only; it
must not affect extraction, retrieval, prompts, LLM calls, or strict metrics.

Advanced options:

```text
--ontology-semantic-floor FLOAT
--ontology-similarity-formula [hybrid|simple_resnik_like]
```

Default values:

```text
ontology_aware_metrics = False
ontology_semantic_floor = 0.30
ontology_similarity_formula = "hybrid"
```

## File Structure

Create:

- `phentrieve/evaluation/ontology_credit.py`
  - HPO relationship classification and pair-credit calculation.
- `phentrieve/evaluation/ontology_matching.py`
  - Document-level soft one-to-one matching and partial best-match diagnostics.
- `tests/unit/evaluation/test_ontology_credit.py`
  - Unit coverage for pair-credit rules.
- `tests/unit/evaluation/test_ontology_matching.py`
  - Unit coverage for one-to-one matching and assertion handling.

Modify:

- `phentrieve/evaluation/extraction_metrics.py`
  - Corpus-level ontology metric aggregation and serialization dataclasses.
- `phentrieve/benchmark/extraction_benchmark.py`
  - Extraction config fields, metrics calculation, JSON persistence.
- `phentrieve/benchmark/extraction_cli.py`
  - Typer options and result display.
- `phentrieve/benchmark/extraction_reporter.py`
  - Report ontology metrics when present.
- `phentrieve/benchmark/llm_benchmark.py`
  - `run_llm_benchmark()` option, ontology metric calculation, payload output.
- `phentrieve/benchmark/llm_cli.py`
  - Typer options, checkpoint compatibility, artifact metrics output.
- `tests/unit/test_extraction_metrics.py`
  - Corpus aggregation tests.
- `tests/unit/test_llm_benchmark.py`
  - LLM payload/artifact tests.
- `tests/unit/cli/test_benchmark_commands.py`
  - CLI option pass-through tests.
- `tests/integration/test_benchmark_workflow.py`
  - Lightweight integration coverage for saved extraction output.

## Task 1: Pair-Credit Primitives

**Files:**

- Create: `phentrieve/evaluation/ontology_credit.py`
- Create: `tests/unit/evaluation/test_ontology_credit.py`

- [ ] **Step 1: Write failing pair-credit tests**

Add tests that monkeypatch graph loading and semantic similarity so the behavior
is deterministic:

```python
from phentrieve.evaluation.ontology_credit import (
    MatchKind,
    OntologyCreditConfig,
    calculate_pair_credit,
)


def test_exact_pair_receives_full_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: ({"HP:1": {"HP:1"}}, {"HP:1": 1}),
    )

    credit = calculate_pair_credit("HP:1", "HP:1")

    assert credit.credit == 1.0
    assert credit.match_kind == MatchKind.EXACT


def test_direct_descendant_receives_high_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:child": {"HP:root", "HP:parent", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:child": 2},
        ),
    )

    credit = calculate_pair_credit("HP:child", "HP:parent")

    assert credit.credit == 0.95
    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.distance == 1


def test_direct_ancestor_receives_lower_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:child": {"HP:root", "HP:parent", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:child": 2},
        ),
    )

    credit = calculate_pair_credit("HP:parent", "HP:child")

    assert credit.credit == 0.85
    assert credit.match_kind == MatchKind.ANCESTOR
    assert credit.distance == 1


def test_assertion_independent_sibling_credit_uses_minimum(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:a": {"HP:root", "HP:parent", "HP:a"},
                "HP:b": {"HP:root", "HP:parent", "HP:b"},
                "HP:parent": {"HP:root", "HP:parent"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:a": 2, "HP:b": 2},
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.2,
    )

    credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.credit == 0.65
    assert credit.match_kind == MatchKind.SIBLING


def test_unrelated_below_floor_receives_zero(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:a": {"HP:root", "HP:a"},
                "HP:b": {"HP:other", "HP:b"},
            },
            {"HP:root": 0, "HP:other": 0, "HP:a": 1, "HP:b": 1},
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.29,
    )

    credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.credit == 0.0
    assert credit.match_kind == MatchKind.UNRELATED


def test_real_hpo_intellectual_disability_child_credit():
    credit = calculate_pair_credit(
        "HP:0001256",
        "HP:0001249",
        OntologyCreditConfig(semantic_floor=0.30),
    )

    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.credit >= 0.90
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
uv run pytest tests/unit/evaluation/test_ontology_credit.py -q
```

Expected: import failure for `phentrieve.evaluation.ontology_credit`.

- [ ] **Step 3: Implement pair-credit primitives**

Create `phentrieve/evaluation/ontology_credit.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)


class MatchKind(str, Enum):
    EXACT = "exact"
    DESCENDANT = "descendant"
    ANCESTOR = "ancestor"
    SIBLING = "sibling"
    COUSIN = "cousin"
    SEMANTIC = "semantic"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class OntologyCreditConfig:
    semantic_floor: float = 0.30
    descendant_base: float = 0.95
    descendant_step_penalty: float = 0.05
    descendant_min: float = 0.75
    ancestor_base: float = 0.85
    ancestor_step_penalty: float = 0.08
    ancestor_min: float = 0.50
    sibling_min: float = 0.65
    cousin_min: float = 0.45
    similarity_formula: SimilarityFormula = SimilarityFormula.HYBRID


@dataclass(frozen=True)
class PairCredit:
    predicted_id: str
    gold_id: str
    credit: float
    match_kind: MatchKind
    semantic_similarity: float
    distance: int | None


def calculate_pair_credit(
    predicted_id: str,
    gold_id: str,
    config: OntologyCreditConfig | None = None,
) -> PairCredit:
    config = config or OntologyCreditConfig()
    ancestors, depths = load_hpo_graph_data()
    if predicted_id == gold_id:
        return PairCredit(predicted_id, gold_id, 1.0, MatchKind.EXACT, 1.0, 0)

    distance = _ontology_distance(predicted_id, gold_id, ancestors, depths)
    if _is_descendant(predicted_id, gold_id, ancestors):
        credit = max(
            config.descendant_min,
            config.descendant_base
            - config.descendant_step_penalty * ((distance or 1) - 1),
        )
        return PairCredit(
            predicted_id, gold_id, credit, MatchKind.DESCENDANT, 0.0, distance
        )
    if _is_descendant(gold_id, predicted_id, ancestors):
        credit = max(
            config.ancestor_min,
            config.ancestor_base
            - config.ancestor_step_penalty * ((distance or 1) - 1),
        )
        return PairCredit(
            predicted_id, gold_id, credit, MatchKind.ANCESTOR, 0.0, distance
        )

    similarity = _safe_similarity(predicted_id, gold_id, config)
    relationship = _sibling_or_cousin(predicted_id, gold_id, ancestors, depths)
    if relationship == MatchKind.SIBLING:
        return PairCredit(
            predicted_id,
            gold_id,
            max(similarity, config.sibling_min),
            MatchKind.SIBLING,
            similarity,
            distance,
        )
    if relationship == MatchKind.COUSIN:
        return PairCredit(
            predicted_id,
            gold_id,
            max(similarity, config.cousin_min),
            MatchKind.COUSIN,
            similarity,
            distance,
        )
    if similarity >= config.semantic_floor:
        return PairCredit(
            predicted_id, gold_id, similarity, MatchKind.SEMANTIC, similarity, distance
        )
    return PairCredit(predicted_id, gold_id, 0.0, MatchKind.UNRELATED, similarity, distance)
```

Also implement private helpers:

- `_is_descendant(child, parent, ancestors)`;
- `_ontology_distance(term_a, term_b, ancestors, depths)`;
- `_parents_of(term_id, ancestors, depths)`;
- `_sibling_or_cousin(term_a, term_b, ancestors, depths)`;
- `_safe_similarity(predicted_id, gold_id, config)`.

Use ancestor sets and depths from `load_hpo_graph_data()`; do not introduce a new
ontology provider in V1.

- [ ] **Step 4: Run pair-credit tests**

Run:

```bash
uv run pytest tests/unit/evaluation/test_ontology_credit.py -q
```

Expected: all tests pass.

## Task 2: Document-Level Ontology Matching

**Files:**

- Create: `phentrieve/evaluation/ontology_matching.py`
- Create: `tests/unit/evaluation/test_ontology_matching.py`

- [ ] **Step 1: Write failing matching tests**

Add tests for exact, partial, assertion mismatch, and one-to-one behavior:

```python
from phentrieve.evaluation.extraction_metrics import ExtractionResult
from phentrieve.evaluation.ontology_matching import (
    calculate_document_ontology_metrics,
)


def test_exact_document_match_scores_one(monkeypatch):
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:1", "PRESENT")],
        gold=[("HP:1", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_precision == 1.0
    assert metrics.soft_recall == 1.0
    assert metrics.soft_f1 == 1.0
    assert metrics.soft_tp == 1.0


def test_partial_child_match_scores_fraction(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        lambda pred, gold, config=None: _fake_credit(pred, gold, 0.95, "descendant"),
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:child", "PRESENT")],
        gold=[("HP:parent", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.95
    assert metrics.soft_fp == 0.05
    assert metrics.soft_fn == 0.05
    assert metrics.soft_f1 == 0.95


def test_assertion_mismatch_gets_no_credit(monkeypatch):
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:1", "ABSENT")],
        gold=[("HP:1", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.0
    assert metrics.soft_precision == 0.0
    assert metrics.soft_recall == 0.0


def test_one_prediction_cannot_satisfy_two_gold_terms(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_matching.calculate_pair_credit",
        lambda pred, gold, config=None: _fake_credit(pred, gold, 0.9, "semantic"),
    )
    result = ExtractionResult(
        doc_id="doc",
        predicted=[("HP:p", "PRESENT")],
        gold=[("HP:g1", "PRESENT"), ("HP:g2", "PRESENT")],
    )

    metrics = calculate_document_ontology_metrics(result)

    assert metrics.soft_tp == 0.9
    assert metrics.soft_recall == 0.45
    assert metrics.partial_recall == 0.9
```

Define `_fake_credit()` in the test module using `PairCredit` and `MatchKind`.

- [ ] **Step 2: Run matching tests and confirm they fail**

Run:

```bash
uv run pytest tests/unit/evaluation/test_ontology_matching.py -q
```

Expected: import failure for `phentrieve.evaluation.ontology_matching`.

- [ ] **Step 3: Implement document matching**

Create `phentrieve/evaluation/ontology_matching.py` with dataclasses:

```python
@dataclass(frozen=True)
class MatchedPair:
    predicted: tuple[str, str]
    gold: tuple[str, str]
    credit: PairCredit


@dataclass(frozen=True)
class DocumentOntologyMetrics:
    doc_id: str
    prediction_count: int
    gold_count: int
    strict_tp: int
    soft_tp: float
    soft_fp: float
    soft_fn: float
    soft_precision: float
    soft_recall: float
    soft_f1: float
    partial_precision: float
    partial_recall: float
    partial_f1: float
    matches: list[MatchedPair]
    unmatched_predictions: list[tuple[str, str]]
    unmatched_gold: list[tuple[str, str]]
```

Implement:

```python
def calculate_document_ontology_metrics(
    result: ExtractionResult,
    config: OntologyCreditConfig | None = None,
) -> DocumentOntologyMetrics:
    ...
```

Use exact-locking first, then `networkx.algorithms.matching.max_weight_matching`
over remaining same-assertion candidates. Exclude edges with `credit <= 0`.

- [ ] **Step 4: Run matching tests**

Run:

```bash
uv run pytest tests/unit/evaluation/test_ontology_matching.py -q
```

Expected: all tests pass.

## Task 3: Corpus-Level Ontology Metrics

**Files:**

- Modify: `phentrieve/evaluation/extraction_metrics.py`
- Test: `tests/unit/test_extraction_metrics.py`

- [ ] **Step 1: Add failing corpus aggregation tests**

Add tests that monkeypatch document ontology metrics:

```python
def test_calculate_ontology_aware_metrics_aggregates_micro(monkeypatch):
    evaluator = CorpusExtractionMetrics()
    results = [
        ExtractionResult("doc1", [("HP:1", "PRESENT")], [("HP:1", "PRESENT")]),
        ExtractionResult("doc2", [("HP:2", "PRESENT")], [("HP:3", "PRESENT")]),
    ]

    metrics = evaluator.calculate_ontology_aware_metrics(results)

    assert metrics.strict.micro["f1"] == evaluator.calculate_all_metrics(results).micro["f1"]
    assert "f1" in metrics.soft.micro
    assert "f1" in metrics.partial.micro
```

Also assert that an empty result list returns zero metrics and an empty match
breakdown.

- [ ] **Step 2: Run focused extraction metric tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/test_extraction_metrics.py -q
```

Expected: `CorpusExtractionMetrics` has no `calculate_ontology_aware_metrics`.

- [ ] **Step 3: Add corpus dataclasses and aggregation method**

Add dataclasses to `extraction_metrics.py`:

```python
@dataclass
class OntologyMetricBlock:
    micro: dict[str, float]
    macro: dict[str, float]
    weighted: dict[str, float]


@dataclass
class OntologyAwareCorpusMetrics:
    strict: OntologyMetricBlock
    soft: OntologyMetricBlock
    partial: OntologyMetricBlock
    match_breakdown: dict[str, dict[str, float]]
    document_metrics: list[Any]
```

Add:

```python
def calculate_ontology_aware_metrics(
    self,
    results: list[ExtractionResult],
    config: OntologyCreditConfig | None = None,
) -> OntologyAwareCorpusMetrics:
    ...
```

Micro soft metrics aggregate `soft_tp`, `soft_fp`, and `soft_fn` across docs
before calculating PR/F1. Macro averages per-document values. Weighted averages
per-document values by `gold_count` with minimum weight 1.

- [ ] **Step 4: Run corpus tests**

Run:

```bash
uv run pytest tests/unit/test_extraction_metrics.py -q
```

Expected: all extraction metric tests pass.

## Task 4: Extraction Benchmark Command Option

**Files:**

- Modify: `phentrieve/benchmark/extraction_cli.py`
- Modify: `phentrieve/benchmark/extraction_benchmark.py`
- Modify: `phentrieve/benchmark/extraction_reporter.py`
- Test: `tests/integration/test_benchmark_workflow.py`

- [ ] **Step 1: Add failing extraction output test**

Extend `test_extraction_benchmark_uses_effective_config_overrides()` or add a
neighbor test that runs with:

```python
config=ExtractionConfig(
    model_name="sentence-transformers/LaBSE",
    dataset="all",
    averaging="micro",
    bootstrap_ci=False,
    ontology_aware_metrics=True,
)
```

After `benchmark.run_benchmark(...)`, assert:

```python
saved_payload = json.loads(
    (output_dir / "extraction_results.json").read_text(encoding="utf-8")
)
summary = json.loads(
    (output_dir / "extraction_summary.json").read_text(encoding="utf-8")
)

assert saved_payload["metadata"]["config"]["ontology_aware_metrics"] is True
assert "ontology_metrics" in saved_payload["corpus_metrics"]
assert "soft_micro_f1" in summary
assert "partial_micro_f1" in summary
```

- [ ] **Step 2: Run focused integration test and confirm failure**

Run:

```bash
uv run pytest tests/integration/test_benchmark_workflow.py::test_extraction_benchmark_saves_ontology_metrics_when_enabled -q
```

Expected: `ExtractionConfig` has no `ontology_aware_metrics`.

- [ ] **Step 3: Add extraction config fields**

Modify `ExtractionConfig`:

```python
ontology_aware_metrics: bool = False
ontology_semantic_floor: float = 0.30
ontology_similarity_formula: str = "hybrid"
```

- [ ] **Step 4: Wire metrics into extraction benchmark saving**

In `ExtractionBenchmark.run_benchmark()`:

```python
ontology_metrics = None
if config.ontology_aware_metrics:
    ontology_config = _build_ontology_credit_config(config)
    ontology_metrics = evaluator.calculate_ontology_aware_metrics(
        results,
        config=ontology_config,
    )
```

Pass `ontology_metrics` into `_save_results()`.

In `_save_results()`, add:

- config fields in `metadata.config`;
- nested `corpus_metrics.ontology_metrics`;
- flat `soft_micro_*` and `partial_micro_*` fields in `extraction_summary.json`.

- [ ] **Step 5: Add Typer options**

In `extraction_cli.run()` add:

```python
ontology_aware_metrics: bool = typer.Option(
    False,
    "--ontology-aware-metrics/--no-ontology-aware-metrics",
    help="Calculate ontology-aware soft and partial benchmark metrics.",
),
ontology_semantic_floor: float = typer.Option(
    0.30,
    "--ontology-semantic-floor",
    help="Minimum fallback semantic similarity for ontology-aware credit.",
),
ontology_similarity_formula: str = typer.Option(
    "hybrid",
    "--ontology-similarity-formula",
    help="Ontology fallback similarity formula: hybrid or simple_resnik_like.",
),
```

Pass them into `ExtractionConfig`.

- [ ] **Step 6: Display ontology result rows when enabled**

Update `_display_results()` so if the metrics object has ontology metrics, it
prints a second table titled `Ontology-Aware Benchmark Results` with strict,
soft, and partial micro F1.

- [ ] **Step 7: Run extraction command tests**

Run:

```bash
uv run pytest tests/integration/test_benchmark_workflow.py::test_extraction_benchmark_saves_ontology_metrics_when_enabled -q
```

Expected: pass.

## Task 5: LLM Benchmark Command Option

**Files:**

- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/test_llm_benchmark.py`
- Test: `tests/unit/cli/test_benchmark_commands.py`

- [ ] **Step 1: Add failing CLI pass-through test**

In `tests/unit/cli/test_benchmark_commands.py`, add:

```python
def test_benchmark_llm_command_passes_ontology_metric_options(tmp_path, monkeypatch):
    runner = CliRunner()
    test_file = tmp_path / "cases.json"
    test_file.write_text("[]", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_llm_benchmark_cli(**kwargs):
        captured.update(kwargs)
        return {
            "cases": 0,
            "llm_model": kwargs["llm_model"],
            "llm_mode": kwargs["llm_mode"],
            "dataset": kwargs["dataset"],
            "output_path": str(tmp_path / "result.json"),
        }

    monkeypatch.setattr(
        "phentrieve.benchmark.llm_cli.run_llm_benchmark_cli",
        fake_run_llm_benchmark_cli,
    )

    result = runner.invoke(
        cli_app,
        [
            "benchmark",
            "llm",
            "--test-file",
            str(test_file),
            "--llm-model",
            "gemini-3.1-flash-lite-preview",
            "--ontology-aware-metrics",
            "--ontology-semantic-floor",
            "0.25",
            "--ontology-similarity-formula",
            "simple_resnik_like",
        ],
    )

    assert result.exit_code == 0
    assert captured["ontology_aware_metrics"] is True
    assert captured["ontology_semantic_floor"] == 0.25
    assert captured["ontology_similarity_formula"] == "simple_resnik_like"
```

- [ ] **Step 2: Add failing payload/artifact test**

In `tests/unit/test_llm_benchmark.py`, add a unit test that monkeypatches
`llm_benchmark.run_llm_benchmark()` to return metrics with `ontology_metrics`,
then asserts `_write_benchmark_artifacts()` persists the nested block in
`metrics/benchmark_two_phase.json`.

Use this assertion:

```python
metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
assert "ontology_metrics" in metrics_payload["assertion_aware_metrics"]
```

- [ ] **Step 3: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/cli/test_benchmark_commands.py::test_benchmark_llm_command_passes_ontology_metric_options -q
uv run pytest tests/unit/test_llm_benchmark.py::test_run_llm_benchmark_cli_writes_ontology_metrics_artifact -q
```

Expected: option and output assertions fail.

- [ ] **Step 4: Add LLM CLI option plumbing**

Modify `benchmark_llm()` in `phentrieve/benchmark/llm_cli.py` with the same
Typer options as extraction. Add parameters to `run_llm_benchmark_cli()` and pass
them to `llm_benchmark.run_llm_benchmark()`.

Include these values in checkpoint compatibility state:

```python
"ontology_aware_metrics": ontology_aware_metrics,
"ontology_semantic_floor": ontology_semantic_floor,
"ontology_similarity_formula": ontology_similarity_formula,
```

- [ ] **Step 5: Add LLM benchmark metric calculation**

Modify `run_llm_benchmark()` in `phentrieve/benchmark/llm_benchmark.py`:

```python
assertion_metrics = evaluator.calculate_all_metrics(assertion_results)
id_only_metrics = evaluator.calculate_all_metrics(id_only_results)

if ontology_aware_metrics:
    ontology_config = build_ontology_credit_config(
        semantic_floor=ontology_semantic_floor,
        similarity_formula=ontology_similarity_formula,
    )
    assertion_ontology_metrics = evaluator.calculate_ontology_aware_metrics(
        assertion_results,
        config=ontology_config,
    )
    id_only_ontology_metrics = evaluator.calculate_ontology_aware_metrics(
        id_only_results,
        config=ontology_config,
    )
```

Serialize nested ontology metrics inside each metric namespace:

```python
metrics={
    "assertion_aware": {
        **_serialize_corpus_metrics(assertion_metrics),
        "ontology_metrics": _serialize_ontology_metrics(assertion_ontology_metrics),
    },
    "id_only": {
        **_serialize_corpus_metrics(id_only_metrics),
        "ontology_metrics": _serialize_ontology_metrics(id_only_ontology_metrics),
    },
}
```

Only include `ontology_metrics` when enabled.

- [ ] **Step 6: Update artifact metrics JSON writer**

`_write_benchmark_artifacts()` currently writes each metric namespace as a whole
dict, so the nested ontology block should persist automatically once the metric
payload includes it. Add the test anyway to prevent future regressions.

- [ ] **Step 7: Run LLM tests**

Run:

```bash
uv run pytest tests/unit/cli/test_benchmark_commands.py::test_benchmark_llm_command_passes_ontology_metric_options -q
uv run pytest tests/unit/test_llm_benchmark.py::test_run_llm_benchmark_cli_writes_ontology_metrics_artifact -q
```

Expected: pass.

## Task 6: Serialization Helpers And Report Shape

**Files:**

- Modify: `phentrieve/evaluation/extraction_metrics.py`
- Modify: `phentrieve/benchmark/extraction_benchmark.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`

- [ ] **Step 1: Add serializer tests**

Assert serialized ontology metrics contain only JSON primitives:

```python
json.dumps(_serialize_ontology_metrics(metrics))
```

Also assert match breakdown values use:

```json
{"count": 1, "credit": 0.95}
```

- [ ] **Step 2: Implement serializer helper**

Add one helper and reuse it from both benchmark paths:

```python
def serialize_ontology_metrics(metrics: OntologyAwareCorpusMetrics) -> dict[str, Any]:
    return {
        "strict": _block(metrics.strict),
        "soft": _block(metrics.soft),
        "partial": _block(metrics.partial),
        "match_breakdown": metrics.match_breakdown,
    }
```

Place it in `phentrieve/evaluation/extraction_metrics.py` unless importing it
there creates a cycle. If a cycle appears, put it in
`phentrieve/evaluation/ontology_matching.py`.

- [ ] **Step 3: Add optional detailed pair output**

For V1, include detailed matched pairs in `extraction_detailed_analysis.json`
only when both `detailed_output` and `ontology_aware_metrics` are true. Do not
inflate normal summary files with all pair records.

## Task 7: End-To-End Verification

**Files:**

- No new production files.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
uv run pytest \
  tests/unit/evaluation/test_ontology_credit.py \
  tests/unit/evaluation/test_ontology_matching.py \
  tests/unit/test_extraction_metrics.py \
  tests/unit/test_llm_benchmark.py \
  tests/unit/cli/test_benchmark_commands.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run integration benchmark workflow test**

Run:

```bash
uv run pytest tests/integration/test_benchmark_workflow.py::test_extraction_benchmark_saves_ontology_metrics_when_enabled -q
```

Expected: pass.

- [ ] **Step 3: Run command help smoke checks**

Run:

```bash
uv run phentrieve benchmark extraction run --help | rg "ontology-aware-metrics"
uv run phentrieve benchmark llm --help | rg "ontology-aware-metrics"
```

Expected: both commands show the option.

- [ ] **Step 4: Run required repo checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all pass.

## Self-Review Checklist

- [ ] The option is available on both benchmark command paths.
- [ ] Default behavior remains strict-only.
- [ ] Strict metrics are byte-for-byte unchanged when the option is disabled.
- [ ] LLM checkpoints include ontology option fields so resumes cannot mix
      incompatible metric settings.
- [ ] Ontology metrics are computed from already-produced predictions and gold
      terms only.
- [ ] Detailed matched pairs are auditable when detailed output is requested.
- [ ] No new dependency is introduced.
