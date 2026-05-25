# Full-Text Multi-Vector Parity Implementation Plan

> Status: Completed and filed on 2026-05-25 based on git-history audit. The
> implementation landed on `main` through the 2026-05-24 commit sequence for
> conversion similarities, shared chunk candidate routing, batched multi-vector
> retrieval, full-text orchestration routing, and adaptive child routing.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standard full-text HPO extraction and adaptive child retrieval use HPO-level multi-vector aggregation whenever the connected retriever is backed by a multi-vector index.

**Architecture:** Add a Chroma-style batched multi-vector retrieval method to `DenseRetriever`, extend the existing multi-vector conversion utility to carry similarities, and centralize per-chunk retrieval selection in `phentrieve/retrieval/utils.py`. Keep `precomputed_query_results` as the authoritative orchestrator bypass, and route both initial full-text retrieval and adaptive child retrieval through the shared helper.

**Tech Stack:** Python 3.11 typing, ChromaDB-style result dictionaries, existing `DenseRetriever`, existing `aggregate_multi_vector_results`, pytest, Ruff, mypy.

---

## Sources Read

- Spec: `.planning/specs/2026-05-24-full-text-multi-vector-parity-design.md`
- Context analysis: `.planning/analysis/2026-05-23-phentrieve-rag-prompting-literature-report.md`
- Implementation files:
  - `phentrieve/retrieval/dense_retriever.py`
  - `phentrieve/retrieval/utils.py`
  - `phentrieve/text_processing/hpo_extraction_orchestrator.py`
  - `phentrieve/retrieval/adaptive_rechunker.py`
  - `phentrieve/text_processing/full_text_service.py`
- Existing tests:
  - `tests/unit/retrieval/test_dense_retriever_real.py`
  - `tests/unit/retrieval/test_retrieval_utils.py`
  - `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`
  - `tests/unit/text_processing/test_orchestrate_with_precomputed.py`
  - `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`

## File Structure

- Modify: `phentrieve/retrieval/utils.py`
  - Extend `convert_multi_vector_to_chromadb_format(...)` with `include_similarities: bool = False`.
  - Add `query_chunk_candidates(...)` as the single non-precomputed retrieval selector for text chunks.
- Modify: `phentrieve/retrieval/dense_retriever.py`
  - Import `convert_multi_vector_to_chromadb_format`.
  - Add `DenseRetriever.query_batch_multi_vector(...)` after `query_multi_vector(...)`.
- Modify: `phentrieve/text_processing/hpo_extraction_orchestrator.py`
  - Import `query_chunk_candidates`.
  - Replace the non-precomputed `retriever.query_batch(...)` call with `query_chunk_candidates(...)`.
  - Leave `precomputed_query_results` length validation and bypass behavior in place.
- Modify: `phentrieve/retrieval/adaptive_rechunker.py`
  - Import `query_chunk_candidates`.
  - Replace child-chunk `retriever.query_batch(...)` with `query_chunk_candidates(...)`.
  - Update local comments/docstrings from "query_batch call" to retrieval call where they describe the cost invariant.
- Read only: `phentrieve/text_processing/full_text_service.py`
  - No direct code change expected. `run_standard_backend(...)` already creates the default multi-vector retriever and calls the orchestrator, so it should benefit from orchestrator routing.
- Modify: `tests/unit/retrieval/test_retrieval_utils.py`
  - Add conversion similarity coverage and shared helper routing coverage.
- Modify: `tests/unit/retrieval/test_dense_retriever_real.py`
  - Add `DenseRetriever.query_batch_multi_vector(...)` coverage.
- Modify: `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`
  - Add multi-vector and single-vector routing tests plus the duplicate component-hit regression test.
- Modify: `tests/unit/text_processing/test_orchestrate_with_precomputed.py`
  - Strengthen bypass coverage so precomputed results skip both retrieval methods.
- Modify: `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`
  - Add child retrieval routing tests for multi-vector and single-vector retrievers.

## Task 1: Preserve Conversion Compatibility And Add Similarities

**Files:**
- Modify: `phentrieve/retrieval/utils.py:1-55`
- Test: `tests/unit/retrieval/test_retrieval_utils.py:1-109`

- [ ] **Step 1: Write failing conversion tests**

Add these tests to `TestConvertMultiVectorToChromadbFormat` in `tests/unit/retrieval/test_retrieval_utils.py`:

```python
    def test_default_does_not_include_similarities(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.91,
            }
        ]

        converted = convert_multi_vector_to_chromadb_format(results)

        assert "similarities" not in converted

    def test_include_similarities_true_adds_nested_scores(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.91,
                "matched_component": "synonym",
                "matched_text": "Convulsions",
            },
            {
                "hpo_id": "HP:0001251",
                "label": "Ataxia",
                "similarity": 0.75,
            },
        ]

        converted = convert_multi_vector_to_chromadb_format(
            results,
            include_similarities=True,
        )

        assert converted["ids"] == [["HP:0001250", "HP:0001251"]]
        assert converted["similarities"] == [[0.91, 0.75]]
        assert converted["distances"] == [[pytest.approx(0.09), pytest.approx(0.25)]]
        assert converted["metadatas"][0][0]["hpo_id"] == "HP:0001250"
        assert converted["metadatas"][0][0]["matched_component"] == "synonym"
        assert converted["metadatas"][0][0]["matched_text"] == "Convulsions"

    def test_empty_result_includes_empty_similarities_when_requested(self):
        converted = convert_multi_vector_to_chromadb_format(
            [],
            include_similarities=True,
        )

        assert converted == {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "similarities": [[]],
        }
```

- [ ] **Step 2: Run conversion tests and verify failure**

Run:

```bash
uv run pytest tests/unit/retrieval/test_retrieval_utils.py::TestConvertMultiVectorToChromadbFormat -n 0 -v
```

Expected: failure with `TypeError: convert_multi_vector_to_chromadb_format() got an unexpected keyword argument 'include_similarities'`.

- [ ] **Step 3: Implement minimal conversion support**

Change `convert_multi_vector_to_chromadb_format(...)` in `phentrieve/retrieval/utils.py` to this shape:

```python
def convert_multi_vector_to_chromadb_format(
    multi_vector_results: list[dict[str, Any]],
    include_similarities: bool = False,
) -> dict[str, Any]:
    """
    Convert multi-vector aggregated results to ChromaDB-style format.

    This allows reusing existing format_results() and output formatters.

    Args:
        multi_vector_results: List of aggregated results from query_multi_vector()
        include_similarities: Whether to include nested similarity scores

    Returns:
        Dictionary in ChromaDB query result format
    """
    if not multi_vector_results:
        empty = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        if include_similarities:
            empty["similarities"] = [[]]
        return empty

    ids = []
    metadatas = []
    documents = []
    distances = []
    similarities = []

    for result in multi_vector_results:
        hpo_id = result["hpo_id"]
        ids.append(hpo_id)
        metadata = {
            "hpo_id": hpo_id,
            "label": result.get("label", ""),
        }
        if "component_scores" in result:
            metadata["component_scores"] = result["component_scores"]
        if "matched_component" in result:
            metadata["matched_component"] = result["matched_component"]
        if "matched_text" in result:
            metadata["matched_text"] = result["matched_text"]
        metadatas.append(metadata)
        documents.append(result.get("label", ""))

        similarity = float(result.get("similarity") or 0.0)
        similarities.append(similarity)
        distances.append(1.0 - similarity)

    converted: dict[str, Any] = {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents],
        "distances": [distances],
    }
    if include_similarities:
        converted["similarities"] = [similarities]
    return converted
```

- [ ] **Step 4: Run conversion tests and verify pass**

Run:

```bash
uv run pytest tests/unit/retrieval/test_retrieval_utils.py::TestConvertMultiVectorToChromadbFormat -n 0 -v
```

Expected: all tests in `TestConvertMultiVectorToChromadbFormat` pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/retrieval/utils.py tests/unit/retrieval/test_retrieval_utils.py
git commit -m "feat: include similarities in multi-vector conversion"
```

## Task 2: Add Shared Chunk Candidate Retrieval Selection

**Files:**
- Modify: `phentrieve/retrieval/utils.py:1-120`
- Test: `tests/unit/retrieval/test_retrieval_utils.py:1-180`

- [ ] **Step 1: Write failing helper tests**

Update the import in `tests/unit/retrieval/test_retrieval_utils.py`:

```python
from phentrieve.retrieval.utils import (
    convert_multi_vector_to_chromadb_format,
    query_chunk_candidates,
)
```

Add this helper and test class after `TestConvertMultiVectorToChromadbFormat`:

```python
class _RecordingRetriever:
    def __init__(self, index_type: str = "single_vector", detection_error: Exception | None = None):
        self.index_type = index_type
        self.detection_error = detection_error
        self.batch_calls: list[dict[str, object]] = []
        self.batch_multi_vector_calls: list[dict[str, object]] = []

    def detect_index_type(self) -> str:
        if self.detection_error is not None:
            raise self.detection_error
        return self.index_type

    def query_batch(
        self,
        *,
        texts: list[str],
        n_results: int,
        include_similarities: bool,
    ) -> list[dict[str, object]]:
        self.batch_calls.append(
            {
                "texts": texts,
                "n_results": n_results,
                "include_similarities": include_similarities,
            }
        )
        return [{"ids": [["single"]], "metadatas": [[]], "similarities": [[]]}]

    def query_batch_multi_vector(
        self,
        *,
        texts: list[str],
        n_results: int,
    ) -> list[dict[str, object]]:
        self.batch_multi_vector_calls.append(
            {
                "texts": texts,
                "n_results": n_results,
            }
        )
        return [{"ids": [["multi"]], "metadatas": [[]], "similarities": [[]]}]


class TestQueryChunkCandidates:
    def test_uses_multi_vector_batch_when_index_type_matches(self):
        retriever = _RecordingRetriever(index_type="multi_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a", "chunk b"],
            n_results=7,
        )

        assert results[0]["ids"] == [["multi"]]
        assert retriever.batch_multi_vector_calls == [
            {"texts": ["chunk a", "chunk b"], "n_results": 7}
        ]
        assert retriever.batch_calls == []

    def test_uses_query_batch_for_single_vector_index(self):
        retriever = _RecordingRetriever(index_type="single_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=3,
        )

        assert results[0]["ids"] == [["single"]]
        assert retriever.batch_calls == [
            {
                "texts": ["chunk a"],
                "n_results": 3,
                "include_similarities": True,
            }
        ]
        assert retriever.batch_multi_vector_calls == []

    def test_detection_error_falls_back_to_query_batch(self):
        retriever = _RecordingRetriever(detection_error=RuntimeError("metadata failed"))

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=4,
        )

        assert results[0]["ids"] == [["single"]]
        assert len(retriever.batch_calls) == 1
        assert retriever.batch_multi_vector_calls == []

    def test_multi_vector_index_without_method_falls_back_to_query_batch(self):
        class NoMultiVectorMethod(_RecordingRetriever):
            query_batch_multi_vector = None

        retriever = NoMultiVectorMethod(index_type="multi_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=5,
        )

        assert results[0]["ids"] == [["single"]]
        assert len(retriever.batch_calls) == 1

    def test_dynamic_mock_attributes_do_not_force_multi_vector_route(self):
        from unittest.mock import MagicMock

        retriever = MagicMock()
        retriever.detect_index_type.return_value = MagicMock()
        retriever.query_batch.return_value = [
            {"ids": [["single"]], "metadatas": [[]], "similarities": [[]]}
        ]

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=2,
        )

        assert results[0]["ids"] == [["single"]]
        retriever.query_batch.assert_called_once_with(
            texts=["chunk a"],
            n_results=2,
            include_similarities=True,
        )
        retriever.query_batch_multi_vector.assert_not_called()
```

- [ ] **Step 2: Run helper tests and verify failure**

Run:

```bash
uv run pytest tests/unit/retrieval/test_retrieval_utils.py::TestQueryChunkCandidates -n 0 -v
```

Expected: import failure because `query_chunk_candidates` does not exist.

- [ ] **Step 3: Implement the helper**

Add logging and the helper to `phentrieve/retrieval/utils.py`:

```python
import logging
from typing import Any

logger = logging.getLogger(__name__)
```

```python
def query_chunk_candidates(
    *,
    retriever: Any,
    text_chunks: list[str],
    n_results: int,
    include_similarities: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve per-chunk HPO candidates using the retriever's index mode."""
    index_type = "single_vector"
    detect_index_type = getattr(retriever, "detect_index_type", None)
    if callable(detect_index_type):
        try:
            detected = detect_index_type()
            if detected in {"single_vector", "multi_vector"}:
                index_type = detected
            else:
                logger.debug("Unknown index type %r; using query_batch", detected)
        except Exception:
            logger.warning(
                "Could not detect retriever index type; using query_batch",
                exc_info=True,
            )

    query_batch_multi_vector = getattr(retriever, "query_batch_multi_vector", None)
    if index_type == "multi_vector" and callable(query_batch_multi_vector):
        logger.info("Batch querying %d chunks with multi-vector aggregation", len(text_chunks))
        return query_batch_multi_vector(
            texts=text_chunks,
            n_results=n_results,
        )

    if index_type == "multi_vector":
        logger.warning(
            "Retriever is connected to a multi-vector index but does not expose "
            "query_batch_multi_vector(); using query_batch"
        )

    logger.info("Batch querying %d chunks with query_batch", len(text_chunks))
    return retriever.query_batch(
        texts=text_chunks,
        n_results=n_results,
        include_similarities=include_similarities,
    )
```

- [ ] **Step 4: Run helper tests and full retrieval utility tests**

Run:

```bash
uv run pytest tests/unit/retrieval/test_retrieval_utils.py -n 0 -v
```

Expected: all tests in `tests/unit/retrieval/test_retrieval_utils.py` pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/retrieval/utils.py tests/unit/retrieval/test_retrieval_utils.py
git commit -m "feat: add chunk candidate retrieval selector"
```

## Task 3: Add DenseRetriever.query_batch_multi_vector

**Files:**
- Modify: `phentrieve/retrieval/dense_retriever.py:23-30`
- Modify: `phentrieve/retrieval/dense_retriever.py:472-542`
- Test: `tests/unit/retrieval/test_dense_retriever_real.py:1-767`

- [ ] **Step 1: Write failing dense retriever tests**

Update the import in `tests/unit/retrieval/test_dense_retriever_real.py`:

```python
from phentrieve.config import MIN_SIMILARITY_THRESHOLD, MULTI_VECTOR_RESULT_MULTIPLIER
```

Add this helper and test class after `TestDenseRetrieverQueryBatch`:

```python
def _raw_multi_vector_entry(items):
    ids = []
    documents = []
    metadatas = []
    distances = []
    similarities = []
    for item in items:
        ids.append(item["id"])
        documents.append(item.get("document", ""))
        metadatas.append(item["metadata"])
        similarity = item["similarity"]
        similarities.append(similarity)
        distances.append(1.0 - similarity)
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
        "similarities": [similarities],
    }


class TestDenseRetrieverQueryBatchMultiVector:
    def test_batch_multi_vector_empty_list_does_not_query(self):
        retriever = DenseRetriever(Mock(), Mock())
        retriever.query_batch = Mock()

        results = retriever.query_batch_multi_vector([])

        assert results == []
        retriever.query_batch.assert_not_called()

    def test_batch_multi_vector_aggregates_component_hits_per_hpo_id(self):
        retriever = DenseRetriever(Mock(), Mock(), min_similarity=0.0)
        retriever._index_type = "multi_vector"
        retriever.query_batch = Mock(
            return_value=[
                _raw_multi_vector_entry(
                    [
                        {
                            "id": "HP:0001250__label__0",
                            "document": "Seizure",
                            "metadata": {
                                "hpo_id": "HP:0001250",
                                "component": "label",
                                "label": "Seizure",
                            },
                            "similarity": 0.82,
                        },
                        {
                            "id": "HP:0001250__synonym__0",
                            "document": "Convulsions",
                            "metadata": {
                                "hpo_id": "HP:0001250",
                                "component": "synonym",
                                "label": "Seizure",
                                "synonym_text": "Convulsions",
                            },
                            "similarity": 0.94,
                        },
                        {
                            "id": "HP:0001251__label__0",
                            "document": "Ataxia",
                            "metadata": {
                                "hpo_id": "HP:0001251",
                                "component": "label",
                                "label": "Ataxia",
                            },
                            "similarity": 0.73,
                        },
                    ]
                )
            ]
        )

        results = retriever.query_batch_multi_vector(["recurrent convulsions"], n_results=2)

        assert len(results) == 1
        assert results[0]["ids"] == [["HP:0001250", "HP:0001251"]]
        assert results[0]["documents"] == [["Seizure", "Ataxia"]]
        assert results[0]["similarities"] == [[0.94, 0.73]]
        assert results[0]["distances"][0] == [pytest.approx(0.06), pytest.approx(0.27)]
        assert results[0]["metadatas"][0][0]["component_scores"] == {
            "label": 0.82,
            "synonyms": [0.94],
            "definition": None,
        }
        assert results[0]["metadatas"][0][0]["matched_component"] == "synonym"
        retriever.query_batch.assert_called_once_with(
            texts=["recurrent convulsions"],
            n_results=2 * MULTI_VECTOR_RESULT_MULTIPLIER,
            include_similarities=True,
        )

    def test_batch_multi_vector_aggregates_each_chunk_independently(self):
        retriever = DenseRetriever(Mock(), Mock(), min_similarity=0.0)
        retriever._index_type = "multi_vector"
        retriever.query_batch = Mock(
            return_value=[
                _raw_multi_vector_entry(
                    [
                        {
                            "id": "HP:0001250__label__0",
                            "document": "Seizure",
                            "metadata": {
                                "hpo_id": "HP:0001250",
                                "component": "label",
                                "label": "Seizure",
                            },
                            "similarity": 0.91,
                        }
                    ]
                ),
                _raw_multi_vector_entry(
                    [
                        {
                            "id": "HP:0001251__label__0",
                            "document": "Ataxia",
                            "metadata": {
                                "hpo_id": "HP:0001251",
                                "component": "label",
                                "label": "Ataxia",
                            },
                            "similarity": 0.88,
                        }
                    ]
                ),
            ]
        )

        results = retriever.query_batch_multi_vector(["chunk one", "chunk two"], n_results=1)

        assert [result["ids"] for result in results] == [[["HP:0001250"]], [["HP:0001251"]]]
        assert [result["similarities"] for result in results] == [[[0.91]], [[0.88]]]

    def test_batch_multi_vector_failure_returns_empty_result_per_text(self):
        retriever = DenseRetriever(Mock(), Mock(), min_similarity=0.0)
        retriever._index_type = "multi_vector"
        retriever.query_batch = Mock(side_effect=RuntimeError("query failed"))

        results = retriever.query_batch_multi_vector(["a", "b"], n_results=3)

        assert results == [
            {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "similarities": [[]],
            },
            {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "similarities": [[]],
            },
        ]

    def test_batch_multi_vector_invalid_strategy_returns_empty_results(self):
        retriever = DenseRetriever(Mock(), Mock(), min_similarity=0.0)
        retriever._index_type = "multi_vector"
        retriever.query_batch = Mock(
            return_value=[
                _raw_multi_vector_entry(
                    [
                        {
                            "id": "HP:0001250__label__0",
                            "document": "Seizure",
                            "metadata": {
                                "hpo_id": "HP:0001250",
                                "component": "label",
                                "label": "Seizure",
                            },
                            "similarity": 0.91,
                        }
                    ]
                )
            ]
        )

        results = retriever.query_batch_multi_vector(
            ["seizure"],
            n_results=1,
            aggregation_strategy="not_a_strategy",
        )

        assert results == [
            {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "similarities": [[]],
            }
        ]
```

- [ ] **Step 2: Run dense retriever tests and verify failure**

Run:

```bash
uv run pytest tests/unit/retrieval/test_dense_retriever_real.py::TestDenseRetrieverQueryBatchMultiVector -n 0 -v
```

Expected: failure with `AttributeError: 'DenseRetriever' object has no attribute 'query_batch_multi_vector'`.

- [ ] **Step 3: Implement `query_batch_multi_vector`**

Add this import in `phentrieve/retrieval/dense_retriever.py`:

```python
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format
```

Add this method to `DenseRetriever` after `query_multi_vector(...)`:

```python
    def query_batch_multi_vector(
        self,
        texts: list[str],
        n_results: int = 10,
        aggregation_strategy: str
        | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
        component_weights: dict[str, float] | None = None,
        custom_formula: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query a multi-vector index for multiple texts with per-text HPO aggregation.

        Returns one ChromaDB-style result dictionary per input text, matching
        the shape consumed by process_chunk_matches().
        """
        if not texts:
            return []

        index_type = self.detect_index_type()
        if index_type != "multi_vector":
            logging.warning(
                "query_batch_multi_vector called on %s index. Results may be incorrect.",
                _sanitize(index_type),
            )

        try:
            raw_n_results = n_results * MULTI_VECTOR_RESULT_MULTIPLIER
            raw_batch_results = self.query_batch(
                texts=texts,
                n_results=raw_n_results,
                include_similarities=True,
            )

            converted_results: list[dict[str, Any]] = []
            for raw_results in raw_batch_results:
                aggregated = aggregate_multi_vector_results(
                    results=raw_results,
                    strategy=aggregation_strategy,
                    weights=component_weights,
                    custom_formula=custom_formula,
                    min_similarity=self.min_similarity,
                )[:n_results]
                converted_results.append(
                    convert_multi_vector_to_chromadb_format(
                        aggregated,
                        include_similarities=True,
                    )
                )

            while len(converted_results) < len(texts):
                converted_results.append(
                    convert_multi_vector_to_chromadb_format(
                        [],
                        include_similarities=True,
                    )
                )
            return converted_results

        except Exception as e:
            logging.error(
                "Error in batch multi-vector query to HPO index: %s",
                _sanitize(e),
            )
            return [
                convert_multi_vector_to_chromadb_format(
                    [],
                    include_similarities=True,
                )
                for _ in texts
            ]
```

- [ ] **Step 4: Run focused dense retriever tests and existing query tests**

Run:

```bash
uv run pytest tests/unit/retrieval/test_dense_retriever_real.py::TestDenseRetrieverQueryBatchMultiVector tests/unit/retrieval/test_dense_retriever_real.py::TestDenseRetrieverQueryBatch -n 0 -v
```

Expected: all selected dense retriever tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/retrieval/dense_retriever.py tests/unit/retrieval/test_dense_retriever_real.py
git commit -m "feat: add batch multi-vector retrieval"
```

## Task 4: Route Standard Full-Text Orchestration Through The Shared Helper

**Files:**
- Modify: `phentrieve/text_processing/hpo_extraction_orchestrator.py:1-123`
- Test: `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py:1-286`
- Test: `tests/unit/text_processing/test_orchestrate_with_precomputed.py:1-94`

- [ ] **Step 1: Write failing orchestrator routing tests**

In `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`, update `_make_mock_retriever(...)` so existing characterization tests are explicitly single-vector:

```python
def _make_mock_retriever(batch_results):
    """Build a mock DenseRetriever whose query_batch returns ``batch_results``."""
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    retriever.query_batch.return_value = batch_results
    return retriever
```

Add these tests to `TestRetrieverInteraction`:

```python
    def test_multi_vector_retriever_uses_query_batch_multi_vector(self):
        retriever = MagicMock()
        retriever.detect_index_type.return_value = "multi_vector"
        retriever.query_batch.return_value = [
            _chroma_batch_entry([("HP:9999999", "Wrong raw component", 0.99)])
        ]
        retriever.query_batch_multi_vector.return_value = [
            _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)])
        ]

        result = orchestrate_hpo_extraction(
            text_chunks=["Patient had seizures."],
            retriever=retriever,
            num_results_per_chunk=3,
            chunk_retrieval_threshold=0.5,
        )

        retriever.query_batch_multi_vector.assert_called_once_with(
            texts=["Patient had seizures."],
            n_results=3,
        )
        retriever.query_batch.assert_not_called()
        assert [term["id"] for term in result.aggregated_results] == ["HP:0001250"]

    def test_single_vector_retriever_keeps_query_batch_route(self):
        retriever = _make_mock_retriever(
            [_chroma_batch_entry([("HP:0001250", "Seizure", 0.9)])]
        )

        orchestrate_hpo_extraction(
            text_chunks=["Patient had seizures."],
            retriever=retriever,
            num_results_per_chunk=4,
            chunk_retrieval_threshold=0.5,
        )

        retriever.query_batch.assert_called_once_with(
            texts=["Patient had seizures."],
            n_results=4,
            include_similarities=True,
        )
        retriever.query_batch_multi_vector.assert_not_called()

    def test_multi_vector_route_prevents_duplicate_component_matches(self):
        raw_component_entry = {
            "metadatas": [
                [
                    {
                        "hpo_id": "HP:0001250",
                        "label": "Seizure",
                        "component": "label",
                    },
                    {
                        "hpo_id": "HP:0001250",
                        "label": "Seizure",
                        "component": "synonym",
                    },
                ]
            ],
            "similarities": [[0.88, 0.86]],
            "distances": [[0.12, 0.14]],
            "documents": [["Seizure", "Convulsions"]],
            "ids": [["HP:0001250__label__0", "HP:0001250__synonym__0"]],
        }
        retriever = MagicMock()
        retriever.detect_index_type.return_value = "multi_vector"
        retriever.query_batch.return_value = [raw_component_entry]
        retriever.query_batch_multi_vector.return_value = [
            _chroma_batch_entry([("HP:0001250", "Seizure", 0.88)])
        ]

        result = orchestrate_hpo_extraction(
            text_chunks=["Patient had recurrent convulsions."],
            retriever=retriever,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.5,
        )

        assert len(result.chunk_results[0]["matches"]) == 1
        assert result.chunk_results[0]["matches"][0]["id"] == "HP:0001250"
        assert result.aggregated_results[0]["count"] == 1
```

- [ ] **Step 2: Strengthen precomputed bypass tests**

In `tests/unit/text_processing/test_orchestrate_with_precomputed.py`, update the fixture:

```python
@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.detect_index_type.return_value = "multi_vector"
    r.query_batch.return_value = [
        {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"id": "HP:0001250", "label": "Seizure"}]],
            "similarities": [[0.85]],
            "distances": [[0.15]],
            "documents": [[""]],
        }
    ]
    r.query_batch_multi_vector.return_value = [
        {
            "ids": [["HP:9999999"]],
            "metadatas": [[{"id": "HP:9999999", "label": "Wrong"}]],
            "similarities": [[0.99]],
            "distances": [[0.01]],
            "documents": [[""]],
        }
    ]
    return r
```

Update `test_precomputed_skips_retrieval(...)`:

```python
def test_precomputed_skips_retrieval(mock_retriever):
    raw = [
        {
            "ids": [["HP:0001"]],
            "metadatas": [[{"id": "HP:0001", "label": "Foo"}]],
            "similarities": [[0.9]],
            "distances": [[0.1]],
            "documents": [[""]],
        }
    ]
    result = orchestrate_hpo_extraction(
        text_chunks=["any text"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
        precomputed_query_results=raw,
    )

    assert mock_retriever.query_batch.call_count == 0
    assert mock_retriever.query_batch_multi_vector.call_count == 0
    assert result.raw_query_results == raw
    assert any(t["id"] == "HP:0001" for t in result.aggregated_results)
```

Add this length validation test:

```python
def test_precomputed_length_mismatch_still_raises(mock_retriever):
    with pytest.raises(ValueError, match="precomputed_query_results length"):
        orchestrate_hpo_extraction(
            text_chunks=["a", "b"],
            retriever=mock_retriever,
            num_results_per_chunk=1,
            precomputed_query_results=[],
        )

    assert mock_retriever.query_batch.call_count == 0
    assert mock_retriever.query_batch_multi_vector.call_count == 0
```

- [ ] **Step 3: Run orchestrator tests and verify failure**

Run:

```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py::TestRetrieverInteraction tests/unit/text_processing/test_orchestrate_with_precomputed.py -n 0 -v
```

Expected: multi-vector routing and duplicate regression tests fail because the orchestrator still calls `retriever.query_batch(...)`.

- [ ] **Step 4: Implement orchestrator routing**

Add this import to `phentrieve/text_processing/hpo_extraction_orchestrator.py`:

```python
from phentrieve.retrieval.utils import query_chunk_candidates
```

Replace the non-precomputed branch:

```python
    else:
        all_query_results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=text_chunks,
            n_results=num_results_per_chunk,
            include_similarities=True,
        )
```

Do not change the `precomputed_query_results` branch.

- [ ] **Step 5: Run focused orchestrator tests and verify pass**

Run:

```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py -n 0 -v
```

Expected: both orchestrator test files pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add phentrieve/text_processing/hpo_extraction_orchestrator.py tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py
git commit -m "feat: route full-text retrieval by index type"
```

## Task 5: Route Adaptive Rechunking Child Retrieval Through The Shared Helper

**Files:**
- Modify: `phentrieve/retrieval/adaptive_rechunker.py:1-648`
- Test: `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py:1-292`

- [ ] **Step 1: Write failing adaptive routing tests**

Update the `mock_retriever` fixture in `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`:

```python
@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    return retriever
```

Add these tests to `TestRunAdaptiveRechunking`:

```python
    def test_multi_vector_retriever_queries_children_with_query_batch_multi_vector(self):
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        retriever = MagicMock()
        retriever.detect_index_type.return_value = "multi_vector"
        retriever.query_batch_multi_vector.return_value = [make_raw([0.9, 0.5])] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=retriever,
            language="en",
            config=config,
            num_results_per_chunk=6,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        retriever.query_batch_multi_vector.assert_called_once()
        call_kwargs = retriever.query_batch_multi_vector.call_args.kwargs
        assert call_kwargs["n_results"] == 6
        assert call_kwargs["texts"]
        retriever.query_batch.assert_not_called()

    def test_single_vector_retriever_queries_children_with_query_batch(self, mock_retriever):
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        mock_retriever.query_batch.return_value = [make_raw([0.9, 0.5])] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=6,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        mock_retriever.query_batch.assert_called_once()
        assert mock_retriever.query_batch.call_args.kwargs["include_similarities"] is True
        mock_retriever.query_batch_multi_vector.assert_not_called()
```

- [ ] **Step 2: Run adaptive tests and verify failure**

Run:

```bash
uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py::TestRunAdaptiveRechunking -n 0 -v
```

Expected: the multi-vector child retrieval test fails because `run_adaptive_rechunking(...)` still calls `retriever.query_batch(...)`.

- [ ] **Step 3: Implement adaptive child retrieval routing**

Add this import near the top of `phentrieve/retrieval/adaptive_rechunker.py`:

```python
from phentrieve.retrieval.utils import query_chunk_candidates
```

Replace the child retrieval call:

```python
        child_raw = query_chunk_candidates(
            retriever=retriever,
            text_chunks=children_texts,
            n_results=num_results_per_chunk,
            include_similarities=True,
        )
```

Update comments and docstrings in `run_adaptive_rechunking(...)` so the invariant says one retrieval call per recursion level instead of one `query_batch` call per recursion level.

- [ ] **Step 4: Run adaptive tests and verify pass**

Run:

```bash
uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py -n 0 -v
```

Expected: all adaptive rechunking orchestration tests pass, including existing max-depth call-count assertions.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py
git commit -m "feat: route adaptive child retrieval by index type"
```

## Task 6: Focused Regression Verification

**Files:**
- Verify: `phentrieve/retrieval/dense_retriever.py`
- Verify: `phentrieve/retrieval/utils.py`
- Verify: `phentrieve/text_processing/hpo_extraction_orchestrator.py`
- Verify: `phentrieve/retrieval/adaptive_rechunker.py`
- Verify: `phentrieve/text_processing/full_text_service.py`
- Verify: all updated tests listed above

- [ ] **Step 1: Run focused retrieval tests**

Run:

```bash
uv run pytest tests/unit/retrieval/test_dense_retriever_real.py tests/unit/retrieval/test_retrieval_utils.py -n 0 -v
```

Expected: pass.

- [ ] **Step 2: Run focused orchestrator tests**

Run:

```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py -n 0 -v
```

Expected: pass.

- [ ] **Step 3: Run adaptive rechunking tests**

Run:

```bash
uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py -n 0 -v
```

Expected: pass.

- [ ] **Step 4: Run standard full-text service tests**

Run:

```bash
uv run pytest tests/unit/text_processing/test_full_text_service.py -n 0 -v
```

Expected: pass. This verifies the public standard backend response boundary remains stable while retrieval selection moved into the orchestrator.

- [ ] **Step 5: Run the focused duplicate component-hit regression**

Run:

```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py::TestRetrieverInteraction::test_multi_vector_route_prevents_duplicate_component_matches -n 0 -v
```

Expected: pass. This is the local proof that standard full-text chunk processing no longer emits duplicate matches for one HPO ID when a multi-vector retriever is available.

- [ ] **Step 6: Commit any verification-only test adjustments**

If earlier steps required test-only corrections, run:

```bash
git add tests/unit/retrieval/test_dense_retriever_real.py tests/unit/retrieval/test_retrieval_utils.py tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py
git commit -m "test: verify full-text multi-vector parity"
```

Expected: create a commit only when files changed after Task 5.

## Task 7: Self-Review Against The Spec

**Files:**
- Review: `.planning/specs/2026-05-24-full-text-multi-vector-parity-design.md`
- Review: implementation diff

- [ ] **Step 1: Review spec coverage**

Open the spec and verify these mappings:

```text
Add batch multi-vector retrieval for lists of chunks:
  Task 3, DenseRetriever.query_batch_multi_vector.

Use automatically in standard full-text extraction for multi-vector indexes:
  Task 4, orchestrator query_chunk_candidates call.

Preserve single-vector behavior:
  Task 2 single-vector helper test, Task 4 single-vector orchestrator test.

Preserve precomputed_query_results bypass:
  Task 4 precomputed tests and unchanged branch.

Keep public standard full-text API response shape unchanged:
  Task 6 full_text_service tests.

Focused unit tests for retrieval shape, aggregation, fallback, and routing:
  Tasks 1 through 5.

Adaptive child retrieval uses normalized retrieval path:
  Task 5.

query_batch_multi_vector error handling:
  Task 3 failure tests.

include_similarities conversion:
  Task 1.

shared query_chunk_candidates helper:
  Task 2.
```

Expected: every in-scope spec requirement has a named task.

- [ ] **Step 2: Check type and signature consistency**

Verify these signatures match across implementation and tests:

```text
DenseRetriever.query_batch_multi_vector(
    self,
    texts: list[str],
    n_results: int = 10,
    aggregation_strategy: str | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
    component_weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
) -> list[dict[str, Any]]

query_chunk_candidates(
    *,
    retriever: Any,
    text_chunks: list[str],
    n_results: int,
    include_similarities: bool = True,
) -> list[dict[str, Any]]

convert_multi_vector_to_chromadb_format(
    multi_vector_results: list[dict[str, Any]],
    include_similarities: bool = False,
) -> dict[str, Any]
```

Expected: no test calls use a different parameter name.

- [ ] **Step 3: Review for unresolved wording**

Read this plan and the implementation diff once. Replace any vague instruction or unresolved planning note with concrete code, command, or expected behavior before final verification.

Expected: no unresolved planning notes remain.

- [ ] **Step 4: Commit self-review fixes if needed**

If self-review changes implementation or tests, run:

```bash
git add phentrieve/retrieval/dense_retriever.py phentrieve/retrieval/utils.py phentrieve/text_processing/hpo_extraction_orchestrator.py phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/test_dense_retriever_real.py tests/unit/retrieval/test_retrieval_utils.py tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py
git commit -m "fix: address multi-vector parity self-review"
```

Expected: create a commit only when self-review changed files.

## Task 8: Final Repository Verification

**Files:**
- Verify: whole repository

- [ ] **Step 1: Run required local checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all three commands pass before claiming implementation completion.

- [ ] **Step 2: Run optional broader benchmark gates when data and runtime are available**

Run standard full-text GeneReviews extraction with ontology-aware metrics:

```bash
uv run phentrieve benchmark extraction run tests/data/en/phenobert --dataset GeneReviews --ontology-aware-metrics --detailed-output --output-dir results/extraction/full-text-multi-vector-parity
```

Expected: record strict, soft, and partial F1, runtime per document, and failed document count from the generated benchmark artifacts.

Before treating this as multi-vector parity evidence, confirm the benchmark
retriever connected to the intended multi-vector collection. If the benchmark
harness is still configured for a single-vector collection, record this run as
a full-text smoke benchmark and use service-level evidence for the multi-vector
standard backend path.

Run the 570 German direct retrieval comparison:

```bash
uv run phentrieve benchmark compare-vectors --test-file tests/data/benchmarks/german/570terms_german.json --strategies label_synonyms_max,all_max --cpu
```

Expected: direct query multi-vector metrics do not regress relative to the pre-change baseline because `query_multi_vector(...)` semantics are unchanged.

- [ ] **Step 3: Capture final status**

Record in the final implementation summary:

```text
Focused tests:
- retrieval utility tests: pass
- dense retriever tests: pass
- orchestrator tests: pass
- adaptive rechunker tests: pass
- full_text_service tests: pass

Required checks:
- make check: pass
- make typecheck-fast: pass
- make test: pass

Optional benchmarks:
- GeneReviews standard full-text: run or skipped with reason
- 570 German direct retrieval: run or skipped with reason
```

Expected: the final summary names any skipped optional benchmark and gives the reason.
