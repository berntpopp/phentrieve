from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from phentrieve.config import DEFAULT_MODEL
from phentrieve.llm.config import (
    DEFAULT_LLM_MULTI_VECTOR_AGGREGATION_STRATEGY,
    DEFAULT_LLM_RETRIEVAL_BATCH_SIZE,
    DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK,
    DEFAULT_TOOL_QUERY_RESULTS,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        text_processor: Any | None = None,
        max_num_results: int = DEFAULT_TOOL_QUERY_RESULTS,
        retrieval_batch_size: int = DEFAULT_LLM_RETRIEVAL_BATCH_SIZE,
        model_name: str = DEFAULT_MODEL,
        model_revision: str | None = None,
        trust_remote_code: bool = False,
        code_revision: str | None = None,
        index_dir: str | Path | None = None,
        multi_vector: bool = True,
        multi_vector_aggregation_strategy: str = (
            DEFAULT_LLM_MULTI_VECTOR_AGGREGATION_STRATEGY
        ),
    ) -> None:
        self._retriever = retriever
        self._text_processor = text_processor
        self.max_num_results = max_num_results
        self.retrieval_batch_size = retrieval_batch_size
        self.model_name = model_name
        self.model_revision = model_revision
        self.trust_remote_code = trust_remote_code
        self.code_revision = code_revision
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.multi_vector = multi_vector
        self.multi_vector_aggregation_strategy = multi_vector_aggregation_strategy
        self._cached_embedding_model: Any | None = None
        self._cached_retriever: Any | None = None
        self._cached_pipelines: dict[str, Any] = {}

    def warmup(self, *, language: str) -> None:
        self._get_phentrieve_components(language)

    def query_hpo_terms(
        self,
        *,
        query: str,
        num_results: int = DEFAULT_TOOL_QUERY_RESULTS,
    ) -> list[dict[str, Any]]:
        _embedding_model, _pipeline, retriever = self._get_phentrieve_components("en")
        if retriever is None:
            return []

        capped_results = min(int(num_results), self.max_num_results)
        if self.multi_vector and hasattr(retriever, "query_multi_vector"):
            multi_vector_results = retriever.query_multi_vector(
                query,
                n_results=capped_results,
                aggregation_strategy=self.multi_vector_aggregation_strategy,
            )
            return [
                {
                    "hpo_id": result.get("hpo_id", ""),
                    "term_name": result.get("label", ""),
                    "score": float(result.get("similarity", 0.0)),
                }
                for result in multi_vector_results
            ]

        raw = retriever.query(query, n_results=capped_results)
        raw_metadatas = raw.get("metadatas") or []
        raw_similarities = raw.get("similarities") or []
        metadatas = raw_metadatas[0] if raw_metadatas else []
        similarities = raw_similarities[0] if raw_similarities else []

        results: list[dict[str, Any]] = []
        for index, metadata in enumerate(metadatas):
            similarity = similarities[index] if index < len(similarities) else 0.0
            results.append(
                {
                    "hpo_id": metadata.get("hpo_id", ""),
                    "term_name": metadata.get("label", ""),
                    "score": float(similarity),
                }
            )
        return results

    def query_batch_hpo_terms(
        self,
        *,
        phrases: list[str],
        language: str,
        n_results: int,
    ) -> list[dict[str, Any]]:
        _embedding_model, _pipeline, retriever = self._get_phentrieve_components(
            language
        )
        if retriever is None:
            return []
        if self.multi_vector and hasattr(retriever, "query_multi_vector"):
            results: list[dict[str, Any]] = []
            capped_results = min(int(n_results), self.max_num_results)
            for phrase in phrases:
                multi_vector_results = retriever.query_multi_vector(
                    phrase,
                    n_results=capped_results,
                    aggregation_strategy=self.multi_vector_aggregation_strategy,
                )
                results.append(
                    {
                        "phrase": phrase,
                        "candidates": [
                            {
                                "hpo_id": result.get("hpo_id", ""),
                                "term_name": result.get("label", ""),
                                "score": float(result.get("similarity", 0.0)),
                                "matched_text": result.get("matched_text"),
                                "matched_component": result.get("matched_component"),
                            }
                            for result in multi_vector_results
                        ],
                    }
                )
            return results
        if hasattr(retriever, "query_batch"):
            batch_results: list[dict[str, Any]] = []
            for start in range(0, len(phrases), self.retrieval_batch_size):
                batch_phrases = phrases[start : start + self.retrieval_batch_size]
                batch_results.extend(
                    list(retriever.query_batch(batch_phrases, n_results=n_results))
                )
            return batch_results
        fallback_results: list[dict[str, Any]] = []
        for phrase in phrases:
            phrase_results = self.query_hpo_terms(query=phrase, num_results=n_results)
            fallback_results.append(
                {
                    "metadatas": [
                        [
                            {
                                "hpo_id": result.get("hpo_id", ""),
                                "label": result.get("term_name", ""),
                            }
                            for result in phrase_results
                        ]
                    ],
                    "similarities": [
                        [float(result.get("score", 0.0)) for result in phrase_results]
                    ],
                }
            )
        return fallback_results

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "query_hpo_terms":
            return self.query_hpo_terms(
                query=str(arguments.get("query", "")),
                num_results=int(
                    arguments.get("num_results", DEFAULT_TOOL_QUERY_RESULTS)
                ),
            )
        if tool_name == "process_clinical_text":
            return self._process_clinical_text(
                text=str(arguments.get("text", "")),
                language=str(arguments.get("language", "auto")),
                num_results_per_chunk=int(
                    arguments.get(
                        "num_results_per_chunk",
                        DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK,
                    )
                ),
                chunk_retrieval_threshold=float(
                    arguments.get(
                        "chunk_retrieval_threshold",
                        DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD,
                    )
                ),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def _process_clinical_text(
        self,
        *,
        text: str,
        language: str = "auto",
        num_results_per_chunk: int = (
            DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK
        ),
        chunk_retrieval_threshold: float = (
            DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD
        ),
    ) -> list[dict[str, Any]]:
        if self._text_processor is not None:
            processed = self._text_processor.process(text)
            return [
                {
                    "hpo_id": item.get("hpo_id", ""),
                    "term_name": item.get("term_name", ""),
                    "assertion": item.get("assertion", "affirmed"),
                    "score": item.get("score", 0.0),
                    "evidence_text": item.get("evidence_text"),
                }
                for item in processed
            ]

        try:
            from phentrieve.text_processing.hpo_extraction_orchestrator import (
                orchestrate_hpo_extraction,
            )
        except Exception as exc:  # pragma: no cover - dependency guard
            logger.warning("[TOOL] process_clinical_text unavailable: %s", exc)
            return []

        _embedding_model, pipeline, retriever = self._get_phentrieve_components(
            language
        )
        if retriever is None:
            return []

        chunks = pipeline.process(text)
        if not chunks:
            return []

        chunk_texts = [chunk["text"] for chunk in chunks]
        assertion_statuses: list[str | None] = []
        for chunk in chunks:
            status = chunk.get("status")
            if status is None:
                assertion_statuses.append("affirmed")
            elif isinstance(status, str):
                assertion_statuses.append(status)
            else:
                assertion_statuses.append(status.value)

        aggregated_results, _chunk_results = orchestrate_hpo_extraction(
            text_chunks=chunk_texts,
            retriever=retriever,
            assertion_statuses=assertion_statuses,
            language=language if language != "auto" else "en",
            num_results_per_chunk=num_results_per_chunk,
            chunk_retrieval_threshold=chunk_retrieval_threshold,
        )

        output: list[dict[str, Any]] = []
        for result in aggregated_results:
            assertion = result.get("assertion_status") or "affirmed"
            evidence_parts = [
                attr.get("chunk_text", "")
                for attr in result.get("text_attributions", [])
                if attr.get("chunk_text", "")
            ]
            output.append(
                {
                    "hpo_id": result.get("id", ""),
                    "term_name": result.get("name", ""),
                    "assertion": assertion,
                    "score": result.get("score", 0.0),
                    "evidence_text": "; ".join(evidence_parts)
                    if evidence_parts
                    else "",
                }
            )
        return output

    def _get_phentrieve_components(self, language: str) -> tuple[Any, Any, Any]:
        from phentrieve.config import get_sliding_window_punct_conj_cleaned_config
        from phentrieve.embeddings import load_embedding_model
        from phentrieve.retrieval.dense_retriever import DenseRetriever
        from phentrieve.text_processing.pipeline import TextProcessingPipeline

        language_key = language if language != "auto" else "en"

        if self._retriever is not None:
            self._cached_retriever = self._retriever

        if self._cached_embedding_model is None:
            self._cached_embedding_model = load_embedding_model(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.model_revision,
                code_revision=self.code_revision,
            )

        if language_key not in self._cached_pipelines:
            self._cached_pipelines[language_key] = TextProcessingPipeline(
                language=language_key,
                chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
                assertion_config={"disable": False},
                sbert_model_for_semantic_chunking=self._cached_embedding_model,
            )

        if self._cached_retriever is None:
            self._cached_retriever = DenseRetriever.from_model_name(
                model=self._cached_embedding_model,
                model_name=self.model_name,
                index_dir=self.index_dir,
                multi_vector=self.multi_vector,
            )

        return (
            self._cached_embedding_model,
            self._cached_pipelines[language_key],
            self._cached_retriever,
        )
