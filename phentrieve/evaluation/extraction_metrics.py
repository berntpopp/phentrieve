"""Extraction metrics for document-level HPO extraction evaluation."""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from phentrieve.evaluation.ontology_credit import OntologyCreditConfig
    from phentrieve.evaluation.ontology_matching import DocumentOntologyMetrics


@dataclass
class ExtractionResult:
    """Single document extraction result."""

    doc_id: str
    predicted: list[tuple[str, str]]  # (hpo_id, assertion)
    gold: list[tuple[str, str]]  # (hpo_id, assertion)


@dataclass
class CorpusMetrics:
    """Corpus-level evaluation metrics."""

    micro: dict[str, float]  # precision, recall, f1
    macro: dict[str, float]  # precision, recall, f1
    weighted: dict[str, float]  # weighted by doc size
    confidence_intervals: dict[str, tuple[float, float]]


class MatchBreakdownEntry(TypedDict):
    """Corpus-level count and credit total for one ontology match kind."""

    count: int
    credit: float


@dataclass
class OntologyMetricBlock:
    """Ontology-aware corpus metrics for one matching mode."""

    micro: dict[str, float]
    macro: dict[str, float]
    weighted: dict[str, float]


@dataclass
class OntologyAwareCorpusMetrics:
    """Corpus-level ontology-aware extraction metrics."""

    strict: OntologyMetricBlock
    soft: OntologyMetricBlock
    partial: OntologyMetricBlock
    match_breakdown: dict[str, MatchBreakdownEntry]
    document_metrics: list["DocumentOntologyMetrics"]


def _calculate_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculate precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _doc_metrics(result: ExtractionResult) -> tuple[float, float, float, int]:
    """Calculate metrics for a single document.

    Returns: (precision, recall, f1, gold_count)
    """
    pred_set = set(result.predicted)
    gold_set = set(result.gold)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision, recall, f1 = _calculate_prf(tp, fp, fn)
    return precision, recall, f1, len(gold_set)


def _zero_metrics() -> dict[str, float]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _zero_ontology_block() -> OntologyMetricBlock:
    return OntologyMetricBlock(
        micro=_zero_metrics(),
        macro=_zero_metrics(),
        weighted=_zero_metrics(),
    )


def _ontology_block_from_corpus_metrics(metrics: CorpusMetrics) -> OntologyMetricBlock:
    return OntologyMetricBlock(
        micro=dict(metrics.micro),
        macro=dict(metrics.macro),
        weighted=dict(metrics.weighted),
    )


def _harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _average_metric_dicts(
    metrics: list[dict[str, float]],
    weights: list[int] | None = None,
) -> dict[str, float]:
    if not metrics:
        return _zero_metrics()

    if weights is None:
        weights = [1] * len(metrics)

    total_weight = sum(weights)
    if total_weight == 0:
        return _zero_metrics()

    def weighted_sum(metric_name: str) -> float:
        return sum(
            metric[metric_name] * weight
            for metric, weight in zip(metrics, weights, strict=True)
        )

    return {
        "precision": weighted_sum("precision") / total_weight,
        "recall": weighted_sum("recall") / total_weight,
        "f1": weighted_sum("f1") / total_weight,
    }


def _block(metrics: OntologyMetricBlock) -> dict[str, dict[str, float]]:
    return {
        "micro": dict(metrics.micro),
        "macro": dict(metrics.macro),
        "weighted": dict(metrics.weighted),
    }


def serialize_ontology_metrics(
    metrics: OntologyAwareCorpusMetrics,
) -> dict[str, Any]:
    return {
        "strict": _block(metrics.strict),
        "soft": _block(metrics.soft),
        "partial": _block(metrics.partial),
        "match_breakdown": metrics.match_breakdown,
    }


def calculate_document_ontology_metrics(
    result: ExtractionResult,
    config: "OntologyCreditConfig | None" = None,
) -> "DocumentOntologyMetrics":
    from phentrieve.evaluation.ontology_matching import (
        calculate_document_ontology_metrics as calculate,
    )

    return calculate(result, config)


class CorpusExtractionMetrics:
    """Document-level HPO extraction evaluation."""

    def __init__(self, averaging: str = "micro"):
        self.averaging = averaging

    def calculate_metrics(
        self,
        results: list[ExtractionResult],
    ) -> CorpusMetrics:
        """Calculate corpus-level metrics with the configured averaging strategy."""
        if self.averaging == "micro":
            return self._calculate_micro_metrics(results)
        elif self.averaging == "macro":
            return self._calculate_macro_metrics(results)
        elif self.averaging == "weighted":
            return self._calculate_weighted_metrics(results)
        else:
            raise ValueError(f"Unknown averaging strategy: {self.averaging}")

    def calculate_all_metrics(
        self,
        results: list[ExtractionResult],
    ) -> CorpusMetrics:
        """Calculate all averaging strategies at once."""
        micro = self._compute_micro(results)
        macro = self._compute_macro(results)
        weighted = self._compute_weighted(results)

        return CorpusMetrics(
            micro=micro,
            macro=macro,
            weighted=weighted,
            confidence_intervals={},
        )

    def calculate_ontology_aware_metrics(
        self,
        results: list[ExtractionResult],
        config: "OntologyCreditConfig | None" = None,
    ) -> OntologyAwareCorpusMetrics:
        """Calculate strict and ontology-aware corpus extraction metrics."""
        if not results:
            return OntologyAwareCorpusMetrics(
                strict=_zero_ontology_block(),
                soft=_zero_ontology_block(),
                partial=_zero_ontology_block(),
                match_breakdown={},
                document_metrics=[],
            )

        document_metrics = [
            calculate_document_ontology_metrics(result, config) for result in results
        ]
        strict = _ontology_block_from_corpus_metrics(
            self.calculate_all_metrics(results)
        )

        total_soft_tp = sum(metric.soft_tp for metric in document_metrics)
        total_soft_fp = sum(metric.soft_fp for metric in document_metrics)
        total_soft_fn = sum(metric.soft_fn for metric in document_metrics)
        soft_precision = (
            total_soft_tp / (total_soft_tp + total_soft_fp)
            if total_soft_tp + total_soft_fp > 0
            else 0.0
        )
        soft_recall = (
            total_soft_tp / (total_soft_tp + total_soft_fn)
            if total_soft_tp + total_soft_fn > 0
            else 0.0
        )
        soft_micro = {
            "precision": soft_precision,
            "recall": soft_recall,
            "f1": _harmonic_mean(soft_precision, soft_recall),
        }

        partial_precision_denominator = sum(
            metric.prediction_count for metric in document_metrics
        )
        partial_recall_denominator = sum(
            metric.gold_count for metric in document_metrics
        )
        partial_precision = (
            sum(
                metric.partial_precision * metric.prediction_count
                for metric in document_metrics
            )
            / partial_precision_denominator
            if partial_precision_denominator > 0
            else 0.0
        )
        partial_recall = (
            sum(
                metric.partial_recall * metric.gold_count for metric in document_metrics
            )
            / partial_recall_denominator
            if partial_recall_denominator > 0
            else 0.0
        )
        partial_micro = {
            "precision": partial_precision,
            "recall": partial_recall,
            "f1": _harmonic_mean(partial_precision, partial_recall),
        }

        soft_document_metrics = [
            {
                "precision": metric.soft_precision,
                "recall": metric.soft_recall,
                "f1": metric.soft_f1,
            }
            for metric in document_metrics
        ]
        partial_document_metrics = [
            {
                "precision": metric.partial_precision,
                "recall": metric.partial_recall,
                "f1": metric.partial_f1,
            }
            for metric in document_metrics
        ]
        weights = [max(metric.gold_count, 1) for metric in document_metrics]

        soft = OntologyMetricBlock(
            micro=soft_micro,
            macro=_average_metric_dicts(soft_document_metrics),
            weighted=_average_metric_dicts(soft_document_metrics, weights),
        )
        partial = OntologyMetricBlock(
            micro=partial_micro,
            macro=_average_metric_dicts(partial_document_metrics),
            weighted=_average_metric_dicts(partial_document_metrics, weights),
        )

        match_breakdown: dict[str, MatchBreakdownEntry] = {}
        for metric in document_metrics:
            for match in metric.matches:
                kind = match.credit.match_kind.value
                if kind not in match_breakdown:
                    match_breakdown[kind] = {"count": 0, "credit": 0.0}
                match_breakdown[kind]["count"] += 1
                match_breakdown[kind]["credit"] += match.credit.credit

        return OntologyAwareCorpusMetrics(
            strict=strict,
            soft=soft,
            partial=partial,
            match_breakdown=match_breakdown,
            document_metrics=document_metrics,
        )

    def _compute_micro(self, results: list[ExtractionResult]) -> dict[str, float]:
        """Compute micro-averaged metrics (aggregate all counts first)."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for result in results:
            pred_set = set(result.predicted)
            gold_set = set(result.gold)

            total_tp += len(pred_set & gold_set)
            total_fp += len(pred_set - gold_set)
            total_fn += len(gold_set - pred_set)

        precision, recall, f1 = _calculate_prf(total_tp, total_fp, total_fn)
        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_macro(self, results: list[ExtractionResult]) -> dict[str, float]:
        """Compute macro-averaged metrics (average per-document metrics)."""
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precisions = []
        recalls = []
        f1s = []

        for result in results:
            p, r, f, _ = _doc_metrics(result)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        return {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s),
        }

    def _compute_weighted(self, results: list[ExtractionResult]) -> dict[str, float]:
        """Compute weighted-averaged metrics (weighted by gold set size)."""
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        total_weight = 0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

        for result in results:
            p, r, f, weight = _doc_metrics(result)
            # Use gold set size as weight (at least 1 to avoid division issues)
            weight = max(weight, 1)
            total_weight += weight
            weighted_precision += p * weight
            weighted_recall += r * weight
            weighted_f1 += f * weight

        if total_weight == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        return {
            "precision": weighted_precision / total_weight,
            "recall": weighted_recall / total_weight,
            "f1": weighted_f1 / total_weight,
        }

    def _calculate_micro_metrics(
        self, results: list[ExtractionResult]
    ) -> CorpusMetrics:
        """Calculate micro-averaged metrics."""
        micro = self._compute_micro(results)
        return CorpusMetrics(
            micro=micro,
            macro={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            weighted={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            confidence_intervals={},
        )

    def _calculate_macro_metrics(
        self, results: list[ExtractionResult]
    ) -> CorpusMetrics:
        """Calculate macro-averaged metrics."""
        macro = self._compute_macro(results)
        return CorpusMetrics(
            micro={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            macro=macro,
            weighted={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            confidence_intervals={},
        )

    def _calculate_weighted_metrics(
        self, results: list[ExtractionResult]
    ) -> CorpusMetrics:
        """Calculate weighted-averaged metrics."""
        weighted = self._compute_weighted(results)
        return CorpusMetrics(
            micro={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            macro={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            weighted=weighted,
            confidence_intervals={},
        )

    def bootstrap_confidence_intervals(
        self,
        results: list[ExtractionResult],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics."""
        if not results:
            return {
                "precision": (0.0, 0.0),
                "recall": (0.0, 0.0),
                "f1": (0.0, 0.0),
            }

        # Bootstrap sampling
        bootstrap_metrics: dict[str, list[float]] = {
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for _ in range(n_bootstrap):
            # Sample with replacement (S311: crypto not needed for statistical sampling)
            sample = random.choices(results, k=len(results))  # noqa: S311
            metrics = self._compute_micro(sample)

            bootstrap_metrics["precision"].append(metrics["precision"])
            bootstrap_metrics["recall"].append(metrics["recall"])
            bootstrap_metrics["f1"].append(metrics["f1"])

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        confidence_intervals = {}
        for metric_name, values in bootstrap_metrics.items():
            sorted_values = sorted(values)
            lower_idx = int(lower_percentile / 100 * len(sorted_values))
            upper_idx = int(upper_percentile / 100 * len(sorted_values)) - 1

            # Clamp indices
            lower_idx = max(0, min(lower_idx, len(sorted_values) - 1))
            upper_idx = max(0, min(upper_idx, len(sorted_values) - 1))

            confidence_intervals[metric_name] = (
                sorted_values[lower_idx],
                sorted_values[upper_idx],
            )

        return confidence_intervals
