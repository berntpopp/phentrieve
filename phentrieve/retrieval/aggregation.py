"""
Aggregation strategies for multi-vector HPO term retrieval.

This module provides configurable strategies for combining similarity scores
from multiple component vectors (label, synonyms, definition) into a single
score per HPO term.

See issue #136 for design details.
"""

import ast
import logging
import operator
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Preset aggregation strategies for multi-vector results."""

    LABEL_ONLY = "label_only"
    LABEL_SYNONYMS_MIN = "label_synonyms_min"
    LABEL_SYNONYMS_MAX = "label_synonyms_max"
    ALL_WEIGHTED = "all_weighted"
    ALL_MAX = "all_max"
    ALL_MIN = "all_min"
    CUSTOM = "custom"


# Default component weights for ALL_WEIGHTED strategy
DEFAULT_COMPONENT_WEIGHTS = {
    "label": 0.5,
    "synonyms": 0.3,
    "definition": 0.2,
}


def aggregate_scores(
    label_score: float | None,
    synonym_scores: list[float],
    definition_score: float | None,
    strategy: str | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
    weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
) -> float:
    """
    Aggregate component scores into a single HPO term score.

    Args:
        label_score: Similarity score for the label vector (None if not retrieved)
        synonym_scores: List of similarity scores for synonym vectors
        definition_score: Similarity score for definition vector (None if not present)
        strategy: Aggregation strategy to use
        weights: Component weights for ALL_WEIGHTED strategy
        custom_formula: Custom formula string for CUSTOM strategy

    Returns:
        Aggregated similarity score (0.0 to 1.0)

    Raises:
        ValueError: If strategy is invalid or custom formula is malformed
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = AggregationStrategy(strategy)
        except ValueError:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Valid options: {[s.value for s in AggregationStrategy]}"
            )

    # Collect all available scores
    all_scores: list[float] = []
    if label_score is not None:
        all_scores.append(label_score)
    all_scores.extend(synonym_scores)
    if definition_score is not None:
        all_scores.append(definition_score)

    # Handle edge case: no scores available
    if not all_scores:
        return 0.0

    # Apply strategy
    if strategy == AggregationStrategy.LABEL_ONLY:
        return label_score if label_score is not None else 0.0

    elif strategy == AggregationStrategy.LABEL_SYNONYMS_MIN:
        # Minimum (worst) score from label or synonyms - conservative approach
        label_syn_scores = []
        if label_score is not None:
            label_syn_scores.append(label_score)
        label_syn_scores.extend(synonym_scores)
        return min(label_syn_scores) if label_syn_scores else 0.0

    elif strategy == AggregationStrategy.LABEL_SYNONYMS_MAX:
        # Best match between label and best synonym
        label_syn_scores = []
        if label_score is not None:
            label_syn_scores.append(label_score)
        label_syn_scores.extend(synonym_scores)
        return max(label_syn_scores) if label_syn_scores else 0.0

    elif strategy == AggregationStrategy.ALL_WEIGHTED:
        weights = weights or DEFAULT_COMPONENT_WEIGHTS
        return _weighted_aggregate(
            label_score, synonym_scores, definition_score, weights
        )

    elif strategy == AggregationStrategy.ALL_MAX:
        return max(all_scores)

    elif strategy == AggregationStrategy.ALL_MIN:
        return min(all_scores)

    elif strategy == AggregationStrategy.CUSTOM:
        if not custom_formula:
            raise ValueError("custom_formula required for CUSTOM strategy")
        return _evaluate_custom_formula(
            custom_formula, label_score, synonym_scores, definition_score
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _weighted_aggregate(
    label_score: float | None,
    synonym_scores: list[float],
    definition_score: float | None,
    weights: dict[str, float],
) -> float:
    """
    Compute weighted aggregate of component scores.

    Uses max(synonyms) as the synonym component score.
    Missing components contribute 0 weighted by their weight.
    """
    total = 0.0
    weight_sum = 0.0

    # Label component
    label_weight = weights.get("label", 0.5)
    if label_score is not None:
        total += label_weight * label_score
        weight_sum += label_weight

    # Synonym component (use max of all synonyms)
    syn_weight = weights.get("synonyms", 0.3)
    if synonym_scores:
        total += syn_weight * max(synonym_scores)
        weight_sum += syn_weight

    # Definition component
    def_weight = weights.get("definition", 0.2)
    if definition_score is not None:
        total += def_weight * definition_score
        weight_sum += def_weight

    # Normalize by actual weights used (handle missing components)
    return total / weight_sum if weight_sum > 0 else 0.0


# Safe functions for custom formula evaluation
# Note: We wrap min/max to handle single-argument case (e.g., max(0.9) fails in Python)
_SAFE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "min": lambda *args: min(args) if len(args) > 1 else (args[0] if args else 0.0),
    "max": lambda *args: max(args) if len(args) > 1 else (args[0] if args else 0.0),
    "avg": lambda *args: sum(args) / len(args) if args else 0.0,
}

# Safe operators for custom formula evaluation
_SAFE_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@lru_cache(maxsize=32)
def _parse_formula(formula: str) -> ast.Expression:
    """Parse and cache a custom formula AST."""
    try:
        tree = ast.parse(formula, mode="eval")
        return tree
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}")


def _evaluate_custom_formula(
    formula: str,
    label_score: float | None,
    synonym_scores: list[float],
    definition_score: float | None,
) -> float:
    """
    Safely evaluate a custom aggregation formula.

    Allowed functions: min(), max(), avg()
    Allowed variables: label, synonyms (list), definition
    Allowed operators: +, -, *, /

    Examples:
        "0.5 * max(label, max(synonyms)) + 0.5 * definition"
        "max(label, max(synonyms), definition)"
        "avg(label, max(synonyms))"
    """
    tree = _parse_formula(formula)

    # Build variable context
    context = {
        "label": label_score if label_score is not None else 0.0,
        "synonyms": synonym_scores if synonym_scores else [0.0],
        "definition": definition_score if definition_score is not None else 0.0,
    }

    try:
        result = _safe_eval(tree.body, context)
        # Handle case where formula returns a list (shouldn't happen in well-formed formulas)
        if isinstance(result, list):
            return result[0] if result else 0.0
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating formula '{formula}': {e}")


def _safe_eval(node: ast.AST, context: dict[str, Any]) -> float | list[float]:
    """Safely evaluate an AST node with restricted operations.

    Returns a float for most expressions, or a list[float] for the 'synonyms' variable.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Invalid constant type: {type(node.value)}")

    elif isinstance(node, ast.Name):
        if node.id in context:
            value = context[node.id]
            # synonyms is a list, everything else is float
            if isinstance(value, list):
                return value
            return float(value)
        raise ValueError(f"Unknown variable: {node.id}")

    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, context)
        right = _safe_eval(node.right, context)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return float(_SAFE_OPERATORS[op_type](left, right))
        raise ValueError(f"Unsupported operator: {op_type.__name__}")

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")

        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(
                f"Unknown function: {func_name}. Allowed: {list(_SAFE_FUNCTIONS.keys())}"
            )

        # Evaluate arguments
        args = []
        for arg in node.args:
            val = _safe_eval(arg, context)
            # Handle lists (for synonyms)
            if isinstance(val, list):
                args.extend(val)
            else:
                args.append(val)

        return _SAFE_FUNCTIONS[func_name](*args)

    elif isinstance(node, ast.IfExp):
        # Support ternary: x if condition else y
        # Mainly for "definition or 0.0" style expressions
        test = _safe_eval(node.test, context)
        if test:
            return _safe_eval(node.body, context)
        return _safe_eval(node.orelse, context)

    elif isinstance(node, ast.BoolOp):
        # Support "or" for default values: "definition or 0.0"
        if isinstance(node.op, ast.Or):
            for bool_val in node.values:
                result = _safe_eval(bool_val, context)
                if result:
                    # Ensure we return a float, not a list
                    if isinstance(result, list):
                        return result[0] if result else 0.0
                    return result
            return 0.0
        raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def group_results_by_hpo_id(
    results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Group multi-vector ChromaDB results by HPO ID.

    Args:
        results: ChromaDB query results with keys:
            - ids: List of document IDs
            - metadatas: List of metadata dicts with 'hpo_id', 'component'
            - distances or similarities: List of scores

    Returns:
        Dictionary mapping HPO IDs to component scores:
        {
            "HP:0001250": {
                "label": 0.85,
                "synonyms": [0.92, 0.78],
                "definition": 0.65,
                "label_text": "Seizure",
            }
        }
    """
    grouped: dict[str, dict[str, Any]] = {}

    # Handle both single query and batch query result formats
    ids_list = results.get("ids", [[]])[0] if results.get("ids") else []
    metadatas_list = (
        results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    )

    # Get scores - prefer similarities over distances
    if "similarities" in results and results["similarities"]:
        scores_list = results["similarities"][0]
    elif "distances" in results and results["distances"]:
        # Convert distances to similarities
        scores_list = [1.0 - d for d in results["distances"][0]]
    else:
        scores_list = []

    for i, (_, metadata) in enumerate(zip(ids_list, metadatas_list)):
        hpo_id = metadata.get("hpo_id", "")
        component = metadata.get("component", "unknown")
        score = scores_list[i] if i < len(scores_list) else 0.0
        label_text = metadata.get("label", "")

        if hpo_id not in grouped:
            grouped[hpo_id] = {
                "label": None,
                "synonyms": [],
                "definition": None,
                "label_text": label_text,
            }

        if component == "label":
            grouped[hpo_id]["label"] = score
        elif component == "synonym":
            grouped[hpo_id]["synonyms"].append(score)
        elif component == "definition":
            grouped[hpo_id]["definition"] = score

        # Preserve label text
        if label_text and not grouped[hpo_id]["label_text"]:
            grouped[hpo_id]["label_text"] = label_text

    return grouped


def aggregate_multi_vector_results(
    results: dict[str, Any],
    strategy: str | AggregationStrategy = AggregationStrategy.LABEL_SYNONYMS_MAX,
    weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Aggregate multi-vector ChromaDB results into per-HPO-term scores.

    Args:
        results: Raw ChromaDB query results
        strategy: Aggregation strategy to use
        weights: Component weights for ALL_WEIGHTED strategy
        custom_formula: Custom formula for CUSTOM strategy
        min_similarity: Minimum aggregated similarity threshold

    Returns:
        List of result dictionaries sorted by aggregated score:
        [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.92,
                "component_scores": {
                    "label": 0.85,
                    "synonyms": [0.92, 0.78],
                    "definition": 0.65,
                },
            }
        ]
    """
    # Group by HPO ID
    grouped = group_results_by_hpo_id(results)

    # Aggregate scores for each HPO term
    aggregated = []
    for hpo_id, components in grouped.items():
        score = aggregate_scores(
            label_score=components["label"],
            synonym_scores=components["synonyms"],
            definition_score=components["definition"],
            strategy=strategy,
            weights=weights,
            custom_formula=custom_formula,
        )

        if score >= min_similarity:
            aggregated.append(
                {
                    "hpo_id": hpo_id,
                    "label": components["label_text"],
                    "similarity": score,
                    "component_scores": {
                        "label": components["label"],
                        "synonyms": components["synonyms"],
                        "definition": components["definition"],
                    },
                }
            )

    # Sort by aggregated similarity descending
    aggregated.sort(key=lambda x: x["similarity"], reverse=True)

    return aggregated
