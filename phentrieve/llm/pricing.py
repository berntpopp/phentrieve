"""LLM cost estimation based on token usage and model pricing.

Loads per-model pricing from ``pricing.yaml`` and estimates costs for
benchmark runs. Returns ``None`` gracefully when a model has no pricing data.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from phentrieve.llm.types import TokenUsage

logger = logging.getLogger(__name__)

_PRICING_FILE = Path(__file__).parent / "pricing.yaml"


@lru_cache(maxsize=1)
def load_pricing() -> dict[str, dict[str, float]]:
    """Load the pricing table from ``pricing.yaml``.

    Returns:
        Dict mapping model name to ``{input_per_1m, output_per_1m}``.
    """
    try:
        import yaml

        with open(_PRICING_FILE) as f:
            data = yaml.safe_load(f)
        return data.get("models", {})  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("Could not load pricing data: %s", exc)
        return {}


def estimate_cost(token_usage: TokenUsage, model: str) -> dict[str, float] | None:
    """Estimate the USD cost for a given token usage and model.

    Args:
        token_usage: Accumulated token counts.
        model: The LiteLLM model identifier (e.g. ``gemini/gemini-2.0-flash``).

    Returns:
        Dict with ``input_cost``, ``output_cost``, ``total_cost`` in USD,
        or ``None`` if the model is not in the pricing table.
    """
    pricing = load_pricing()
    model_pricing = pricing.get(model)
    if model_pricing is None:
        return None

    input_cost = token_usage.prompt_tokens * model_pricing["input_per_1m"] / 1_000_000
    output_cost = (
        token_usage.completion_tokens * model_pricing["output_per_1m"] / 1_000_000
    )
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }
