"""QueryRequest uses the shared AggregationStrategy / AssertionPreference types (#427)."""

from typing import get_args

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from api.schemas.query_schemas import QueryRequest
from phentrieve.config import DEFAULT_AGGREGATION_STRATEGY, AssertionPreference
from phentrieve.retrieval.aggregation import AggregationStrategy

pytestmark = pytest.mark.unit


def test_aggregation_strategy_is_enum_and_defaults_to_config() -> None:
    request = QueryRequest(text="seizures")

    assert isinstance(request.aggregation_strategy, AggregationStrategy)
    assert request.aggregation_strategy == DEFAULT_AGGREGATION_STRATEGY


@pytest.mark.parametrize("strategy", [s.value for s in AggregationStrategy])
def test_aggregation_strategy_accepts_every_enum_value(strategy: str) -> None:
    request = QueryRequest(text="seizures", aggregation_strategy=strategy)

    assert request.aggregation_strategy is AggregationStrategy(strategy)


def test_aggregation_strategy_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(text="seizures", aggregation_strategy="bogus")


def test_query_assertion_preference_accepts_shared_alias_values() -> None:
    for preference in get_args(AssertionPreference):
        request = QueryRequest(text="seizures", query_assertion_preference=preference)
        assert request.query_assertion_preference == preference

    with pytest.raises(ValidationError):
        QueryRequest(text="seizures", query_assertion_preference="bogus")


def test_json_schema_keeps_inline_string_enums() -> None:
    props = QueryRequest.model_json_schema()["properties"]

    aggregation = props["aggregation_strategy"]
    assert aggregation["anyOf"] == [
        {"type": "string", "enum": [s.value for s in AggregationStrategy]},
        {"type": "null"},
    ]
    assert aggregation["default"] == DEFAULT_AGGREGATION_STRATEGY
    assert aggregation["title"] == "Aggregation Strategy"
    assert props["query_assertion_preference"]["enum"] == list(
        get_args(AssertionPreference)
    )


@pytest.mark.parametrize("strategy", ["custom", "bogus"])
def test_compare_vectors_rejects_non_preset_strategies(strategy: str) -> None:
    from phentrieve.cli.benchmark_commands import app

    result = CliRunner().invoke(app, ["compare-vectors", "--strategies", strategy])

    assert result.exit_code == 1
    assert "Invalid strategy" in result.output
    assert "label_synonyms_max" in result.output
    assert "custom" not in result.output.split("Valid options:")[1]
