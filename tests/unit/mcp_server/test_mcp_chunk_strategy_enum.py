"""L4: chunk_text strategy is an enumerated, validated parameter and the valid
strategies are advertised in capabilities."""

import typing

import pytest

from api.mcp.capabilities import build_capabilities
from api.mcp.envelope import McpToolError
from api.mcp.service_adapters import chunk_text_service
from api.mcp.tools._common import ChunkStrategy
from phentrieve.text_processing.config_resolver import KNOWN_CHUNK_STRATEGIES

pytestmark = pytest.mark.unit


def test_chunkstrategy_literal_matches_source_of_truth():
    """Drift guard: the MCP enum members equal config_resolver's known set."""
    literal = ChunkStrategy.__args__[0]  # unwrap Annotated -> Literal
    members = set(typing.get_args(literal))
    assert members == set(KNOWN_CHUNK_STRATEGIES)


def test_unknown_strategy_rejected_with_allowed_values():
    with pytest.raises(McpToolError) as ei:
        chunk_text_service(text="x", language="en", strategy="bogus-strategy")
    assert ei.value.error_code == "validation_failed"
    assert "simple" in str(ei.value)
    assert ei.value.details.get("allowed_values")


def test_known_nonsemantic_strategy_runs():
    out = chunk_text_service(
        text="First sentence. Second sentence.", language="en", strategy="simple"
    )
    assert out["chunk_count"] >= 1


def test_capabilities_lists_chunk_strategies():
    cap = build_capabilities()
    assert cap["chunk_strategies"] == list(KNOWN_CHUNK_STRATEGIES)
    assert "sliding_window_punct_conj_cleaned" in cap["chunk_strategies"]
