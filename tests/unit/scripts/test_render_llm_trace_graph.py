"""Tests for interactive LLM trace graph rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_llm_trace_graph  # noqa: E402


def _sample_trace() -> dict:
    return {
        "phase1": {
            "extracted": [
                {
                    "phrase": "symptomatic anaemia",
                    "category": "abnormal",
                    "chunk_ids": [1],
                    "evidence_text": "severe symptomatic anaemia",
                    "actionable": True,
                },
                {
                    "phrase": "tongue biting",
                    "category": "abnormal",
                    "chunk_ids": [2],
                    "evidence_text": "tongue biting",
                    "actionable": True,
                },
            ],
            "groups": [
                {
                    "group_id": 1,
                    "source_chunk_ids": [1, 2],
                    "status": "completed",
                    "extracted_count": 2,
                    "extracted": [
                        {
                            "phrase": "symptomatic anaemia",
                            "chunk_ids": [1],
                            "evidence_text": "severe symptomatic anaemia",
                        },
                        {
                            "phrase": "tongue biting",
                            "chunk_ids": [2],
                            "evidence_text": "tongue biting",
                        },
                    ],
                }
            ],
            "initial_mode": "grouped_large",
            "final_mode": "grouped_large",
            "fallback_triggered": False,
            "failure_class": None,
        },
        "phase2a": {
            "candidate_sets": [
                {
                    "phrase": "symptomatic anaemia",
                    "category": "abnormal",
                    "grounded_context": {
                        "chunk_ids": [1],
                        "primary_chunk_text": "32-year-old man presented for severe symptomatic anaemia",
                        "neighbor_chunk_texts": [
                            "Within 4 months his haemoglobin dropped by 62 points.",
                        ],
                    },
                    "candidates": [
                        {
                            "id": "HP:0001903",
                            "term": "Anemia",
                            "score": 0.91,
                            "matched_text": "symptomatic anaemia",
                            "matched_component": "label",
                        },
                        {
                            "id": "HP:0020062",
                            "term": "Decreased hemoglobin concentration",
                            "score": 0.77,
                            "matched_text": "haemoglobin dropped",
                            "matched_component": "definition",
                        },
                    ],
                },
                {
                    "phrase": "tongue biting",
                    "category": "abnormal",
                    "grounded_context": {
                        "chunk_ids": [2],
                        "primary_chunk_text": "Witnessed tongue biting during the episode.",
                        "neighbor_chunk_texts": [],
                    },
                    "candidates": [],
                },
            ]
        },
        "phase2b_local": {
            "resolved": [
                {
                    "phrase": "symptomatic anaemia",
                    "match_type": "matched_text_exact",
                    "hpo_id": "HP:0001903",
                    "term_name": "Anemia",
                }
            ],
            "unresolved": [
                {
                    "phrase": "tongue biting",
                    "category": "abnormal",
                    "candidates": [],
                }
            ],
        },
        "phase2b_llm": {
            "resolved": [
                {
                    "phrase": "tongue biting",
                    "hpo_id": "HP:0012169",
                    "term_name": "Self-biting",
                }
            ]
        },
        "final_annotations": [
            {
                "hpo_id": "HP:0001903",
                "term_name": "Anemia",
                "assertion_status": "present",
                "evidence": [
                    {
                        "phrase": "symptomatic anaemia",
                        "evidence_text": "severe symptomatic anaemia",
                        "chunk_ids": [1],
                    }
                ],
            },
            {
                "hpo_id": "HP:0012169",
                "term_name": "Self-biting",
                "assertion_status": "present",
                "evidence": [
                    {
                        "phrase": "tongue biting",
                        "evidence_text": "tongue biting",
                        "chunk_ids": [2],
                    }
                ],
            },
        ],
        "projected_predictions": ["HP:0001903", "HP:0012169"],
        "projection": {
            "assertion_projection": {"present": "present"},
        },
    }


def test_build_trace_graph_creates_expected_stage_nodes_and_edges() -> None:
    graph = render_llm_trace_graph.build_trace_graph(_sample_trace(), title="CSC_2")

    node_ids = {node["id"] for node in graph["nodes"]}
    assert "document" in node_ids
    assert "chunk:1" in node_ids
    assert "chunk:2" in node_ids
    assert "phrase:symptomatic anaemia" in node_ids
    assert "candidate:phrase:symptomatic anaemia:HP:0001903:0" in node_ids
    assert "local:phrase:symptomatic anaemia" in node_ids
    assert "llm:phrase:tongue biting" in node_ids
    assert "final:HP:0012169:1" in node_ids

    edges = {(edge["from"], edge["to"], edge["label"]) for edge in graph["edges"]}
    assert ("document", "chunk:1", "contains") in edges
    assert ("chunk:1", "phrase:symptomatic anaemia", "phase1") in edges
    assert (
        "phrase:symptomatic anaemia",
        "candidate:phrase:symptomatic anaemia:HP:0001903:0",
        "retrieved",
    ) in edges
    assert (
        "candidate:phrase:symptomatic anaemia:HP:0001903:0",
        "local:phrase:symptomatic anaemia",
        "accepted",
    ) in edges


def test_build_trace_graph_limits_candidates_and_skips_neighbors_by_default() -> None:
    graph = render_llm_trace_graph.build_trace_graph(
        _sample_trace(),
        title="CSC_2",
        max_candidates_per_phrase=1,
    )
    node_ids = {node["id"] for node in graph["nodes"]}
    assert "candidate:phrase:symptomatic anaemia:HP:0001903:0" in node_ids
    assert "candidate:phrase:symptomatic anaemia:HP:0020062:1" not in node_ids
    assert "candidate-summary:symptomatic anaemia" in node_ids
    assert not any(node["group"] == "neighbor_chunk" for node in graph["nodes"])


def test_build_trace_graph_unwraps_prediction_record_shape() -> None:
    wrapped = {"case_id": "CSC_2", "trace": _sample_trace()}
    graph = render_llm_trace_graph.build_trace_graph(wrapped, title="wrapped")
    node_ids = {node["id"] for node in graph["nodes"]}
    assert "phrase:tongue biting" in node_ids


def test_render_html_embeds_graph_payload_and_vis_network_loader() -> None:
    graph = render_llm_trace_graph.build_trace_graph(_sample_trace(), title="CSC_2")
    html = render_llm_trace_graph.render_trace_html(graph, title="CSC_2")

    assert "vis-network" in html
    assert "Trace Viewer: CSC_2" in html
    assert "GRAPH_PAYLOAD" in html
    assert "matched_text_exact" in html
    assert "Max candidates per phrase" in html
    assert "hideEdgesOnDrag" in html

    start = html.index("const GRAPH_PAYLOAD = ") + len("const GRAPH_PAYLOAD = ")
    end = html.index(";\n", start)
    payload = json.loads(html[start:end])
    assert payload["meta"]["title"] == "CSC_2"
    assert len(payload["nodes"]) == len(graph["nodes"])


def test_build_trace_graph_accepts_string_evidence_in_final_annotations() -> None:
    trace = _sample_trace()
    trace["final_annotations"] = [
        {
            "hpo_id": "HP:0001903",
            "term_name": "Anemia",
            "assertion_status": "present",
            "evidence": "symptomatic anaemia",
        }
    ]
    graph = render_llm_trace_graph.build_trace_graph(trace, title="CSC_2")
    edges = {(edge["from"], edge["to"], edge["label"]) for edge in graph["edges"]}
    assert (
        ("local:phrase:symptomatic anaemia", "final:HP:0001903:0", "final") in edges
        or ("llm:phrase:symptomatic anaemia", "final:HP:0001903:0", "final") in edges
        or ("phrase:symptomatic anaemia", "final:HP:0001903:0", "final") in edges
    )
