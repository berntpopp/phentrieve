#!/usr/bin/env python3
"""Render a single LLM pipeline trace as an interactive HTML graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

VIS_NETWORK_CDN = "https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"
VIS_NETWORK_CSS_CDN = "https://unpkg.com/vis-network@9.1.9/styles/vis-network.min.css"

GROUP_STYLE = {
    "document": {"color": "#374151", "shape": "box"},
    "phase1_group": {"color": "#4f46e5", "shape": "box"},
    "chunk": {"color": "#0891b2", "shape": "box"},
    "neighbor_chunk": {"color": "#67e8f9", "shape": "box"},
    "phrase": {"color": "#0f766e", "shape": "ellipse"},
    "candidate": {"color": "#7c3aed", "shape": "dot"},
    "empty_candidate_set": {"color": "#dc2626", "shape": "diamond"},
    "local_resolution": {"color": "#16a34a", "shape": "box"},
    "llm_resolution": {"color": "#f59e0b", "shape": "box"},
    "final_annotation": {"color": "#1d4ed8", "shape": "box"},
    "projection": {"color": "#6b7280", "shape": "box"},
}


def _unwrap_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "trace" in payload and isinstance(payload["trace"], dict):
        return payload["trace"]
    return payload


def _truncate(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _json_details(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _safe_chunk_id(chunk_id: Any) -> str:
    return (
        str(int(chunk_id)) if isinstance(chunk_id, (int, float, str)) else str(chunk_id)
    )


def build_trace_graph(payload: dict[str, Any], *, title: str) -> dict[str, Any]:
    trace = _unwrap_trace_payload(payload)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    node_counts: Counter[str] = Counter()

    def add_node(
        node_id: str,
        *,
        label: str,
        group: str,
        level: int,
        details: dict[str, Any],
    ) -> None:
        if node_id in seen_nodes:
            return
        style = GROUP_STYLE[group]
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "group": group,
                "level": level,
                "shape": style["shape"],
                "color": style["color"],
                "details": details,
            }
        )
        seen_nodes.add(node_id)
        node_counts[group] += 1

    def add_edge(
        source: str,
        target: str,
        *,
        label: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        edge_key = (source, target, label)
        if edge_key in seen_edges:
            return
        edges.append(
            {
                "from": source,
                "to": target,
                "label": label,
                "arrows": "to",
                "details": details or {},
            }
        )
        seen_edges.add(edge_key)

    add_node(
        "document",
        label=title,
        group="document",
        level=0,
        details={"title": title},
    )

    phase1 = trace.get("phase1", {}) or {}
    phase2a = trace.get("phase2a", {}) or {}
    phase2b_local = trace.get("phase2b_local", {}) or {}
    phase2b_llm = trace.get("phase2b_llm", {}) or {}

    groups = phase1.get("groups", []) if isinstance(phase1, dict) else []
    phase1_extracted = phase1.get("extracted", []) if isinstance(phase1, dict) else []
    candidate_sets = (
        phase2a.get("candidate_sets", []) if isinstance(phase2a, dict) else []
    )

    for group in groups:
        group_id = f"group:{group.get('group_id', len(nodes))}"
        add_node(
            group_id,
            label=f"Phase 1 group {group.get('group_id', '?')}",
            group="phase1_group",
            level=1,
            details=group,
        )
        add_edge("document", group_id, label="phase1 group", details=group)
        for chunk_id in group.get("source_chunk_ids", []) or group.get("chunk_ids", []):
            chunk_node_id = f"chunk:{_safe_chunk_id(chunk_id)}"
            add_node(
                chunk_node_id,
                label=f"Chunk {_safe_chunk_id(chunk_id)}",
                group="chunk",
                level=2,
                details={"chunk_id": chunk_id},
            )
            add_edge(group_id, chunk_node_id, label="uses chunk")

    chunk_details: dict[str, dict[str, Any]] = {}
    phrase_to_context: dict[str, dict[str, Any]] = {}

    for item in candidate_sets:
        phrase = str(item.get("phrase", "")).strip()
        if not phrase:
            continue
        phrase_node_id = f"phrase:{phrase}"
        context = dict(item.get("grounded_context", {}) or {})
        phrase_to_context[phrase] = context
        chunk_ids = context.get("chunk_ids", []) or []
        primary_text = str(context.get("primary_chunk_text", "") or "")
        neighbor_texts = [
            str(text)
            for text in context.get("neighbor_chunk_texts", [])
            if str(text).strip()
        ]

        for index, chunk_id in enumerate(chunk_ids):
            node_id = f"chunk:{_safe_chunk_id(chunk_id)}"
            details = chunk_details.setdefault(
                node_id,
                {
                    "chunk_id": chunk_id,
                    "primary_chunk_text": primary_text if index == 0 else "",
                    "neighbor_chunk_texts": neighbor_texts if index == 0 else [],
                },
            )
            if index == 0 and primary_text:
                details["primary_chunk_text"] = primary_text
            add_node(
                node_id,
                label=f"Chunk {_safe_chunk_id(chunk_id)}\n{_truncate(primary_text or phrase, 48)}",
                group="chunk",
                level=2,
                details=details,
            )
            add_edge("document", node_id, label="contains", details=details)
            add_edge(node_id, phrase_node_id, label="phase1")

        for neighbor_index, text in enumerate(neighbor_texts, start=1):
            neighbor_id = f"neighbor:{phrase}:{neighbor_index}"
            add_node(
                neighbor_id,
                label=f"Neighbor\n{_truncate(text, 48)}",
                group="neighbor_chunk",
                level=2,
                details={"phrase": phrase, "text": text},
            )
            add_edge(neighbor_id, phrase_node_id, label="context")

        matching_extracted = next(
            (entry for entry in phase1_extracted if entry.get("phrase") == phrase),
            {
                "phrase": phrase,
                "category": item.get("category"),
                "chunk_ids": chunk_ids,
                "evidence_text": primary_text,
            },
        )
        add_node(
            phrase_node_id,
            label=f"{phrase}\n[{matching_extracted.get('category', item.get('category', 'unknown'))}]",
            group="phrase",
            level=3,
            details=matching_extracted,
        )

        candidates = item.get("candidates", []) or []
        if not candidates:
            empty_id = f"empty:{phrase}"
            add_node(
                empty_id,
                label="No candidates",
                group="empty_candidate_set",
                level=4,
                details={"phrase": phrase, "grounded_context": context},
            )
            add_edge(phrase_node_id, empty_id, label="retrieval miss")
            continue

        for candidate_index, candidate in enumerate(candidates):
            candidate_id = str(candidate.get("id", candidate.get("hpo_id", "")) or "")
            candidate_label = str(
                candidate.get(
                    "term", candidate.get("label", candidate.get("term_name", ""))
                )
            )
            node_id = f"candidate:phrase:{phrase}:{candidate_id}:{candidate_index}"
            add_node(
                node_id,
                label=f"{candidate_label}\n{candidate_id}",
                group="candidate",
                level=4,
                details=candidate,
            )
            add_edge(
                phrase_node_id,
                node_id,
                label="retrieved",
                details={"score": candidate.get("score")},
            )

    for item in (
        phase2b_local.get("resolved", []) if isinstance(phase2b_local, dict) else []
    ):
        phrase = str(item.get("phrase", "")).strip()
        if not phrase:
            continue
        node_id = f"local:phrase:{phrase}"
        add_node(
            node_id,
            label=f"Local match\n{item.get('term_name', item.get('hpo_id', ''))}",
            group="local_resolution",
            level=5,
            details=item,
        )
        candidate_match = next(
            (
                node["id"]
                for node in nodes
                if node["group"] == "candidate"
                and node["details"].get("id", node["details"].get("hpo_id"))
                == item.get("hpo_id")
                and node["id"].startswith(f"candidate:phrase:{phrase}:")
            ),
            None,
        )
        source_id = candidate_match or f"phrase:{phrase}"
        add_edge(source_id, node_id, label="accepted", details=item)

    for item in (
        phase2b_local.get("unresolved", []) if isinstance(phase2b_local, dict) else []
    ):
        phrase = str(item.get("phrase", "")).strip()
        if not phrase:
            continue
        node_id = f"local-unresolved:{phrase}"
        add_node(
            node_id,
            label="Local unresolved",
            group="local_resolution",
            level=5,
            details=item,
        )
        add_edge(f"phrase:{phrase}", node_id, label="unresolved", details=item)

    for item in (
        phase2b_llm.get("resolved", []) if isinstance(phase2b_llm, dict) else []
    ):
        phrase = str(item.get("phrase", "")).strip()
        if not phrase:
            continue
        node_id = f"llm:phrase:{phrase}"
        add_node(
            node_id,
            label=f"LLM match\n{item.get('term_name', item.get('hpo_id', ''))}",
            group="llm_resolution",
            level=6,
            details=item,
        )
        add_edge(f"phrase:{phrase}", node_id, label="llm resolved", details=item)

    for index, annotation in enumerate(trace.get("final_annotations", []) or []):
        node_id = f"final:{annotation.get('hpo_id', 'unknown')}:{index}"
        label = f"{annotation.get('term_name', annotation.get('hpo_id', 'unknown'))}\n{annotation.get('hpo_id', '')}"
        add_node(
            node_id,
            label=label,
            group="final_annotation",
            level=7,
            details=annotation,
        )
        raw_evidence = annotation.get("evidence", []) or []
        if isinstance(raw_evidence, list):
            evidence_items = [item for item in raw_evidence if isinstance(item, dict)]
        elif isinstance(raw_evidence, dict):
            evidence_items = [raw_evidence]
        elif isinstance(raw_evidence, str) and raw_evidence.strip():
            evidence_items = [
                {
                    "phrase": raw_evidence.strip(),
                    "evidence_text": raw_evidence,
                }
            ]
        else:
            evidence_items = []
        linked = False
        for evidence in evidence_items:
            phrase = str(evidence.get("phrase", "")).strip()
            if phrase:
                llm_node = f"llm:phrase:{phrase}"
                local_node = f"local:phrase:{phrase}"
                if llm_node in seen_nodes:
                    add_edge(llm_node, node_id, label="final", details=evidence)
                    linked = True
                elif local_node in seen_nodes:
                    add_edge(local_node, node_id, label="final", details=evidence)
                    linked = True
                elif f"phrase:{phrase}" in seen_nodes:
                    add_edge(
                        f"phrase:{phrase}", node_id, label="final", details=evidence
                    )
                    linked = True
        if not linked:
            add_edge("document", node_id, label="final")

    projected = trace.get("projected_predictions", []) or []
    if projected:
        projection_id = "projection"
        add_node(
            projection_id,
            label="Projected predictions",
            group="projection",
            level=8,
            details={
                "projection": trace.get("projection", {}),
                "projected_predictions": projected,
            },
        )
        for node in nodes:
            if node["group"] == "final_annotation":
                add_edge(node["id"], projection_id, label="projected")

    graph = {
        "meta": {
            "title": title,
            "node_counts": dict(node_counts),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def render_trace_html(graph: dict[str, Any], *, title: str) -> str:
    payload_json = json.dumps(graph, ensure_ascii=False)
    group_names = sorted({node["group"] for node in graph["nodes"]})
    filter_controls = "\n".join(
        (
            f'<label><input type="checkbox" class="group-filter" '
            f'data-group="{escape(group)}" checked> {escape(group)}</label>'
        )
        for group in group_names
    )
    summary_items = "".join(
        f"<li><strong>{escape(group)}:</strong> {count}</li>"
        for group, count in sorted(graph["meta"]["node_counts"].items())
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Trace Viewer: {escape(title)}</title>
  <link rel="stylesheet" href="{VIS_NETWORK_CSS_CDN}">
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px 1fr 360px;
      height: 100vh;
    }}
    .panel {{
      overflow: auto;
      padding: 16px;
      border-right: 1px solid #e5e7eb;
      background: #ffffff;
    }}
    .panel:last-child {{
      border-right: none;
      border-left: 1px solid #e5e7eb;
    }}
    #graph {{
      width: 100%;
      height: 100vh;
      background: #f8fafc;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    h2 {{
      font-size: 14px;
      margin: 18px 0 8px;
      color: #374151;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .controls label {{
      display: block;
      margin: 6px 0;
      font-size: 14px;
    }}
    .search {{
      width: 100%;
      box-sizing: border-box;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 14px;
    }}
    .button-row {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #cbd5e1;
      background: white;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 10px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .muted {{
      color: #6b7280;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <h1>Trace Viewer: {escape(title)}</h1>
      <div class="muted">Interactive process graph for a single LLM trace.</div>
      <h2>Summary</h2>
      <ul>{summary_items}</ul>
      <div class="muted" style="margin-top:8px;">Edges: {graph["meta"]["edge_count"]}</div>

      <h2>Filters</h2>
      <div class="controls">{filter_controls}</div>

      <h2>Search</h2>
      <input id="search" class="search" type="search" placeholder="Find phrase, chunk, or term">
      <div class="button-row">
        <button id="fit-btn" type="button">Fit graph</button>
        <button id="reset-btn" type="button">Reset filters</button>
      </div>
    </aside>
    <main id="graph"></main>
    <aside class="panel">
      <h1>Details</h1>
      <div class="muted">Click any node or edge to inspect the underlying trace payload.</div>
      <pre id="details">Select a node or edge.</pre>
    </aside>
  </div>

  <script src="{VIS_NETWORK_CDN}"></script>
  <script>
    const GRAPH_PAYLOAD = {payload_json};
    const allNodes = GRAPH_PAYLOAD.nodes.map((node) => ({{ ...node }}));
    const allEdges = GRAPH_PAYLOAD.edges.map((edge, index) => ({{
      id: `edge:${{index}}`,
      ...edge,
      font: {{ align: 'top' }},
      color: {{ color: '#94a3b8' }}
    }}));

    const nodes = new vis.DataSet(allNodes);
    const edges = new vis.DataSet(allEdges);
    const network = new vis.Network(
      document.getElementById('graph'),
      {{ nodes, edges }},
      {{
        layout: {{
          hierarchical: {{
            enabled: true,
            direction: 'LR',
            sortMethod: 'directed',
            nodeSpacing: 180,
            levelSeparation: 220
          }}
        }},
        physics: false,
        interaction: {{
          hover: true,
          navigationButtons: true,
          keyboard: true
        }},
        edges: {{
          smooth: true,
          arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
          font: {{ size: 11, color: '#475569' }}
        }},
        nodes: {{
          shape: 'box',
          margin: 10,
          font: {{ face: 'Inter, system-ui, sans-serif', size: 12 }},
          borderWidth: 1
        }}
      }}
    );

    const details = document.getElementById('details');
    const search = document.getElementById('search');
    const filters = Array.from(document.querySelectorAll('.group-filter'));

    function activeGroups() {{
      return new Set(filters.filter((input) => input.checked).map((input) => input.dataset.group));
    }}

    function applyFilters() {{
      const enabled = activeGroups();
      const query = search.value.trim().toLowerCase();
      const visibleNodes = new Set();

      for (const node of allNodes) {{
        const text = JSON.stringify(node).toLowerCase();
        const visible = enabled.has(node.group) && (!query || text.includes(query));
        nodes.update({{ id: node.id, hidden: !visible }});
        if (visible) visibleNodes.add(node.id);
      }}

      for (const edge of allEdges) {{
        const visible = visibleNodes.has(edge.from) && visibleNodes.has(edge.to);
        edges.update({{ id: edge.id, hidden: !visible }});
      }}
    }}

    function showDetails(payload) {{
      details.textContent = JSON.stringify(payload, null, 2);
    }}

    filters.forEach((input) => input.addEventListener('change', applyFilters));
    search.addEventListener('input', applyFilters);
    document.getElementById('fit-btn').addEventListener('click', () => network.fit({{ animation: true }}));
    document.getElementById('reset-btn').addEventListener('click', () => {{
      filters.forEach((input) => input.checked = true);
      search.value = '';
      applyFilters();
      network.fit({{ animation: true }});
    }});

    network.on('click', (params) => {{
      if (params.nodes.length) {{
        const node = nodes.get(params.nodes[0]);
        showDetails(node.details || node);
        return;
      }}
      if (params.edges.length) {{
        const edge = edges.get(params.edges[0]);
        showDetails(edge.details || edge);
      }}
    }});

    applyFilters();
    network.fit();
  </script>
</body>
</html>"""


def _derive_default_output_path(trace_path: Path) -> Path:
    return trace_path.with_suffix(".html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single LLM trace as an interactive HTML graph."
    )
    parser.add_argument(
        "trace_json", type=Path, help="Path to a single trace JSON file"
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Path to write the interactive HTML output (default: <trace>.html)",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Optional title override for the HTML viewer",
    )
    args = parser.parse_args()

    payload = json.loads(args.trace_json.read_text(encoding="utf-8"))
    title = args.title or args.trace_json.stem
    graph = build_trace_graph(payload, title=title)
    html = render_trace_html(graph, title=title)
    output_path = args.output_html or _derive_default_output_path(args.trace_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
