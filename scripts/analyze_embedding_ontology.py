#!/usr/bin/env python3
"""Ontology-embedding fidelity analysis.

Loads HPO graph data + cached BioLORD embeddings, computes four correlation
metrics and produces five plots in a timestamped output directory.

Run `python scripts/analyze_embedding_ontology.py --help` for usage.

This is a standalone research script - intentionally not exposed via the
`phentrieve` CLI. See .planning/specs/2026-04-17-ontology-embedding-fidelity-design.md
for the full contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("analyze_embedding_ontology")


@dataclass
class Args:
    model_name: str
    output_dir: Path
    k: int
    n_pairs: int
    umap_neighbors: int
    umap_min_dist: float
    metric: str
    sample: int | None
    seed: int
    skip_interactive: bool
    refresh_cache: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(
        description="Ontology-embedding fidelity analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default="FremyCompany/BioLORD-2023-M")
    p.add_argument(
        "--output-dir",
        default="data/results/ontology_fidelity",
        type=Path,
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-interactive", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args(argv)
    return Args(
        model_name=ns.model_name,
        output_dir=ns.output_dir,
        k=ns.k,
        n_pairs=ns.n_pairs,
        umap_neighbors=ns.umap_neighbors,
        umap_min_dist=ns.umap_min_dist,
        metric=ns.metric,
        sample=ns.sample,
        seed=ns.seed,
        skip_interactive=ns.skip_interactive,
        refresh_cache=ns.refresh_cache,
        log_level=ns.log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_hpo_bundle(
    data_dir_override: str | None = None,
) -> tuple[list[dict], dict[str, set[str]], dict[str, int], str | None, Path]:
    """Open HPODatabase directly, load terms + graph + version, close. Hard-fail
    on missing DB (unlike document_creator.load_hpo_terms which soft-fails)."""
    from phentrieve.config import DEFAULT_HPO_DB_FILENAME
    from phentrieve.data_processing.hpo_database import HPODatabase
    from phentrieve.utils import get_default_data_dir, resolve_data_path

    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME
    if not db_path.exists():
        raise FileNotFoundError(
            f"HPO database not found: {db_path}. Run 'phentrieve data prepare' first."
        )
    db = HPODatabase(db_path)
    try:
        terms = db.load_all_terms()
        ancestors, depths = db.load_graph_data()
        hpo_version = db.get_metadata("hpo_version")
    finally:
        db.close()
    logger.info(
        "HPO DB loaded: %d terms, version=%s, path=%s",
        len(terms),
        hpo_version,
        db_path,
    )
    return terms, ancestors, depths, hpo_version, db_path


def align_terms_and_embeddings(
    terms: list[dict],
    hpo_ids: list[str],
    embeddings: np.ndarray,
    symmetric_diff_tolerance: float = 0.05,
) -> tuple[list[dict], np.ndarray]:
    """Filter both sides to the intersection of (HPO DB IDs) and (cache IDs).

    - If |symmetric difference| / min(|A|, |B|) > tolerance, raise ValueError —
      the cache is likely stale.
    - Otherwise, log a warning with counts and proceed on the intersection.
    - Preserves cache row order for returned embeddings.
    """
    db_ids = {t["id"] for t in terms}
    cache_ids = set(hpo_ids)
    inter = db_ids & cache_ids
    only_db = db_ids - cache_ids
    only_cache = cache_ids - db_ids

    sym_diff = len(only_db) + len(only_cache)
    denom = max(min(len(db_ids), len(cache_ids)), 1)
    if sym_diff / denom > symmetric_diff_tolerance:
        raise ValueError(
            f"HPO DB / embedding cache divergence too large: "
            f"{sym_diff} mismatched IDs out of min({len(db_ids)}, {len(cache_ids)}) "
            f"({sym_diff / denom:.1%} > {symmetric_diff_tolerance:.0%}). "
            "Rebuild the index or refresh the cache (--refresh-cache)."
        )
    if sym_diff:
        logger.warning(
            "HPO DB / embedding cache mismatch: %d only in DB, %d only in cache",
            len(only_db),
            len(only_cache),
        )
    if not inter:
        raise ValueError("Zero overlap between HPO DB and embedding cache.")

    keep_cache_rows = [i for i, hid in enumerate(hpo_ids) if hid in inter]
    aligned_embeddings = embeddings[keep_cache_rows]
    aligned_ids_set = {hpo_ids[i] for i in keep_cache_rows}
    aligned_terms = [t for t in terms if t["id"] in aligned_ids_set]

    # Sort aligned_terms into the same order as aligned_embeddings rows.
    ordered_ids = [hpo_ids[i] for i in keep_cache_rows]
    by_id = {t["id"]: t for t in aligned_terms}
    aligned_terms = [by_id[i] for i in ordered_ids]

    return aligned_terms, aligned_embeddings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        terms, ancestors, depths, hpo_version, db_path = load_hpo_bundle()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    try:
        hpo_ids, embeddings = load_cached_embeddings(
            model_name=args.model_name,
            refresh=args.refresh_cache,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    try:
        aligned_terms, aligned_embeddings = align_terms_and_embeddings(
            terms, hpo_ids, embeddings
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if args.sample is not None and args.sample < len(aligned_terms):
        import numpy as np

        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(len(aligned_terms), size=args.sample, replace=False)
        sample_idx.sort()
        aligned_terms = [aligned_terms[i] for i in sample_idx]
        aligned_embeddings = aligned_embeddings[sample_idx]
        logger.info("Sampled %d terms (seed=%d)", len(aligned_terms), args.seed)
    elif args.sample is not None:
        logger.info(
            "Requested --sample %d >= %d aligned terms; using all.",
            args.sample,
            len(aligned_terms),
        )

    logger.info(
        "Aligned: %d terms × %d dims", len(aligned_terms), aligned_embeddings.shape[1]
    )

    # Metrics, UMAP, and writers are added in subsequent tasks.
    return 0


if __name__ == "__main__":
    sys.exit(main())
