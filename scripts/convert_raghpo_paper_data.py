#!/usr/bin/env python3
"""Convert released RAG-HPO paper benchmark files into Phentrieve JSON fixtures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from raghpo_paper_converter import RagHpoPaperConverter
from shared_utils import setup_logging

logger = logging.getLogger(__name__)


def _validate_paths(
    workbook_path: Path,
    test_cases_csv_path: Path,
    hpo_terms_path: Path,
    hpo_json_path: Path | None,
) -> bool:
    missing = [
        path
        for path in (workbook_path, test_cases_csv_path, hpo_terms_path, hpo_json_path)
        if path is not None and not path.exists()
    ]
    for path in missing:
        logger.error("Missing required input: %s", path)
    return not missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert released RAG-HPO paper benchmark files into "
            "Phentrieve-compatible JSON documents"
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Path to 'RAG-HPO Tests and Data Analysis copy.xlsx'",
    )
    parser.add_argument(
        "--test-cases-csv",
        type=Path,
        required=True,
        help="Path to Test_Cases.csv",
    )
    parser.add_argument(
        "--hpo-terms",
        type=Path,
        required=True,
        help="Path to data/hpo_core_data/hpo_terms.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root for converted JSON files",
    )
    parser.add_argument(
        "--dataset",
        choices=["CSC", "GSC", "all"],
        default="all",
        help="Dataset subset to convert (default: all)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--normalize-obsolete-ids",
        action="store_true",
        help=(
            "Replace obsolete HPO IDs with their current ontology replacements when "
            "available"
        ),
    )
    parser.add_argument(
        "--hpo-json",
        type=Path,
        help=(
            "Path to an HPO ontology JSON export such as data/hp.json. Required when "
            "--normalize-obsolete-ids is enabled."
        ),
    )
    parser.add_argument(
        "--drop-obsolete-without-replacement",
        action="store_true",
        help=(
            "When normalizing obsolete HPO IDs, drop any obsolete terms that still "
            "have no current replacement."
        ),
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    if args.normalize_obsolete_ids and args.hpo_json is None:
        logger.error("--hpo-json is required when --normalize-obsolete-ids is enabled")
        sys.exit(1)
    if not _validate_paths(
        args.workbook,
        args.test_cases_csv,
        args.hpo_terms,
        args.hpo_json,
    ):
        sys.exit(1)

    converter = RagHpoPaperConverter(
        hpo_terms_path=args.hpo_terms,
        hpo_json_path=args.hpo_json,
        normalize_obsolete_ids=args.normalize_obsolete_ids,
        drop_obsolete_without_replacement=args.drop_obsolete_without_replacement,
    )
    report = converter.convert(
        workbook_path=args.workbook,
        test_cases_csv_path=args.test_cases_csv,
        output_root=args.output,
        dataset=args.dataset,
    )
    logger.info(
        "Converted %s documents with %s warnings",
        report["summary"]["total_documents"],
        report["summary"]["warnings"],
    )


if __name__ == "__main__":
    main()
