#!/usr/bin/env python3
"""Convert released RAG-HPO paper benchmark files into Phentrieve JSON fixtures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

from raghpo_paper_converter import RagHpoPaperConverter
from shared_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_DIR = Path("tests/data/en/raghpo_paper/source")
DEFAULT_WORKBOOK_NAME = "RAG-HPO Tests and Data Analysis copy.xlsx"
DEFAULT_TEST_CASES_NAME = "Test_Cases.csv"
WORKBOOK_URL = (
    "https://raw.githubusercontent.com/PoseyPod/RAG-HPO/main/"
    "RAG-HPO%20Tests%20and%20Data%20Analysis%20copy.xlsx"
)
TEST_CASES_URL = (
    "https://raw.githubusercontent.com/PoseyPod/RAG-HPO/main/Test_Cases.csv"
)


def _validate_paths(
    workbook_path: Path,
    test_cases_csv_path: Path,
    hpo_terms_path: Path | None,
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


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)
    urlretrieve(url, destination)  # noqa: S310 - fixed HTTPS upstream release URLs
    return destination


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.download_source:
        return args.workbook, args.test_cases_csv

    download_dir = args.download_dir
    workbook_path = args.workbook or (download_dir / DEFAULT_WORKBOOK_NAME)
    test_cases_csv_path = args.test_cases_csv or (
        download_dir / DEFAULT_TEST_CASES_NAME
    )

    if not workbook_path.exists():
        _download_file(WORKBOOK_URL, workbook_path)
    else:
        logger.info("Using cached workbook: %s", workbook_path)

    if not test_cases_csv_path.exists():
        _download_file(TEST_CASES_URL, test_cases_csv_path)
    else:
        logger.info("Using cached test cases CSV: %s", test_cases_csv_path)

    return workbook_path, test_cases_csv_path


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
        help="Path to 'RAG-HPO Tests and Data Analysis copy.xlsx'",
    )
    parser.add_argument(
        "--test-cases-csv",
        type=Path,
        help="Path to Test_Cases.csv",
    )
    parser.add_argument(
        "--download-source",
        action="store_true",
        help=(
            "Download the released RAG-HPO workbook and Test_Cases.csv into a local "
            "cache when the paths are not provided or do not exist."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=(
            "Local cache directory for downloaded RAG-HPO source files "
            "(default: tests/data/en/raghpo_paper/source)"
        ),
    )
    parser.add_argument(
        "--hpo-terms",
        type=Path,
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
    if not args.download_source and (
        args.workbook is None or args.test_cases_csv is None
    ):
        logger.error(
            "--workbook and --test-cases-csv are required unless --download-source is enabled"
        )
        sys.exit(1)

    if args.hpo_terms is None and args.hpo_json is None:
        logger.error("either --hpo-terms or --hpo-json is required")
        sys.exit(1)

    if args.normalize_obsolete_ids and args.hpo_json is None:
        logger.error("--hpo-json is required when --normalize-obsolete-ids is enabled")
        sys.exit(1)
    workbook_path, test_cases_csv_path = _resolve_input_paths(args)
    if not _validate_paths(
        workbook_path,
        test_cases_csv_path,
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
        workbook_path=workbook_path,
        test_cases_csv_path=test_cases_csv_path,
        output_root=args.output,
        dataset=args.dataset,
    )
    summary = report["summary"]
    total_documents = int(summary["total_documents"])
    warnings_count = int(summary["warnings"])
    logger.info(
        "Converted %s documents with %s warnings",
        total_documents,
        warnings_count,
    )


if __name__ == "__main__":
    main()
