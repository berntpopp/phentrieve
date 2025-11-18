#!/usr/bin/env python3
"""
Convert PhenoBERT corpus datasets to Phentrieve JSON format.

Converts three PhenoBERT datasets:
- GSC+ (BiolarkGSC+): 228 PubMed abstracts
- ID-68: 68 clinical notes from intellectual disability families
- GeneReviews: 10 clinical cases from GeneReviews database

Usage:
    # Convert all datasets
    python scripts/convert_phenobert_data.py \\
        --phenobert-data path/to/PhenoBERT/phenobert/data \\
        --output data/test_texts/phenobert \\
        --hpo-data data/hpo_core_data

    # Convert specific dataset
    python scripts/convert_phenobert_data.py \\
        --phenobert-data path/to/PhenoBERT/phenobert/data \\
        --output data/test_texts/phenobert \\
        --hpo-data data/hpo_core_data \\
        --dataset GSC+

    # Help
    python scripts/convert_phenobert_data.py --help

Output:
    Creates directory structure:
    data/test_texts/phenobert/
    ├── GSC_plus/
    │   └── annotations/
    │       ├── GSC+_doc001.json
    │       └── ...
    ├── ID68/
    │   └── annotations/
    │       └── ...
    ├── GeneReviews/
    │   └── annotations/
    │       └── ...
    └── conversion_report.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Import from local module (scripts/ directory)
from phenobert_converter import (
    AnnotationParser,
    DatasetLoader,
    HPOLookup,
    OutputWriter,
    PhenoBERTConverter,
)

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging format and level.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_paths(args: argparse.Namespace) -> bool:
    """
    Validate input paths exist.

    Args:
        args: Parsed command-line arguments

    Returns:
        True if all paths valid, False otherwise
    """
    errors = []

    if not args.phenobert_data.exists():
        errors.append(f"PhenoBERT data directory not found: {args.phenobert_data}")

    if not args.hpo_data.exists():
        errors.append(f"HPO data directory not found: {args.hpo_data}")

    hpo_terms_file = args.hpo_data / "hpo_terms.tsv"
    if not hpo_terms_file.exists():
        errors.append(f"HPO terms file not found: {hpo_terms_file}")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


def main() -> None:
    """Main conversion entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PhenoBERT corpus datasets to Phentrieve JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--phenobert-data",
        type=Path,
        required=True,
        help="Path to PhenoBERT data directory (phenobert/data/)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for converted JSON files",
    )

    parser.add_argument(
        "--hpo-data",
        type=Path,
        required=True,
        help="Path to HPO data directory (data/hpo_core_data/)",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset",
        choices=["GSC+", "ID-68", "GeneReviews", "all"],
        default="all",
        help="Dataset to convert (default: all)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("=" * 70)
    logger.info("PhenoBERT Corpus Conversion")
    logger.info("=" * 70)

    # Validate paths
    if not validate_paths(args):
        logger.error("Path validation failed. Exiting.")
        sys.exit(1)

    # Initialize components (dependency injection pattern)
    try:
        logger.info("Initializing converter components...")

        dataset_loader = DatasetLoader(args.phenobert_data)
        annotation_parser = AnnotationParser()
        hpo_lookup = HPOLookup(args.hpo_data)
        converter = PhenoBERTConverter(hpo_lookup, dataset_loader, annotation_parser)
        writer = OutputWriter(args.output, args.phenobert_data)

        logger.info("✓ Components initialized")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # Determine datasets to convert
    if args.dataset == "all":
        datasets = dataset_loader.discover_datasets()
        if not datasets:
            logger.error("No datasets found in PhenoBERT data directory")
            sys.exit(1)
        logger.info(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    else:
        datasets = [args.dataset]
        logger.info(f"Converting dataset: {args.dataset}")

    # Convert each dataset
    total_processed = 0
    total_errors = 0

    for dataset_name in datasets:
        logger.info("-" * 70)
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info("-" * 70)

        dataset_processed = 0
        dataset_errors = 0

        try:
            for corpus_file, ann_file in dataset_loader.load_file_pairs(dataset_name):
                try:
                    # Convert document
                    doc = converter.convert_document(
                        corpus_file, ann_file, dataset_name
                    )

                    # Write output
                    writer.write_document(doc, dataset_name)

                    dataset_processed += 1
                    total_processed += 1

                    if dataset_processed % 10 == 0:
                        logger.info(f"Processed {dataset_processed} documents...")

                except Exception as e:
                    logger.error(f"Error converting {corpus_file.name}: {e}")
                    dataset_errors += 1
                    total_errors += 1

            logger.info(
                f"✓ {dataset_name}: {dataset_processed} documents converted"
                + (f" ({dataset_errors} errors)" if dataset_errors > 0 else "")
            )

        except Exception as e:
            logger.error(f"Fatal error processing {dataset_name}: {e}", exc_info=True)
            total_errors += 1

    # Write conversion report
    try:
        writer.write_report()
    except Exception as e:
        logger.error(f"Failed to write report: {e}")

    # Final summary
    logger.info("=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total documents converted: {total_processed}")
    logger.info(f"Total annotations: {writer.stats.total_annotations}")

    for dataset, stats in writer.stats.datasets.items():
        logger.info(
            f"  {dataset}: {stats['docs']} documents, {stats['annotations']} annotations"
        )

    if total_errors > 0:
        logger.warning(f"Total errors: {total_errors}")
        logger.info(f"Output directory: {args.output}")
        sys.exit(1)
    else:
        logger.info("✓ All conversions completed successfully")
        logger.info(f"Output directory: {args.output}")
        sys.exit(0)


if __name__ == "__main__":
    main()
