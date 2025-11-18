#!/usr/bin/env python3
"""
Generate chunking variants for annotated documents.

This script processes JSON annotation files and augments them with
ground-truth chunking variants using the Voronoi boundary algorithm.

Usage:
    # Single file
    python scripts/generate_chunking_variants.py \\
        --input tests/data/en/phenobert/GSC_plus/annotations/GSC_plus_1003450.json

    # Directory (recursive)
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --pattern "*/annotations/*.json"

    # Specify expansion ratios
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --expansion-ratios 0.0 0.5 1.0

    # Dry run (preview without writing)
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --dry-run

    # Force overwrite existing chunks
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --force
"""

import argparse
import json
import logging
from pathlib import Path

from annotation_chunker import (
    CHUNKER_VERSION,
    DEFAULT_EXPANSION_RATIOS,
    generate_chunk_variants,
)
from shared_utils import setup_logging

logger = logging.getLogger(__name__)


def process_file(
    file_path: Path,
    expansion_ratios: list[float],
    strategy_name: str,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Process single file and add chunk variants.

    Args:
        file_path: Path to JSON file
        expansion_ratios: Expansion ratios to use
        strategy_name: Strategy name for chunk_variants key
        dry_run: If True, don't write changes
        force: If True, overwrite existing chunks

    Returns:
        True if processed successfully, False otherwise
    """
    try:
        # Read file
        with open(file_path) as f:
            doc = json.load(f)

        # Check if already processed
        if not force and "chunk_variants" in doc:
            if strategy_name in doc["chunk_variants"]:
                logger.debug(f"Skipping {file_path.name} (already chunked)")
                return True

        # Generate chunks
        chunks = generate_chunk_variants(doc, expansion_ratios)

        if dry_run:
            logger.info(f"[DRY RUN] Would process: {file_path.name}")
            return True

        # Augment document (in-place)
        if "chunk_variants" not in doc:
            doc["chunk_variants"] = {}

        doc["chunk_variants"][strategy_name] = chunks

        # Write back atomically
        with open(file_path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        logger.info(f"Processed: {file_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate chunking variants for annotated documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Single input file",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory (recursive)",
    )

    parser.add_argument(
        "--pattern",
        default="*/annotations/*.json",
        help="File pattern for --input-dir (default: */annotations/*.json)",
    )

    parser.add_argument(
        "--expansion-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_EXPANSION_RATIOS,
        help=f"Expansion ratios (default: {DEFAULT_EXPANSION_RATIOS})",
    )

    parser.add_argument(
        "--strategy-name",
        default="voronoi_v1",
        help="Strategy name for chunk_variants (default: voronoi_v1)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be done",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunks",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info(f"Script version: {CHUNKER_VERSION}")
    logger.info(f"Expansion ratios: {args.expansion_ratios}")

    # Collect files
    if args.input:
        if not args.input.exists():
            logger.error(f"File not found: {args.input}")
            return
        files = [args.input]
    else:
        if not args.input_dir.exists():
            logger.error(f"Directory not found: {args.input_dir}")
            return
        files = sorted(args.input_dir.glob(args.pattern))

    if not files:
        logger.error("No files found!")
        return

    logger.info(f"Processing {len(files)} files...")

    # Process files
    success_count = 0
    for file_path in files:
        if process_file(
            file_path,
            args.expansion_ratios,
            args.strategy_name,
            args.dry_run,
            args.force,
        ):
            success_count += 1

    # Summary
    logger.info(f"✓ Successfully processed {success_count}/{len(files)} files")

    if args.dry_run:
        logger.info("[DRY RUN] No changes written")


if __name__ == "__main__":
    main()
