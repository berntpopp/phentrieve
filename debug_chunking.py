#!/usr/bin/env python
"""Debug script to diagnose the chunking pipeline configuration issue."""

import logging
import json

from sentence_transformers import SentenceTransformer
from phentrieve.config import get_detailed_chunking_config
from phentrieve.text_processing.pipeline import TextProcessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Text for testing
TEST_TEXT = "Patient has hearing loss and developmental delay."


def debug_detailed_config():
    """Debug the detailed configuration."""
    config = get_detailed_chunking_config()
    print(f"Type of config: {type(config)}")
    print(f"Config structure: {json.dumps(config, indent=2)}")

    # Check each item in the config
    for i, item in enumerate(config):
        print(f"Item {i}:")
        print(f"  Type: {type(item)}")
        print(f"  Content: {item}")
        if isinstance(item, dict) and "config" in item:
            print(f"  Config type: {type(item['config'])}")
            print(f"  Config content: {item['config']}")


def debug_pipeline():
    """Debug the pipeline initialization and processing."""
    try:
        # Load the model
        logger.info("Loading model...")
        model = SentenceTransformer("FremyCompany/BioLORD-2023-M")
        logger.info("Model loaded successfully")

        # Get the configuration
        config = get_detailed_chunking_config()
        logger.info(f"Using config: {config}")

        # Initialize the pipeline
        logger.info("Initializing pipeline...")
        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=config,
            assertion_config={"enable_keyword": True, "enable_dependency": True},
            sbert_model_for_semantic_chunking=model,
        )
        logger.info("Pipeline initialized successfully")

        # Process the test text
        logger.info(f"Processing test text: {TEST_TEXT}")
        try:
            results = pipeline.process(TEST_TEXT)
            logger.info(f"Processing succeeded with {len(results)} chunks")
            for i, chunk in enumerate(results):
                logger.info(f"Chunk {i+1}: {chunk['text']}")
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("=== DEBUG DETAILED CONFIG ===")
    debug_detailed_config()

    logger.info("\n=== DEBUG PIPELINE ===")
    debug_pipeline()
