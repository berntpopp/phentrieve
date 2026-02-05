"""
Post-processing layer for LLM annotations.

This module provides optional post-processing steps to validate and refine
annotations produced by the primary annotation strategies:

- ValidationPostProcessor: Re-check annotations against original text
- RefinementPostProcessor: Upgrade to more specific HPO terms
- AssertionReviewPostProcessor: Validate negation detection
"""

from phentrieve.llm.postprocess.assertion_review import AssertionReviewPostProcessor
from phentrieve.llm.postprocess.base import PostProcessor
from phentrieve.llm.postprocess.refinement import RefinementPostProcessor
from phentrieve.llm.postprocess.validation import ValidationPostProcessor

__all__ = [
    "AssertionReviewPostProcessor",
    "PostProcessor",
    "RefinementPostProcessor",
    "ValidationPostProcessor",
]
