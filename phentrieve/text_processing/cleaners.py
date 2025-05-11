"""
Text cleaning and normalization utilities for Phentrieve.

This module provides functions for pre-processing text before chunking
and assertion detection, ensuring consistent text formatting.
"""

import re
from typing import Optional


def normalize_line_endings(text: str) -> str:
    """
    Normalize different line ending styles to Unix-style newlines.

    Args:
        text: Input text with potentially mixed line endings

    Returns:
        Text with normalized line endings (all '\n')
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_internal_newlines_and_extra_spaces(text_chunk: str) -> str:
    """
    Replace internal newlines with spaces and normalize whitespace.

    This function:
    1. Replaces any newlines surrounded by optional whitespace with a single space
    2. Replaces sequences of 2+ spaces with a single space
    3. Strips leading/trailing whitespace

    Args:
        text_chunk: Text chunk to clean

    Returns:
        Cleaned text with normalized spacing
    """
    if not text_chunk:
        return ""
    cleaned = re.sub(r"\s*\n\s*", " ", text_chunk)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
