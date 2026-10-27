"""Shared limits for CPU-heavy API work (input size and inference concurrency).

``INFERENCE_SLOTS`` caps how many model-inference jobs run in worker threads at
once. Every request otherwise gets its own threadpool thread (anyio's default
pool has 40), and each of those runs torch with many intra-op threads, so a few
concurrent requests slow every user down. Wrap ``run_in_threadpool`` calls that
run embedding/NLP work with ``async with INFERENCE_SLOTS:``.
"""

from __future__ import annotations

import logging
import os

import anyio

logger = logging.getLogger(__name__)

# Upper bound for free-text inputs (REST text/query bodies and MCP text args).
MAX_TEXT_INPUT_CHARS = 50_000

_DEFAULT_MAX_CONCURRENT_INFERENCE = 2


def _read_max_concurrent_inference() -> int:
    raw = os.getenv("PHENTRIEVE_MAX_CONCURRENT_INFERENCE")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_CONCURRENT_INFERENCE
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid PHENTRIEVE_MAX_CONCURRENT_INFERENCE=%r; using %d.",
            raw,
            _DEFAULT_MAX_CONCURRENT_INFERENCE,
        )
        return _DEFAULT_MAX_CONCURRENT_INFERENCE
    return max(value, 1)


MAX_CONCURRENT_INFERENCE = _read_max_concurrent_inference()
INFERENCE_SLOTS = anyio.Semaphore(MAX_CONCURRENT_INFERENCE)

__all__ = [
    "INFERENCE_SLOTS",
    "MAX_CONCURRENT_INFERENCE",
    "MAX_TEXT_INPUT_CHARS",
]
