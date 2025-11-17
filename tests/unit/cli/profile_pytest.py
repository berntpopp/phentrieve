"""Profile pytest collection to find bottlenecks."""

import cProfile
import pstats
import sys

# Run pytest collection with profiling
profiler = cProfile.Profile()
profiler.enable()

# Import and run pytest
import pytest

sys.exit(
    pytest.main(
        [
            "tests/unit/cli/test_data_commands.py",
            "--collect-only",
            "--no-cov",
            "-p",
            "no:cov",
        ]
    )
)

profiler.disable()

# Print top 50 time-consuming functions
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(50)
