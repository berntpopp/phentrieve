"""
Pytest configuration for script tests.

Provides shared fixtures for testing standalone scripts.
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
