# tests/unit/__init__.py

"""Unit tests package for Cockatoo V1.

This package contains all unit tests organized by component.
Tests are automatically discovered and collected by pytest.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

__version__ = "1.0.0"
__author__ = "Cockatoo V1 Team"