"""Pytest configuration to ensure project root is on ``sys.path``.

This makes the top-level ``simulated_mind`` package importable when running
``pytest`` in ``importlib`` mode (as enabled via ``pytest.ini``).
"""
from __future__ import annotations

# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env file

import sys
from pathlib import Path
import pytest
import unittest.mock as _umock

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    # Prepend to guarantee priority over globally installed packages.
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def mocker():
    """Lightweight fallback for the pytest-mock ``mocker`` fixture.

    Provides access to the standard ``unittest.mock`` module when the
    external ``pytest-mock`` plugin is not installed. Only a subset of the
    API is exposed but it is sufficient for the test-suite, which only
    requires the fixture object itself.
    """
    return _umock
