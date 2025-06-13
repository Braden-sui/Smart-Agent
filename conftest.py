"""Pytest configuration to ensure project root is on ``sys.path``.

This makes the top-level ``simulated_mind`` package importable when running
``pytest`` in ``importlib`` mode (as enabled via ``pytest.ini``).
"""
from __future__ import annotations

# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables from .env file

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    # Prepend to guarantee priority over globally installed packages.
    sys.path.insert(0, str(PROJECT_ROOT))
