"""Compatibility shim for legacy imports.

Tests (and possibly external code) expect ``simulated_mind.logging.journal.Journal``.
The core implementation now lives in ``simulated_mind.journal.journal``.
This package re-exports :class:`Journal` to keep that import path working.
"""

from ..journal.journal import Journal  # noqa: F401
