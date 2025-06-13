"""Re-export Journal for backward-compatibility test imports.

Allows `from simulated_mind.logging.journal import Journal` to work even
though the primary implementation lives in `simulated_mind.journal.journal`.
"""

from ..journal.journal import Journal  # noqa: F401

__all__ = ["Journal"]
