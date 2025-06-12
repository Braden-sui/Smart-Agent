"""LearningLoop: analyses history in mem0 and suggests improvements.

MVP implementation simply records that a review occurred. Future versions will
look for repetitive failure patterns, enrich planner templates, etc.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..memory.dao import MemoryDAO
from ..journal.journal import Journal


class LearningLoop:  # pragma: no cover – placeholder
    def __init__(self, memory: MemoryDAO, journal: Journal | None = None):
        self.memory = memory
        self.journal = journal or Journal.null()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, context: dict[str, Any] | None = None) -> None:
        """Run a single learning review and log outcome to memory."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {},
            "note": "MVP review completed",
        }
        self.memory.store_memory(user_id="learning_system", memory_id=f"review:{payload['timestamp']}", content=payload, tags=["learning", "review"])
        self.journal.log_event("learning.review", payload)
