"""ThoughtNode data structure and supporting enums for Graph-of-Thoughts.

This module defines the core in-memory representation used by the Graph-of-Thoughts
reasoning engine. All validation rules are enforced at construction time and
on every state-mutating method call to guarantee graph integrity.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

__all__ = [
    "ThoughtType",
    "ThoughtStatus",
    "ThoughtNode",
]

LOGGER = logging.getLogger(__name__)


class ThoughtType(str, Enum):
    """Enumeration of canonical thought categories used by GoT."""

    DECOMPOSITION = "decomposition"
    EXPLORATION = "exploration"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    OTHER = "other"


class ThoughtStatus(str, Enum):
    """Finite-state machine statuses for a :class:`ThoughtNode`."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    PRUNED = "pruned"


@dataclass
class ThoughtNode:
    """A single reasoning step within a Graph-of-Thoughts execution.

    All mutating operations update :pyattr:`updated_at` automatically and
    re-validate the node. Any detected integrity issue raises
    :class:`ValueError`.
    """

    id: str  # UUID4 string
    content: str
    thought_type: ThoughtType
    status: ThoughtStatus = ThoughtStatus.PENDING
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0 inclusive
    reasoning_steps: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # -------------------------------------------------------------
    # Lifecycle hooks & validation helpers
    # -------------------------------------------------------------

    def __post_init__(self) -> None:  # noqa: D401 – Imperative style acceptable
        """Run integrity validation immediately after dataclass init."""
        self._validate_id()
        self._validate_content()
        self._validate_confidence()
        # Ensure no duplicates across parent/child lists
        self.parent_ids = list(dict.fromkeys(self.parent_ids))
        self.child_ids = list(dict.fromkeys(self.child_ids))

    # -------------------- Public API --------------------

    def add_evidence(self, key: str, value: Any, confidence: float = 1.0) -> None:
        """Attach supporting evidence to the node.

        Parameters
        ----------
        key:
            Unique key describing the evidence type (e.g. "citation_1").
        value:
            Arbitrary JSON-serialisable payload.
        confidence:
            Confidence score in *[0, 1]* for the supplied evidence.
        """
        if not key or not isinstance(key, str):
            raise ValueError("Evidence key must be a non-empty string")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Evidence confidence must be within [0.0, 1.0]")
        self.evidence[key] = {"data": value, "confidence": confidence, "ts": datetime.utcnow().isoformat()}
        self._touch()

    def update_confidence(self, new_confidence: float) -> None:
        """Update the node-level confidence value with validation."""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be within [0.0, 1.0]")
        self.confidence = new_confidence
        self._touch()

    def validate_integrity(self) -> bool:
        """Run full validation pass. Returns *True* if node is valid."""
        try:
            self._validate_id()
            self._validate_content()
            self._validate_confidence()
            return True
        except ValueError as exc:
            LOGGER.error("ThoughtNode integrity check failed: %s", exc)
            return False

    # Serialisation helpers -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to a JSON-serialisable dictionary (defensive copy)."""
        data = asdict(self)
        # Convert Enum members to their value for JSON compatibility
        data["thought_type"] = self.thought_type.value
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return json.loads(json.dumps(data))  # Deep copy via JSON roundtrip

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtNode":
        """Reconstruct :class:`ThoughtNode` from dict, validating input."""
        required_fields = {"id", "content", "thought_type"}
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(f"Missing required fields for ThoughtNode: {missing}")
        try:
            node = cls(
                id=data["id"],
                content=data["content"],
                thought_type=ThoughtType(data["thought_type"]),
                status=ThoughtStatus(data.get("status", ThoughtStatus.PENDING)),
                parent_ids=list(data.get("parent_ids", [])),
                child_ids=list(data.get("child_ids", [])),
                confidence=float(data.get("confidence", 0.0)),
                reasoning_steps=list(data.get("reasoning_steps", [])),
                evidence=dict(data.get("evidence", {})),
                metadata=dict(data.get("metadata", {})),
                created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else datetime.utcnow(),
                updated_at=datetime.fromisoformat(data.get("updated_at")) if data.get("updated_at") else datetime.utcnow(),
            )
        except Exception as exc:  # noqa: BLE001 – Provide context
            raise ValueError(f"Invalid ThoughtNode data: {exc}") from exc
        return node

    # -------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------

    def _validate_id(self) -> None:
        try:
            uuid_obj = uuid.UUID(self.id)
            if str(uuid_obj) != self.id:
                raise ValueError("ID must be a canonical UUID4 string")
        except Exception as exc:
            raise ValueError("Invalid UUID for ThoughtNode.id") from exc

    def _validate_content(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("ThoughtNode.content must be a non-empty string")

    def _validate_confidence(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be within [0.0, 1.0]")

    def _touch(self) -> None:
        """Update *updated_at* timestamp and re-validate integrity."""
        self.updated_at = datetime.utcnow()
        assert self.validate_integrity(), "ThoughtNode entered invalid state after modification"

    # -------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------

    def __hash__(self) -> int:  # Enables use in sets / dict keys
        return hash(self.id)

    def __repr__(self) -> str:  # pragma: no cover – readability
        return (
            "ThoughtNode("  # noqa: WPS237 – long repr acceptable
            f"id={self.id!r}, content_len={len(self.content)}, type={self.thought_type}, "
            f"status={self.status}, confidence={self.confidence:.2f})"
        )
