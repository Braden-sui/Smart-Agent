"""Simple transparency‐oriented journal helper.

Agents and subsystems call ``Journal.log_event(label, payload)`` to emit
structured events. A default sink prints to stdout; a null journal suppresses
all output but maintains the same interface.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict


class Journal:
    """Lightweight event logger with pluggable sink."""

    def __init__(self, sink: Callable[[str, Dict[str, Any]], None] | None = None):
        self._sink = sink or self._default_sink

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(self, label: str, payload: Dict[str, Any] | None = None) -> None:  # noqa: D401
        """Emit an event *label* and JSON-serialise *payload* to the sink."""
        payload = payload or {}
        try:
            self._sink(label, payload)
        except Exception:
            # Never allow logging failures to break core logic.
            pass

    # ------------------------------------------------------------------
    # Sinks
    # ------------------------------------------------------------------

    @staticmethod
    def _default_sink(label: str, payload: Dict[str, Any]) -> None:
        print(f"[JOURNAL] {label} | {json.dumps(payload, default=str)}")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def null(cls) -> "Journal":
        """Return a Journal that drops all events."""
        return cls(lambda _label, _payload: None)
