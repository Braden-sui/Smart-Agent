from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..journal.journal import Journal


class Action:
    """Generic action produced by an agent during the decide() step."""

    def __init__(self, kind: str, payload: Any | None = None):
        self.kind = kind
        self.payload = payload

    def __repr__(self) -> str:  # pragma: no cover
        return f"Action(kind={self.kind!r}, payload={self.payload!r})"


class BaseAgent(ABC):
    """Minimal agent interface used by MetaAgent and SubAgents with transparency logging."""

    def __init__(self, name: str, journal: Journal | None = None):
        self.name = name
        self.journal = journal or Journal.null()

    # ---- Perception → Decision → Action Loop ---------------------------------

    def perceive(self, input_event: Any) -> None:
        """Consume an input event and update internal state if required."""
        # Default behaviour: store last event; subclasses can override.
        self._last_event = input_event
        self.journal.log_event("agent.perceive", {"agent": self.name, "event": repr(input_event)})

    @abstractmethod
    def decide(self) -> Action:
        """Produce the next action given current internal state."""

    def act(self, action: Action) -> None:
        """Execute the chosen action. Default: noop (for thinking agents)."""
        # Concrete subclasses that interact with the outside world override this.
        self.journal.log_event("agent.act", {"agent": self.name, "action": repr(action)})

    # ---- Convenience ----------------------------------------------------------

    def process(self, input_event: Any) -> None:
        """Run a full perceive→decide→act cycle for a single event."""
        self.journal.log_event("agent.process.start", {"agent": self.name})
        self.perceive(input_event)
        action = self.decide()
        self.act(action)
        self.journal.log_event("agent.process.end", {"agent": self.name})
