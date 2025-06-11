"""The MetaAgent oversees sub-agents, mem0 access, and safety checks."""
from __future__ import annotations

import uuid
from typing import Any, Dict

from .base_agent import Action, BaseAgent
from .sub_agent import SubAgent
from ..memory.dao import MemoryDAO  # local relative import
from ..safety.guard import SafetyGuard
from ..learning.loop import LearningLoop
from ..logging.journal import Journal


class MetaAgent(BaseAgent):
    """Top-level executive agent that orchestrates the system."""

    def __init__(
        self,
        memory_dao: MemoryDAO,
        safety_guard: SafetyGuard,
        journal: Journal | None = None,
    ):
        super().__init__(name="MetaAgent", journal=journal)
        self.memory = memory_dao
        self.safety = safety_guard
        self.sub_agents: Dict[str, SubAgent] = {}
        # Initialize learning subsystem
        self.learning_loop = LearningLoop(memory_dao, journal=self.journal)

    # ---------------------------------------------------------------------
    # Sub-Agent Lifecycle
    # ---------------------------------------------------------------------

    def spawn_sub_agent(self, spec: dict[str, Any] | None = None) -> str:
        """Instantiate a new SubAgent with optional spec and return its id."""
        agent_id = str(uuid.uuid4())
        agent = SubAgent(agent_id, self.memory, journal=self.journal)
        self.sub_agents[agent_id] = agent
        # Log creation in mem0
        self.memory.store_memory(
            user_id=agent_id,
            memory_id="experience_spawn",
            content={"event": "spawn", "spec": spec or {}},
            tags=["spawn", "agent"],
        )
        self.journal.log_event("meta.spawn_sub_agent", {"agent_id": agent_id})
        return agent_id

    def retire_agent(self, agent_id: str) -> None:
        agent = self.sub_agents.pop(agent_id, None)
        if not agent:
            return
        self.memory.store_memory(
            user_id=agent_id,
            memory_id="experience_retire",
            content={"event": "retire"},
            tags=["retire", "agent"],
        )
        self.journal.log_event("meta.retire_agent", {"agent_id": agent_id})

    # ---------------------------------------------------------------------
    # Decision Loop
    # ---------------------------------------------------------------------

    def decide(self) -> Action:
        """Meta-level decision making: check sub-agent statuses, schedule tasks."""
        # Simplest MVP: no external actions, purely internal orchestration.
        return Action(kind="noop")

    def run_learning_review(self) -> None:
        """Trigger a single learning review cycle."""
        self.learning_loop.review()

    # For now MetaAgent does not override act(); inherits BaseAgent.act
