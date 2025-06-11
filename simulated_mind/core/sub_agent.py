"""SubAgent: a lightweight worker with its own small Planner and memory slice."""
from __future__ import annotations

from typing import Any, Type

from .base_agent import Action, BaseAgent
from .planner import Planner, Goal  # Import Planner and Goal
from ..memory.dao import MemoryDAO
from ..logging.journal import Journal


class SubAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        memory: MemoryDAO,
        journal: Journal | None = None,
    ):
        super().__init__(name=f"SubAgent-{agent_id[:8]}", journal=journal)
        self.id = agent_id
        self.memory = memory
        self.planner = Planner(memory_store=memory, journal=self.journal, goal_class=Goal) # Pass Goal class
        self.journal.log_event("sub_agent.init", {"agent_id": self.id})

    # ------------------------------------------------------------------
    # Decision Logic
    # ------------------------------------------------------------------

    def decide(self) -> Action:
        # Placeholder logic: generate a planning task if none exist.
        retrieved_record = self.memory.retrieve_memory(user_id=self.id, memory_id="agent_tasks")
        tasks = retrieved_record.get("content") if retrieved_record else []
        if not tasks:
            # Use planner to create tasks from last event (if string)
            goal_desc = getattr(self, "_last_event", "generic goal")
            subtasks = self.planner.create_plan(goal_desc)
            self.memory.store_memory(
                user_id=self.id,
                memory_id="agent_tasks",
                content=subtasks,
                tags=["task"],
            )
            self.journal.log_event("sub_agent.plan", {"agent_id": self.id, "subtasks": subtasks})
            return Action(kind="planned", payload=subtasks)
        return Action(kind="noop")
