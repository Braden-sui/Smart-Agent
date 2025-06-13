"""SubAgent: a lightweight worker with its own small Planner and memory slice."""
from __future__ import annotations

from typing import Any, Type

from .base_agent import Action, BaseAgent
import json
from .planner import Planner, Goal
from ..memory.cognitive_dao import RWKV7CognitiveMemoryDAO
from ..memory.types import MemoryType
from simulated_mind.journal.journal import Journal


class SubAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        memory: RWKV7CognitiveMemoryDAO,
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
        retrieved_mems = self.memory.get_memories(
            user_id=self.id, query="agent_tasks", limit=1, memory_type=MemoryType.META
        )
        tasks_json = retrieved_mems[0].get("content") if retrieved_mems else None
        tasks = json.loads(tasks_json) if tasks_json else []
        if not tasks:
            # Use planner to create tasks from last event (if string)
            goal_desc = getattr(self, "_last_event", "generic goal")
            subtasks = self.planner.create_plan(goal_desc)
            # Convert Goal objects to a serializable format (e.g., dicts)
            subtasks_serializable = [subtask.to_dict() for subtask in subtasks]
            self.memory.add_memory(
                user_id=self.id,
                memory_type=MemoryType.META,
                content=json.dumps(subtasks_serializable),
                metadata={"memory_id": "agent_tasks", "tags": ["task"]},
            )
            # Upward reporting: report planned subtasks to CEO/global workspace
            if hasattr(self, 'ceo_user_id') and self.ceo_user_id:
                self.memory.report_to_ceo(
                    ceo_user_id=self.ceo_user_id,
                    subagent_id=self.id,
                    knowledge={"subtasks": subtasks, "goal_desc": goal_desc},
                    task_id=goal_desc,
                    report_type="planning",
                    tags=["auto_report"],
                    metadata={"source": "SubAgent.decide"}
                )
            self.journal.log_event("sub_agent.plan", {"agent_id": self.id, "subtasks": subtasks})
            return Action(kind="planned", payload=subtasks)
        return Action(kind="noop")
