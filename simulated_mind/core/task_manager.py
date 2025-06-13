"""
TaskManager: Manages Goal objects and persists them using MemoryDAO.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import uuid
from .planner import Goal
from ..memory.cognitive_dao import RWKV7CognitiveMemoryDAO
from ..memory.types import MemoryType
from simulated_mind.journal.journal import Journal

class TaskManager:
    def __init__(self, memory_dao: RWKV7CognitiveMemoryDAO, journal: Optional[Journal] = None, user_id: Optional[str] = None):
        self.memory_dao = memory_dao
        self.journal = journal or Journal.null()
        self.user_id = user_id or "default_user"
        self._goal_cache: Dict[str, Goal] = {}

    def create_goal(self, description: str, priority: int = 0, parent_goal: Optional[Goal] = None, created_by: Optional[str] = None) -> Goal:
        goal_id = str(uuid.uuid4())
        goal = Goal(id=goal_id, description=description, priority=priority, parent_goal=parent_goal, created_by=created_by)
        self._goal_cache[goal_id] = goal
        try:
            content_json = json.dumps({
                "id": goal.id,
                "description": goal.description,
                "priority": goal.priority,
                "parent_goal": goal.parent_goal.id if goal.parent_goal else None,
                "created_by": goal.created_by,
                "sub_goals": [sg.id for sg in goal.sub_goals],
            })
            self.memory_dao.add_memory(
                user_id=self.user_id,
                memory_type=MemoryType.META,
                content=content_json,
                metadata={"memory_id": goal_id, "tags": ["goal"], "status": "active"}
            )
            self.journal.log_event("task_manager.create_goal", {"goal_id": goal_id, "description": description})
        except Exception as e:
            self.journal.log_event("task_manager.create_goal_error", {"error": str(e), "goal_id": goal_id})
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        if goal_id in self._goal_cache:
            return self._goal_cache[goal_id]
        try:
            results = self.memory_dao.get_memories(
                user_id=self.user_id,
                memory_type=MemoryType.META,
                query="",
                limit=1,
                metadata_filter={"memory_id": goal_id}
            )
            if results:
                data = results[0]
                content = json.loads(data["content"])
                goal = Goal(
                    id=content["id"],
                    description=content["description"],
                    priority=content.get("priority", 0),
                    parent_goal=None,  # parent lookup could be added if needed
                    created_by=content.get("created_by")
                )
                self._goal_cache[goal_id] = goal
                return goal
            return None
        except Exception as e:
            self.journal.log_event("task_manager.get_goal_error", {"error": str(e), "goal_id": goal_id})
            return None

    def update_goal(self, goal_id: str, **kwargs) -> bool:
        goal = self.get_goal(goal_id)
        if not goal:
            self.journal.log_event("task_manager.update_goal_error", {"error": "Goal not found", "goal_id": goal_id})
            return False
        for k, v in kwargs.items():
            if hasattr(goal, k):
                setattr(goal, k, v)
        try:
            content_json = json.dumps({
                "id": goal.id,
                "description": goal.description,
                "priority": goal.priority,
                "parent_goal": goal.parent_goal.id if goal.parent_goal else None,
                "created_by": goal.created_by,
                "sub_goals": [sg.id for sg in goal.sub_goals],
            })
            self.memory_dao.add_memory(
                user_id=self.user_id,
                memory_type=MemoryType.META,
                content=content_json,
                metadata={"memory_id": goal_id, "tags": ["goal"], "status": "active"}
            )
            self.journal.log_event("task_manager.update_goal", {"goal_id": goal_id, "updates": kwargs})
            return True
        except Exception as e:
            self.journal.log_event("task_manager.update_goal_error", {"error": str(e), "goal_id": goal_id})
            return False

    def list_goals(self, status: Optional[str] = None) -> List[Goal]:
        try:
            results = self.memory_dao.find_memories_by_tags(self.user_id, ["goal"])
            goals = []
            for data in results:
                if not status or data.get("metadata", {}).get("status") == status:
                    content = json.loads(data.get("content", "{}"))
                    goal = Goal(
                        id=content["id"],
                        description=content["description"],
                        priority=content.get("priority", 0),
                        parent_goal=None,  # parent lookup could be added if needed
                        created_by=content.get("created_by")
                    )
                    goals.append(goal)
            return goals
        except Exception as e:
            self.journal.log_event("task_manager.list_goals_error", {"error": str(e)})
            return []

    def complete_goal(self, goal_id: str) -> bool:
        goal = self.get_goal(goal_id)
        if not goal:
            self.journal.log_event("task_manager.complete_goal_error", {"error": "Goal not found", "goal_id": goal_id})
            return False
        try:
            content_json = json.dumps({
                "id": goal.id,
                "description": goal.description,
                "priority": goal.priority,
                "parent_goal": goal.parent_goal.id if goal.parent_goal else None,
                "created_by": goal.created_by,
                "sub_goals": [sg.id for sg in goal.sub_goals],
            })
            self.memory_dao.add_memory(
                user_id=self.user_id,
                memory_type=MemoryType.META,
                content=content_json,
                metadata={"memory_id": goal_id, "tags": ["goal"], "status": "completed"}
            )
            self.journal.log_event("task_manager.complete_goal", {"goal_id": goal_id})
            return True
        except Exception as e:
            self.journal.log_event("task_manager.complete_goal_error", {"error": str(e), "goal_id": goal_id})
            return False
