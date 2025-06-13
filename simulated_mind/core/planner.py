"""Graph-of-Thoughts based Planner.

This planner uses a Graph-of-Thoughts (GoT) engine to decompose goals into
a sequence of actionable steps. It relies on an underlying LLM to generate
reasoning steps and a LogicEngine to execute the final plan.
"""
from __future__ import annotations

import os
import re
import sys
import json
from typing import Any, Callable, Dict, List, Optional

from .logic_engine import LogicEngine, LogicContext, LogicEngineError
from .logic_graph import LogicGraph
from .logic_primitives import PRIMITIVE_REGISTRY


from ..journal.journal import Journal



# Placeholder for Goal dataclass/object
# Expected attributes: id, description, priority, sub_goals, parent_goal
class Goal:
    def __init__(self, id: str, description: str, priority: int = 0, sub_goals: Optional[List[Goal]] = None, parent_goal: Optional[Goal] = None, created_by: Optional[str] = None):
        self.id = id
        self.description = description
        self.priority = priority
        self.sub_goals = sub_goals or []
        self.parent_goal = parent_goal
        self.created_by = created_by

    def __repr__(self):
        return f"Goal(id='{self.id}', description='{self.description}', priority={self.priority})"

MAX_RECURSION_DEPTH = 3 # Configurable maximum recursion depth

class Planner:
    """A planner that uses a Graph-of-Thoughts (GoT) engine to decompose goals into actionable steps. It integrates with a LogicEngine for execution and a TaskManager for goal persistence."""

    def __init__(self, memory_store: Any | None = None, journal: Journal | None = None, goal_class: Optional[Type[Goal]] = None, primitive_registry: Optional[Dict[str, Callable]] = None, task_manager: Any = None, local_llm_client: Any = None):
        self.primitive_registry = primitive_registry or PRIMITIVE_REGISTRY
        try:
            self.memory_store = memory_store
            self.journal = journal or Journal.null()
            self.goal_class = goal_class # Assign goal_class
            self._goal_id_counter = 0 # For generating unique goal IDs if not provided
            self.plan_cache: Dict[str, List[Goal]] = {} # Initialize plan_cache
            self.task_manager = task_manager
            self.local_llm_client = local_llm_client
        except Exception as e_planner_init:
            print(f"ERROR in Planner.__init__: {e_planner_init}\n", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise # Re-raise after printing to ensure test failure


    def _generate_goal_id(self, prefix: str = "goal") -> str:
        self._goal_id_counter += 1
        return f"{prefix}_{self._goal_id_counter}"

    # ------------------------------------------------------------------
    # Public API expected by SubAgent / MetaAgent
    # ------------------------------------------------------------------



    def _create_plan_from_got_graph(self, goal_description: str, initial_context_data: Optional[Dict[str, Any]] = None) -> Optional[List['Goal']]:
        """
        Create a plan using Graph-of-Thoughts reasoning with RWKV7.
        """
        from ..orchestration.graph_of_thoughts import create_got_engine
        self.journal.log_event("planner.got_graph.start", {"goal": goal_description})
        try:
            if not self.local_llm_client or not self.local_llm_client.is_available():
                self.journal.log_event("planner.got_graph.no_rwkv7", {"goal": goal_description})
                return None
            got_engine = create_got_engine(self.local_llm_client, self.journal)
            got_graph_path = "simulated_mind/templates/graphs/got_multi_step_reasoning.yaml"
            absolute_path = os.path.normpath(os.path.join(
                os.path.dirname(__file__), '..', '..', got_graph_path
            ))
            if not os.path.exists(absolute_path):
                self.journal.log_event("planner.got_graph.template_missing", {"path": absolute_path})
                return None
            got_graph = LogicGraph.load_from_yaml(absolute_path)
            reasoning_context = got_engine.create_reasoning_context(
                goal_description, 
                reasoning_type="analytical"
            )
            final_context = got_engine.execute_graph_of_thoughts(
                got_graph, 
                reasoning_context, 
                base_prompt=f"Break down this goal into actionable steps: {goal_description}"
            )
            if final_context.get_error():
                self.journal.log_event("planner.got_graph.execution_error", {
                    "goal": goal_description,
                    "error": final_context.get_error()
                })
                return None
            final_response = final_context.get_variable('final_response', '')
            sub_goals = self._parse_goals_from_got_response(final_response)
            if sub_goals:
                self.journal.log_event("planner.got_graph.success", {
                    "goal": goal_description,
                    "sub_goals_count": len(sub_goals)
                })
                goals = []
                for sub_goal in sub_goals:
                    goal = self._make_goal(
                        description=sub_goal,
                        created_by="got_reasoning",
                        priority=5,
                        parent_goal=None
                    )
                    goals.append(self._persist_goal(goal))
                return goals
            return None
        except Exception as e:
            self.journal.log_event("planner.got_graph.error", {
                "goal": goal_description,
                "error": str(e)
            })
            return None

    def _parse_goals_from_got_response(self, response_text: str) -> List[str]:
        """Parses a list of sub-goals from the raw text response of the GoT engine."""
        self.journal.log_event("planner.got_graph.parse_start", {"raw_response": response_text})
        json_match = re.search(
            r'\['
            r'\s*(?P<content>"(?:[^"\\]|\\.)*"\s*(?:,\s*"(?:[^"\\]|\\.)*")*\s*)?'
            r'\s*\]',
            response_text,
            re.DOTALL,
        )
        if json_match:
            try:
                subtasks = json.loads(json_match.group(0))
                if isinstance(subtasks, list) and all(isinstance(t, str) for t in subtasks):
                    parsed = [t.strip() for t in subtasks if t.strip()]
                    self.journal.log_event("planner.got_graph.parse_success", {"method": "json", "count": len(parsed)})
                    return parsed
            except json.JSONDecodeError as je:
                self.journal.log_event("planner.got_graph.parse_json_error", {"error": str(je)})

        lines = [
            re.sub(r'^\s*\d+\.\s*', '', l).strip(" -*\"'[]") 
            for l in response_text.split('\n') 
            if l.strip()
        ]
        meaningful = [
            l for l in lines
            if len(l) > 10 and not any(kw in l.lower() for kw in ["task", "json", "array", "goal", "here are"]) 
        ]

        if meaningful:
            self.journal.log_event("planner.got_graph.parse_success", {"method": "line_split_meaningful", "count": len(meaningful)})
            return meaningful[:5]
        
        if lines:
            self.journal.log_event("planner.got_graph.parse_success", {"method": "line_split_raw", "count": len(lines)})
            return lines[:5]

        self.journal.log_event("planner.got_graph.parse_fail", {"raw_response": response_text})
        return []



    def create_plan(self, goal_description: str) -> List[Goal]:
        """Creates a plan (list of Goal objects) for a given goal description using the Graph-of-Thoughts engine.

        Persists all subgoals via TaskManager if available.
        """
        self.journal.log_event("planner.create_plan.start", {"goal": goal_description, "engine": "got_reasoning"})

        # Always use the Graph-of-Thoughts engine for planning.
        if getattr(self, 'local_llm_client', None):
            got_goals = self._create_plan_from_got_graph(goal_description)
            if got_goals:
                self.journal.log_event("planner.create_plan.success", {
                    "goal": goal_description, 
                    "source": "got_reasoning", 
                    "sub_tasks_count": len(got_goals)
                })
                return got_goals
        
        self.journal.log_event("planner.create_plan.fail", {
            "goal": goal_description,
            "reason": "GoT planning failed or LLM client not available."
        })
        return []

    def _persist_goal(self, goal: Goal) -> Goal:
        if self.task_manager:
            return self.task_manager.create_goal(
                description=goal.description,
                priority=goal.priority,
                parent_goal=goal.parent_goal,
                created_by=goal.created_by
            )
        return goal

    def _make_goal(self, description: str, created_by: str = None, priority: int = 5, parent_goal: Goal = None) -> Goal:
        return self.goal_class(
            id=self._generate_goal_id(),
            description=description,
            priority=priority,
            parent_goal=parent_goal,
            created_by=created_by
        )
