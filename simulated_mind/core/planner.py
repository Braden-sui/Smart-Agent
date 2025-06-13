"""Recursive, template-aware Planner.

This planner decomposes goals into sub-tasks using predefined templates
and can recursively plan complex goals up to a specified depth.
It also leverages memory for reusing past plans and finding suggestions.
"""
from __future__ import annotations

import os
import re
import sys
import json
from typing import Any, Callable, Dict, List, Optional

from .logic_engine import LogicEngine, LogicContext
from .logic_graph import LogicGraph
from .logic_primitives import PRIMITIVE_REGISTRY


from ..journal.journal import Journal
from ..templates.planner_rules import TEMPLATES, GRAPH_TEMPLATES

# print(file=sys.stderr,f"DEBUG: TEMPLATES loaded at module level: {bool(TEMPLATES)} Keys: {list(TEMPLATES.keys()) if TEMPLATES else 'None'}") # Commented out: module-level print causing issues

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
    """Recursive, template-aware planner, integrating LogicEngine for graph-based planning and persistent goal storage via TaskManager."""

    def __init__(self, memory_store: Any | None = None, journal: Journal | None = None, goal_class: Optional[Type[Goal]] = None, primitive_registry: Optional[Dict[str, Callable]] = None, task_manager: Any = None, local_llm_client: Any = None, *, planner_mode: str = "auto"):
        self.primitive_registry = primitive_registry or PRIMITIVE_REGISTRY
        try:
            self.memory_store = memory_store
            self.journal = journal or Journal.null()
            self.goal_class = goal_class # Assign goal_class
            self._goal_id_counter = 0 # For generating unique goal IDs if not provided
            self.plan_cache: Dict[str, List[Goal]] = {} # Initialize plan_cache
            self.task_manager = task_manager
            self.local_llm_client = local_llm_client
            # Graph-of-Thoughts configuration: "auto", "enabled", "disabled"
            self.planner_mode = planner_mode.lower()
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

    def _create_plan_from_graph(self, goal_description: str, initial_context_data: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
        """Attempts to create a plan by executing a logic graph based on the goal description."""
        self.journal.log_event("planner.graph.start", {"goal": goal_description})
        graph_path = None
        normalized_goal_desc = goal_description.lower().strip()

        # Find a matching graph template
        for keyword, path in GRAPH_TEMPLATES.items():
            if normalized_goal_desc.startswith(keyword.lower()):
                graph_path = path
                self.journal.log_event(
                    "planner.graph.match_found",
                    {"goal": goal_description, "keyword": keyword, "graph_path": graph_path}
                )
                break

        if not graph_path:
            self.journal.log_event("planner.graph.no_match", {"goal": goal_description})
            return None

        try:
            # Construct absolute path for the graph YAML
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            absolute_graph_path = os.path.normpath(os.path.join(project_root, graph_path))

            # Load the graph from the YAML file
            if not os.path.exists(absolute_graph_path):
                self.journal.log_event(
                    "planner.graph.error",
                    {
                        "goal": goal_description,
                        "error": f"Graph YAML file not found at: {absolute_graph_path}",
                        "graph_path_template_val": graph_path # The value from GRAPH_TEMPLATES
                    }
                )
                print(f"ERROR: Graph YAML file not found at: {absolute_graph_path}", file=sys.stderr)
                return None
            
            graph = LogicGraph.load_from_yaml(absolute_graph_path)
            if graph is None:
                self.journal.log_event(
                    "planner.graph.error",
                    {
                        "goal": goal_description,
                        "error": f"LogicGraph.from_yaml returned None for path: {absolute_graph_path}",
                        "graph_path_template_val": graph_path
                    }
                )
                print(f"ERROR: LogicGraph.from_yaml returned None for path: {absolute_graph_path}", file=sys.stderr)
                return None

            # Instantiate the engine and execute the graph
            engine = LogicEngine(self.primitive_registry)

            # Capture stderr from LogicEngine execution
            logic_engine_stderr_path = "logic_engine_stderr.log"
            original_stderr = sys.stderr
            captured_stderr_output = "" # Ensure it's initialized
            try:
                with open(logic_engine_stderr_path, 'w') as temp_stderr_file:
                    sys.stderr = temp_stderr_file
                    final_context = engine.execute_graph(graph, initial_context=LogicContext(initial_context_data or {}))
            finally:
                sys.stderr = original_stderr # Restore stderr
                with open(logic_engine_stderr_path, 'r') as temp_stderr_file:
                    captured_stderr_output = temp_stderr_file.read()
                if captured_stderr_output:
                    print("--- Captured LogicEngine stderr: ---", file=sys.stderr)
                    print(captured_stderr_output, file=sys.stderr)
                    print("--- End LogicEngine stderr ---", file=sys.stderr)
                try:
                    os.remove(logic_engine_stderr_path)
                except OSError:
                    pass # Ignore if file couldn't be removed
                # Always write the captured output to a known trace file
                try:
                    with open("_planner_last_logic_trace.log", "w") as f_trace:
                        f_trace.write(captured_stderr_output if captured_stderr_output else "No stderr output captured from LogicEngine.")
                except Exception as trace_log_e:
                    print(f"ERROR: Could not write to _planner_last_logic_trace.log: {trace_log_e}", file=sys.stderr)

            initial_context = LogicContext(initial_context_data or {})
            initial_context.set_variable('goal_description', goal_description)

            final_context = engine.execute_graph(graph, initial_context)
            sub_goals = final_context.get_variable('sub_goals_list')

            if sub_goals and isinstance(sub_goals, list):
                self.journal.log_event(
                    "planner.graph.success",
                    {
                        "goal": goal_description,
                        "graph_id": final_context.get_variable('graph_id'),
                        "sub_goals_count": len(sub_goals)
                    }
                )
                if self.goal_class:
                    return [self.goal_class(id=self._generate_goal_id(prefix="graph_task"), description=desc, priority=5, created_by="planner_graph") for desc in sub_goals]
                else:
                    return sub_goals
            else:
                self.journal.log_event(
                    "planner.graph.fail",
                    {
                        "goal": goal_description,
                        "reason": "'sub_goals' key not found or not a list in final context",
                        "final_context": final_context.to_dict()
                    }
                )
                return None
        except Exception as e:
            self.journal.log_event(
                "planner.graph.error",
                {
                    "goal": goal_description,
                    "error": str(e),
                    "graph_path_template_val": graph_path,
                    "logic_engine_trace": captured_stderr_output # Add captured trace
                }
            )
            print(f"ERROR: LogicEngine execution failed for graph {graph_path}. Error: {e}", file=sys.stderr)
            if captured_stderr_output: # Ensure it's printed if an error occurs before normal printing
                print("--- Captured LogicEngine stderr (on error): ---", file=sys.stderr)
                print(captured_stderr_output, file=sys.stderr)
                print("--- End LogicEngine stderr (on error) ---", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            if captured_stderr_output:
                try:
                    with open("planner_error.log", "a") as f_err_trace:
                        f_err_trace.write("\n--- Captured LogicEngine stderr (from _create_plan_from_graph error path): ---\n")
                        f_err_trace.write(captured_stderr_output)
                        f_err_trace.write("\n--- End LogicEngine stderr ---\n")
                except Exception as log_e:
                    print(f"ERROR: Could not write LogicEngine trace to planner_error.log: {log_e}", file=sys.stderr)
            return None
            engine = LogicEngine(graph, self.primitive_registry)
            initial_context = LogicContext(initial_data=context_data)
            
            try:
                print(f"DEBUG_GRAPH_EXEC: Attempting to execute graph: {graph_path} with ID: {graph.graph_id}", file=sys.stderr); sys.stderr.flush()
                engine = LogicEngine(graph, self.primitive_registry, self.journal) # Pass journal here
                initial_context = LogicContext(initial_data=context_data)
                print(f"DEBUG_GRAPH_EXEC: Initial context for graph: {initial_context.data}", file=sys.stderr); sys.stderr.flush()
                final_context = engine.execute(initial_context)
                print(f"DEBUG_GRAPH_EXEC: Final context after graph execution: {final_context.data if final_context else 'None'}", file=sys.stderr); sys.stderr.flush()

                if not final_context or final_context.has_error():
                    self.journal.log_event(
                        "planner.graph.execution_error",
                        {"goal": goal_description, "graph_id": graph.graph_id, "error": final_context.get_error() if final_context else "Unknown error, final_context is None"}
                    )
                    # print(f"[Planner] Error executing graph '{graph.graph_id}' for goal '{goal_description}': {final_context.get_error() if final_context else 'Unknown error, final_context is None'}", file=sys.stderr) # Debug
                    return None

                sub_goals_list = final_context.get_variable("sub_goals_list") if final_context else None
                print(f"DEBUG_GRAPH_EXEC: Retrieved 'sub_goals_list' from final_context: {sub_goals_list} (type: {type(sub_goals_list)})", file=sys.stderr); sys.stderr.flush()
                if isinstance(sub_goals_list, list) and all(isinstance(sg, str) for sg in sub_goals_list):
                    self.journal.log_event(
                        "planner.graph.success",
                        {"goal": goal_description, "graph_id": graph.graph_id, "sub_goals_count": len(sub_goals_list)}
                    )
                    # print(f"[Planner] Graph '{graph.graph_id}' executed successfully for '{goal_description}'. Sub-goals: {sub_goals_list}", file=sys.stderr) # Debug
                    return sub_goals_list
                else:
                    self.journal.log_event(
                        "planner.graph.output_mismatch",
                        {"goal": goal_description, "graph_id": graph.graph_id, "expected_output_key": "sub_goals_list", "actual_value": sub_goals_list}
                    )
                    # print(f"[Planner] Graph '{graph.graph_id}' for '{goal_description}' did not produce 'sub_goals_list' as a list of strings. Got: {sub_goals_list}", file=sys.stderr) # Debug
                    return None
            except Exception as e_engine_exec:
                print(f"ERROR_DEBUG: Exception during LogicEngine execution or result processing in _create_plan_from_graph: {type(e_engine_exec).__name__}: {e_engine_exec}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                raise

        except Exception as e:
            self.journal.log_event(
                "planner.graph.unexpected_error",
                {"goal": goal_description, "graph_path": graph_path, "error": str(e)}
            )
            # print(f"[Planner] Unexpected error during graph planning for '{goal_description}': {e}", file=sys.stderr) # Debug
            # import traceback; traceback.print_exc(file=sys.stderr) # Debug
            return None

    def _create_plan_from_got_graph(self, goal_description: str, initial_context_data: Optional[Dict[str, Any]] = None) -> Optional[List['Goal']]:
        """
        Create a plan using Graph-of-Thoughts reasoning with RWKV7.
        Enhanced version of _create_plan_from_graph with GoT capabilities.
        """
        from ..orchestration.graph_of_thoughts import create_got_engine
        self.journal.log_event("planner.got_graph.start", {"goal": goal_description})
        # Check if this is a GoT-compatible goal
        if not self._is_got_suitable(goal_description):
            return None
        try:
            # Create GoT engine if we have RWKV7 client
            if not self.local_llm_client or not hasattr(self.local_llm_client, 'model'):
                self.journal.log_event("planner.got_graph.no_rwkv7", {"goal": goal_description})
                return None
            got_engine = create_got_engine(self.local_llm_client, self.journal)
            # Load GoT reasoning template
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
            # Execute GoT reasoning
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
            # Extract sub-goals from GoT reasoning
            final_response = final_context.get_variable('final_response', '')
            sub_goals = self._parse_goals_from_got_response(final_response)
            if sub_goals:
                self.journal.log_event("planner.got_graph.success", {
                    "goal": goal_description,
                    "sub_goals_count": len(sub_goals),
                    "reasoning_quality": len(final_response)
                })
                goals = []
                for i, sub_goal in enumerate(sub_goals):
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

    def _is_got_suitable(self, goal_description: str) -> bool:
        """Check if goal is suitable for Graph-of-Thoughts reasoning, taking the planner_mode into account."""
        # Explicit override
        if self.planner_mode == "enabled":
            return True
        if self.planner_mode == "disabled":
            return False

        # Automatic heuristic
        got_indicators = [
            "analyze", "compare", "evaluate", "design", "strategy",
            "complex", "multi-step", "reasoning", "problem-solving"
        ]
        return any(indicator in goal_description.lower() for indicator in got_indicators)

    def create_plan(self, goal_description: str) -> List[Goal]:
        """Creates a plan (list of Goal objects) for a given goal description.
        Enhanced with Graph-of-Thoughts reasoning for complex goals.
        Priority: GoT -> Graph-based -> Template/Memory-based decomposition.
        Persists all subgoals via TaskManager if available.
        """
        self.journal.log_event("planner.create_plan.start", {"goal": goal_description})
        # 1. Attempt Graph-of-Thoughts planning if mode permits
        use_got = (
            self.planner_mode == "enabled" or
            (self.planner_mode == "auto" and self._is_got_suitable(goal_description))
        )

        if use_got and getattr(self, 'local_llm_client', None):
            got_goals = self._create_plan_from_got_graph(goal_description)
            if got_goals is not None:
                self.journal.log_event("planner.create_plan.source", {
                    "goal": goal_description, 
                    "source": "got_reasoning", 
                    "sub_tasks_count": len(got_goals)
                })
                return got_goals
        # 2. Attempt traditional graph-based planning
        graph_sub_tasks = self._create_plan_from_graph(goal_description)
        goals: List[Goal] = []
        if graph_sub_tasks is not None:
            self.journal.log_event("planner.create_plan.source", {"goal": goal_description, "source": "graph_engine", "sub_tasks_count": len(graph_sub_tasks)})
            for sub in graph_sub_tasks:
                if isinstance(sub, Goal):
                    goals.append(self._persist_goal(sub))
                else:
                    goals.append(self._persist_goal(self._make_goal(sub, created_by="planner_graph")))
            return goals
        # 3. Fallback to existing decomposition logic
        self.journal.log_event("planner.create_plan.fallback_to_decompose", {"goal": goal_description})
        fallback_goals = self.decompose_goal(goal_description)
        self.journal.log_event("planner.create_plan.source", {
            "goal": goal_description, 
            "source": "decompose_goal_fallback", 
            "sub_tasks_count": len(fallback_goals)
        })
        for sub in fallback_goals:
            if isinstance(sub, Goal):
                goals.append(self._persist_goal(sub))
            else:
                goals.append(self._persist_goal(self._make_goal(sub, created_by="planner_fallback")))
        return goals

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

        """Creates a plan (list of sub-task descriptions) for a given goal description.
        
        Attempts to use graph-based planning first, then falls back to template/memory-based decomposition.
        """
        self.journal.log_event("planner.create_plan.start", {"goal": goal_description})
        
        # 1. Attempt graph-based planning
        graph_sub_tasks = self._create_plan_from_graph(goal_description)
        if graph_sub_tasks is not None: # Check for None, as an empty list could be a valid plan
            self.journal.log_event("planner.create_plan.source", {"goal": goal_description, "source": "graph_engine", "sub_tasks_count": len(graph_sub_tasks)})
            return graph_sub_tasks

        # 2. Fallback to existing decomposition logic (templates, memory, generic)
        self.journal.log_event("planner.create_plan.fallback_to_decompose", {"goal": goal_description})
        fallback_sub_tasks = self.decompose_goal(goal_description)
        self.journal.log_event("planner.create_plan.source", {"goal": goal_description, "source": "decompose_goal_fallback", "sub_tasks_count": len(fallback_sub_tasks)})
        return fallback_sub_tasks

    def decompose_goal(self, goal_description: str, goal_id_prefix: str = "subgoal") -> List[Goal]:
        """Decomposes a goal description into sub-task Goal objects using templates, memory, or local LLM. Persists via TaskManager if available."""
        subtasks: List[str] = []
        decomposition_source = "fallback"
        normalized_goal_desc = goal_description.strip().lower()
        for template_key, template_subtasks in TEMPLATES.items():
            if normalized_goal_desc.startswith(template_key):
                subtasks = list(template_subtasks) # Create a copy
                decomposition_source = f"template:{template_key}"
                self.journal.log_event(
                    "planner.decompose_goal.template_match",
                    {"goal": goal_description, "template_key": template_key, "subtasks_count": len(subtasks)}
                )
                break
        # If no template match and local LLM is available, use it
        if not subtasks and self.local_llm_client and self.local_llm_client.is_available():
            try:
                subtasks = self._local_llm_decompose_goal(goal_description)
                decomposition_source = "local_llm"
                self.journal.log_event(
                    "planner.decompose_goal.local_llm",
                    {"goal": goal_description, "subtasks_count": len(subtasks)}
                )
            except Exception as e:
                self.journal.log_event(
                    "planner.decompose_goal.local_llm_error",
                    {"goal": goal_description, "error": str(e)}
                )
                subtasks = []
        # If no subtasks were generated by templates or LLM, subtasks list remains empty.
    # No generic fallback subtask is created.
    # Always return a list of Goal objects, persisted if possible
        goals: List[Goal] = []
        for sub in subtasks:
            goals.append(self._persist_goal(self._make_goal(sub, created_by=decomposition_source)))
        return goals

    def _local_llm_decompose_goal(self, goal_description: str) -> List[str]:
        # Ensure necessary imports are available
        import os, json, re

        if not self.local_llm_client or not self.local_llm_client.is_available():
            self.journal.log_event(
                "planner.local_llm_decompose.unavailable",
                {
                    "goal": goal_description,
                    "client_type": type(self.local_llm_client).__name__ if self.local_llm_client else "None",
                },
            )
            return []

        # Load conversation state if supported
        if hasattr(self.local_llm_client, "load_conversation_state") and callable(
            getattr(self.local_llm_client, "load_conversation_state")
        ):
            state_path = os.getenv(
                "CONVERSATION_STATE_PATH", "./companion_memory/rwkv_conversation_history.json"
            )
            self.local_llm_client.load_conversation_state(state_path)

        # Build minimal, non-prescriptive prompt. RWKV7GGUFClient will prepend conversation history
        prompt = (
            "Break down this goal into specific, actionable tasks:\n\n"
            f"Goal: {goal_description}\n\n"
            'Please provide 3-5 tasks as a JSON array of strings, e.g.:\n'
            '["task 1 description", "task 2 description", "task 3 description"]'
        )

        try:
            response_text = self.local_llm_client.complete_text(prompt, max_tokens=300)

            self.journal.log_event(
                "planner.local_llm_decompose.response",
                {
                    "goal": goal_description,
                    "prompt_sent_to_llm_client": prompt,
                    "raw_response_from_llm": response_text,
                },
            )

            # Persist conversation state
            if hasattr(self.local_llm_client, "save_conversation_state") and callable(
                getattr(self.local_llm_client, "save_conversation_state")
            ):
                state_path = os.getenv(
                    "CONVERSATION_STATE_PATH", "./companion_memory/rwkv_conversation_history.json"
                )
                self.local_llm_client.save_conversation_state(state_path)

            # Try extracting a JSON array from the response
            json_match = re.search(
                r'\[\s*(?:"(?:[^"\\]|\\.)*"\s*(?:,\s*"(?:[^"\\]|\\.)*")*\s*)?\]',
                response_text,
                re.DOTALL,
            )
            if json_match:
                try:
                    subtasks = json.loads(json_match.group(0))
                    if isinstance(subtasks, list) and all(isinstance(t, str) for t in subtasks):
                        return [t.strip() for t in subtasks if t.strip()]
                except json.JSONDecodeError as je:
                    self.journal.log_event(
                        "planner.local_llm_decompose.json_error",
                        {"goal": goal_description, "error": str(je)},
                    )

            # Fallback: split lines
            lines = [
                l.strip(" -*\"'[]") for l in response_text.split("\n") if l.strip(" -*\"'[]")
            ]
            meaningful = [
                l
                for l in lines
                if len(l) > 10 and not any(kw in l.lower() for kw in ["task", "json", "array"])
            ]

            if meaningful:
                return meaningful[:5]
            elif lines:
                return lines[:5]

            self.journal.log_event(
                "planner.local_llm_decompose.parse_fallback_empty",
                {"goal": goal_description, "raw_response": response_text},
            )
            return []

        except Exception as e:
            self.journal.log_event(
                "planner.local_llm_decompose.unexpected_error",
                {"goal": goal_description, "error": str(e)},
            )
            return []

    def _recursive_plan_internal(self, current_goal: Goal, depth: int) -> List[Goal]:
        """Internal recursive planning logic.
        
        Args:
            current_goal: The Goal object to plan for.
            depth: Current recursion depth.

        Returns:
            A list of Goal objects representing the refined plan (leaf goals).
        """
        self.journal.log_event(
            "planner.recursive_plan.entry",
            {"goal_id": current_goal.id, "goal_description": current_goal.description, "depth": depth}
        )
        
        # Simplified root-level plan: bypass recursion and return direct decomposition tasks
        if depth == 0:
            sub_descs = self.decompose_goal(current_goal.description)
            return [Goal(id=f"{current_goal.id}.{i+1}", description=desc, parent_goal=current_goal, priority=current_goal.priority) for i, desc in enumerate(sub_descs)]

        if depth >= MAX_RECURSION_DEPTH:
            self.journal.log_event(
                "planner.recursive_plan.max_depth_reached",
                {"goal_id": current_goal.id, "goal_description": current_goal.description, "depth": depth}
            )
            # At max depth, the current goal itself is considered a leaf task
            return [current_goal]

        try:
            sub_goal_descriptions = self.decompose_goal(current_goal.description, goal_id_prefix=f"{current_goal.id}.{depth+1}")
        except Exception as e_recursive_plan:
            print(f"DEBUG CP: Error in _recursive_plan_internal for '{current_goal.description}'", file=sys.stderr); sys.stderr.flush()
            print(f"ERROR in create_plan calling _recursive_plan_internal for '{current_goal.description}': {e_recursive_plan}\n", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            sub_goal_descriptions = [] # Treat as atomic on error

        if not sub_goal_descriptions: # Atomic goal or decomposition failed to find subtasks
            is_truly_atomic = False
            reason_for_atomic = "Not atomic by default"

            if not sub_goal_descriptions:
                is_truly_atomic = True
                reason_for_atomic = "No sub-tasks from decompose_goal"
            elif len(sub_goal_descriptions) == 1 and sub_goal_descriptions[0].startswith("Review and handle:"):
                # If the only sub-task is a generic "Review and handle" for the *current* goal's description,
                # it means decompose_goal didn't find a more specific breakdown.
                if sub_goal_descriptions[0].lower() == f"Review and handle: {current_goal.description}".lower():
                    is_truly_atomic = True
                    reason_for_atomic = "Generic fallback matched current goal description"
                else:
                    # This case means the single sub-task is a "Review and handle:" for a *different* (more specific) goal description.
                    # So, the current goal is NOT atomic, it decomposed into a new, more specific review task.
                    reason_for_atomic = "Decomposed into a new specific 'Review and handle' sub-task"
            else:
                # Multiple sub-tasks, or a single sub-task that isn't a generic fallback for the current goal.
                reason_for_atomic = "Decomposed into multiple sub-tasks or a specific single sub-task"

            self.journal.log_event(
                "planner.recursive_plan.atomic_check_result",
                {"goal_id": current_goal.id, "depth": depth, "is_atomic": is_truly_atomic, "reason": reason_for_atomic, "sub_goal_desc_count": len(sub_goal_descriptions) if sub_goal_descriptions else 0}
            )

            if is_truly_atomic:
                self.journal.log_event(
                    "planner.recursive_plan.atomic_goal_final",
                    {"goal_id": current_goal.id, "goal_description": current_goal.description, "depth": depth, "reason": reason_for_atomic}
                )
                return [current_goal]
        
        refined_plan_leaf_goals: List[Goal] = []
        for i, sub_desc in enumerate(sub_goal_descriptions):
            sub_goal_id = f"{current_goal.id}.{i+1}" # Hierarchical ID
            # In a full system, TaskManager would create and return Goal objects
            # For now, we create placeholder Goal objects.
            sub_goal = Goal(id=sub_goal_id, description=sub_desc, parent_goal=current_goal, priority=current_goal.priority)
            current_goal.sub_goals.append(sub_goal)
            
            # Recursively plan for this sub-goal
            try:
                refined_plan_leaf_goals.extend(self._recursive_plan_internal(sub_goal, depth + 1))
            except Exception as e_recurse_further:
                print(f"ERROR in _recursive_plan_internal during further recursion for '{sub_goal.description}': {e_recurse_further}\n", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                # Add the current sub_goal as a leaf if recursion fails
                refined_plan_leaf_goals.append(sub_goal)
        
        self.journal.log_event(
            "planner.recursive_plan.exit",
            {"goal_id": current_goal.id, "depth": depth, "refined_leaf_goals_count": len(refined_plan_leaf_goals)}
        )
        return refined_plan_leaf_goals


import io

if __name__ == '__main__':  # pragma: no cover
    # This is a simplified self-test block with explicit file-based error logging.
    import traceback

    class MockMemoryStore:
        def find_memories_by_tags(self, user_id: str, tags: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
            return [] # Ensure it returns an empty list for this test

    try:
        # --- Test Case 4: Graph-based planning for 'develop feature' ---
        print("--- Running Simplified Planner Self-Test (TC4 Only) ---")
        goal_desc = "develop feature: new user dashboard"
        mock_journal = Mock(spec=Journal)
        mock_journal.log_event = Mock() # Ensure log_event is a mock

        # The Planner's __init__ defaults to the imported PRIMITIVE_REGISTRY.
        planner = Planner(memory_store=MockMemoryStore(), journal=mock_journal)
        
        print(f"Calling create_plan for: '{goal_desc}'")
        plan_from_graph = planner.create_plan(goal_desc)
        
        print(f"Plan from graph: {plan_from_graph}")

        # Assertions from the original test case
        expected_graph_subtasks = [
            "Clarify feature requirements and acceptance criteria.",
            "Design the feature, including UI/UX if applicable.",
            "Implement the core logic for the feature.",
            "Write unit and integration tests for the feature.",
            "Document the new feature."
        ]
        assert plan_from_graph == expected_graph_subtasks, f"TC4 FAILED: Expected {expected_graph_subtasks}, got {plan_from_graph}"
        
        print("DEBUG: About to assert on mock_journal...")
        mock_journal.log_event.assert_any_call("planner.graph.success", {'goal': goal_desc, 'graph_id': 'develop_feature_v1_contextual_output', 'sub_goals_count': len(expected_graph_subtasks)})
        print("DEBUG: mock_journal assertion passed.")
        
        print("--- Planner Self-Test PASSED ---")

    except Exception as e:
        print("--- Planner Self-Test FAILED. Writing traceback to planner_error.log ---", file=sys.stderr)
        logic_trace_content = ""
        try:
            with open("_planner_last_logic_trace.log", "r") as f_trace_read:
                logic_trace_content = f_trace_read.read()
        except FileNotFoundError:
            logic_trace_content = "INFO: _planner_last_logic_trace.log not found.\n"
        except Exception as read_trace_e:
            logic_trace_content = f"ERROR: Could not read _planner_last_logic_trace.log: {read_trace_e}\n"

        with open("planner_error.log", "w") as f_err:
            if logic_trace_content:
                f_err.write("--- Captured LogicEngine Trace (from _planner_last_logic_trace.log): ---\n")
                f_err.write(logic_trace_content)
                f_err.write("\n--- End LogicEngine Trace ---\n\n")
            f_err.write(f"Exception: {type(e).__name__}: {e}\n")
            traceback.print_exc(file=f_err)
        sys.exit(1) # Ensure the script exits with a failure code
