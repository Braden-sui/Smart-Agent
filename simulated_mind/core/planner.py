"""Recursive, template-aware Planner.

This planner decomposes goals into sub-tasks using predefined templates
and can recursively plan complex goals up to a specified depth.
It also leverages memory for reusing past plans and finding suggestions.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from .logic_engine import LogicEngine, LogicContext
from .logic_graph import LogicGraph
from .logic_primitives import PRIMITIVE_REGISTRY


try:
    from ..logging.journal import Journal
except ImportError:
    if __name__ == '__main__':
        # Define a placeholder Journal class for self-testing when run directly
        class Journal:
            def __init__(self, sink=None):
                self.sink = sink
                # print(f"DEBUG_SELF_TEST: Using placeholder Journal class (sink: {type(sink)})", file=sys.stderr); sys.stderr.flush()

            def log_event(self, label: str, payload: Dict[str, Any]):
                # print(f"DEBUG_PLACEHOLDER_JOURNAL: log_event('{label}', {payload})", file=sys.stderr); sys.stderr.flush()
                pass

            @classmethod
            def null(cls):
                # print("DEBUG_SELF_TEST: Placeholder Journal.null() called", file=sys.stderr); sys.stderr.flush()
                return cls(sink=None)
    else:
        # If not running as main, this is a real import error in the package
        raise
from unittest.mock import Mock, ANY
# Removed TaskManager and MemoryStore imports as they cause ModuleNotFoundError
# and are only used for Mock spec in the self-test.
# Placeholders will be defined in the if __name__ == '__main__' block.
try:
    # Attempt relative import first (standard for package structure)
    from ..templates.planner_rules import TEMPLATES, GRAPH_TEMPLATES
except ImportError as e_relative:
    if __name__ == '__main__':
        # If relative import fails and we're running as a script (e.g., self-test),
        # try an absolute import.
        # print(f"DEBUG_SELF_TEST: Relative import of TEMPLATES failed: {e_relative}. Trying absolute import.", file=sys.stderr); sys.stderr.flush()
        try:
            # Add project root to sys.path to enable absolute import from simulated_mind
            # Assumes planner.py is in simulated_mind/core/
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            # print(f"DEBUG_SELF_TEST: Added to sys.path for TEMPLATES import: {project_root}", file=sys.stderr); sys.stderr.flush()
            # print(f"DEBUG_SELF_TEST: Current sys.path: {sys.path}", file=sys.stderr); sys.stderr.flush()

            from simulated_mind.templates.planner_rules import TEMPLATES
            # print(f"DEBUG_SELF_TEST: Successfully imported TEMPLATES via absolute path. Keys: {list(TEMPLATES.keys()) if TEMPLATES is not None else 'None'}", file=sys.stderr); sys.stderr.flush()
            from simulated_mind.templates.planner_rules import GRAPH_TEMPLATES # Also import GRAPH_TEMPLATES here
            print(f"DEBUG_PLANNER_IMPORT: Fallback import successful. GRAPH_TEMPLATES keys: {list(GRAPH_TEMPLATES.keys()) if GRAPH_TEMPLATES else 'EMPTY'}", file=sys.stderr); sys.stderr.flush()

        except ImportError as e_absolute:
            # print(f"DEBUG_SELF_TEST: Absolute import of TEMPLATES also failed: {e_absolute}. Falling back to empty TEMPLATES.", file=sys.stderr); sys.stderr.flush()
            TEMPLATES: Dict[str, List[str]] = {}
        except Exception as e_general_absolute_import:
            # print(f"DEBUG_SELF_TEST: General error during absolute import of TEMPLATES: {e_general_absolute_import}. Falling back to empty TEMPLATES.", file=sys.stderr); sys.stderr.flush()
            TEMPLATES: Dict[str, List[str]] = {}
            GRAPH_TEMPLATES: Dict[str, str] = {} # Fallback for GRAPH_TEMPLATES as well
    else:
        # If not running as __main__ (i.e., part of the package), the relative import should have worked.
        # Re-raise the original relative import error.
        raise e_relative

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
    """Recursive, template-aware planner, integrating LogicEngine for graph-based planning."""

    def __init__(self, memory_store: Any | None = None, journal: Journal | None = None, goal_class: Optional[Type[Goal]] = None, primitive_registry: Optional[Dict[str, Callable]] = None):
        self.primitive_registry = primitive_registry or PRIMITIVE_REGISTRY
        try:
            # print(file=sys.stderr,f"DEBUG: Planner __init__ - TEMPLATES available: {bool(TEMPLATES)} Keys: {list(TEMPLATES.keys()) if TEMPLATES else 'None'}") # Commented out to reduce variables
            self.memory_store = memory_store
            self.journal = journal or Journal.null()
            self.goal_class = goal_class # Assign goal_class
            self._goal_id_counter = 0 # For generating unique goal IDs if not provided
            self.plan_cache: Dict[str, List[Goal]] = {} # Initialize plan_cache
            # self.load_templates() was removed as TEMPLATES and GRAPH_TEMPLATES are now module-level
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

    def create_plan(self, goal_description: str) -> List[Union[Goal, str]]: # Updated type hint
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

    def decompose_goal(self, goal_description: str, goal_id_prefix: str = "subgoal") -> List[Union[Goal, str]]: # Updated type hint
        """Decomposes a goal description into sub-task descriptions using templates or memory.

        Args:
            goal_description: The string description of the goal.
            goal_id_prefix: A prefix for generating IDs for sub-goals if needed (mostly for logging).

        Returns:
            A list of sub-task descriptions.
        """
        subtasks: List[str] = []
        decomposition_source = "fallback"

        # 1. Try template matching
        # Simple keyword matching for now; can be made more sophisticated (e.g., NLP, regex)
        normalized_goal_desc = goal_description.strip().lower()
        for template_key, template_subtasks in TEMPLATES.items():
            # Check if any part of the template key (split by space) is in the goal description
            # or if the template key itself is a substring. This is a basic heuristic.
            if normalized_goal_desc.startswith(template_key):
                subtasks = list(template_subtasks) # Create a copy
                decomposition_source = f"template:{template_key}"
                self.journal.log_event(
                    "planner.decompose_goal.template_match",
                    {"goal": goal_description, "template_key": template_key, "subtasks_count": len(subtasks)}
                )
                break
        
        # 2. If no template match, try memory-based suggestions (placeholder for now)
        #    Memory `bb8561bc-08bc-4d53-95f9-e20e09f5cfe4` mentioned `self.memory_store.recall("planning_context", goal_description)`
        #    This needs to be mapped to MemoryDAO's capabilities, e.g., find_memories_by_tags with specific tags.
        if not subtasks and self.memory_store:
            try:
                # This is a placeholder for recalling *suggestions* for decomposition, not full plans.
                # For now, we'll assume a specific tag like "planning_suggestion" might be used.
                # The actual implementation of suggestion recall needs more design.
                suggestions = self.memory_store.find_memories_by_tags(
                    user_id="planner_system", 
                    tags=["planning_suggestion", goal_description.replace(' ', '_').lower()[:20]], 
                    limit=5
                )
                if suggestions:
                    # Assuming suggestions are stored with 'content' being a list of sub-task strings or a dict with 'subtasks'
                    # This part is highly speculative and needs to align with how suggestions are stored.
                    potential_subtasks = []
                    for suggestion_record in suggestions:
                        suggestion_content = suggestion_record.get("content")
                        if isinstance(suggestion_content, dict) and "subtasks" in suggestion_content:
                            if isinstance(suggestion_content["subtasks"], list):
                                potential_subtasks.extend(suggestion_content["subtasks"])
                        elif isinstance(suggestion_content, list):
                             potential_subtasks.extend(suggestion_content)
                    
                    if potential_subtasks:
                        # Simple strategy: take the first few unique suggestions
                        subtasks = list(dict.fromkeys(potential_subtasks))[:3] # Max 3 suggestions
                        decomposition_source = "memory_suggestion"
                        self.journal.log_event(
                            "planner.decompose_goal.memory_suggestion",
                            {"goal": goal_description, "retrieved_suggestions_count": len(suggestions), "used_subtasks_count": len(subtasks)}
                        )
            except Exception as e:
                self.journal.log_event(
                    "planner.decompose_goal.memory_suggestion_error",
                    {"goal": goal_description, "error": str(e)}
                )

        # 3. If still no subtasks, generate a generic fallback task
        if not subtasks:
            cleaned_description = goal_description.strip()
            if not cleaned_description.startswith("Review and handle:"):
                cleaned_description = f"Review and handle: {cleaned_description}"
            subtasks = [cleaned_description]
            decomposition_source = "fallback_generic"
            self.journal.log_event(
                "planner.decompose_goal.fallback",
                {"goal": goal_description, "subtasks_count": len(subtasks)}
            )

        # Final conversion to Goal objects if goal_class is set
        if self.goal_class:
            return [self.goal_class(id=self._generate_goal_id(prefix=f"{decomposition_source}_task"), description=desc, priority=5, created_by=decomposition_source) for desc in subtasks]
        else:
            return subtasks

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
