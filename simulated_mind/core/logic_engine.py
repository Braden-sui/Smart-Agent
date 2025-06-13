"""
Defines the LogicEngine class for executing logic graphs.
"""
from typing import Dict, Any, Optional, Callable
from ..journal.journal import Journal
from .logic_graph import LogicGraph, GraphNode
from .logic_primitives import PRIMITIVE_REGISTRY, PrimitiveResult, LogicContext, PrimitiveArgs


# --- Custom Exceptions for LogicEngine ---
class LogicEngineError(Exception):
    """Base class for all LogicEngine exceptions."""
    pass

class PrimitiveNotFoundError(LogicEngineError):
    """Raised when a primitive is not found in the registry."""
    pass

class GraphExecutionError(LogicEngineError):
    """Raised for errors during the execution of a graph node."""
    pass

class MaxStepsExceededError(LogicEngineError):
    """Raised when graph execution exceeds the maximum number of steps."""
    pass


class LogicEngine:
    """Executes a logic graph using a set of registered primitives."""

    def __init__(self, primitive_registry: Optional[Dict[str, Callable[[LogicContext, Any], PrimitiveResult]]] = None, journal: Optional[Journal] = None):
        self.primitive_registry = primitive_registry if primitive_registry is not None else PRIMITIVE_REGISTRY
        self.journal = journal or Journal.null()
        if not self.primitive_registry:
            self.journal.log_event("logic_engine:warning", {"reason": "Empty primitive registry"})

    def execute_graph(self, graph: LogicGraph, initial_context: Optional[LogicContext] = None, max_steps: int = 100) -> LogicContext:
        """
        Executes the given logic graph, starting from its entry node.

        Args:
            graph: The LogicGraph instance to execute.
            initial_context: An optional initial LogicContext.
            max_steps: The maximum number of execution steps to prevent infinite loops.

        Returns:
            The final LogicContext after graph execution.

        Raises:
            ValueError: If the graph is null.
            GraphExecutionError: For any errors during node execution.
            PrimitiveNotFoundError: If a node's primitive is not in the registry.
            MaxStepsExceededError: If execution exceeds max_steps.
        """
        if not graph or not graph.entry_node_id:
            raise ValueError("Cannot execute a null or empty graph.")

        context = initial_context or LogicContext()
        context.set_variable('graph_id', graph.graph_id)
        current_node_id: Optional[str] = graph.entry_node_id
        steps_taken = 0

        self.journal.log_event("logic_engine:start", {"graph_id": graph.graph_id, "entry_node": current_node_id})

        while current_node_id and steps_taken < max_steps:
            steps_taken += 1
            node = graph.get_node(current_node_id)

            if not node:
                raise GraphExecutionError(f"Node ID '{current_node_id}' not found in graph '{graph.graph_id}'.")

            self.journal.log_event("logic_engine:step", {"node_id": node.node_id, "primitive": node.primitive})
            context.current_node_id = node.node_id

            primitive_func = self.primitive_registry.get(node.primitive.upper())
            if not primitive_func:
                raise PrimitiveNotFoundError(f"Primitive '{node.primitive}' not found for node '{node.node_id}'.")

            try:
                primitive_args = node.args or {}
                result: PrimitiveResult = primitive_func(context, primitive_args)

                if not result.success:
                    error_msg = result.error_message or "Primitive execution failed without a specific message."
                    raise GraphExecutionError(f"Primitive '{node.primitive}' on node '{node.node_id}' failed: {error_msg}")

                current_node_id = result.next_node_id or node.next_node

            except Exception as e:
                if isinstance(e, LogicEngineError):
                    raise  # Re-raise our own exceptions
                # Wrap external exceptions
                raise GraphExecutionError(f"Unexpected error at node '{node.node_id}': {e}") from e

        if steps_taken >= max_steps:
            raise MaxStepsExceededError(f"Graph '{graph.graph_id}' exceeded {max_steps} steps.")

        self.journal.log_event("logic_engine:complete", {"graph_id": graph.graph_id, "final_node": current_node_id, "steps": steps_taken})
        context.final_status = "COMPLETED"
        return context


