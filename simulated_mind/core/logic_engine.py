"""
Defines the LogicEngine class for executing logic graphs.
"""
from typing import Dict, Any, Optional, Callable

from .logic_graph import LogicGraph, GraphNode
# Assuming PrimitiveArgs is part of logic_primitives, if not, define or import appropriately
from .logic_primitives import PRIMITIVE_REGISTRY, PrimitiveResult, LogicContext, PrimitiveArgs # Added PrimitiveArgs
# BasePrimitive is removed as it's not defined there; we'll use Callable.

class LogicEngine:
    """Executes a logic graph using a set of registered primitives."""

    def __init__(self, primitive_registry: Optional[Dict[str, Callable[[LogicContext, Any], PrimitiveResult]]] = None): # Changed BasePrimitive to Callable
        import sys # Ensure sys is available
        print("!!! LOGIC_ENGINE: __init__ ENTERED !!!", file=sys.stderr); sys.stderr.flush()
        self.primitive_registry = primitive_registry if primitive_registry is not None else PRIMITIVE_REGISTRY
        if not self.primitive_registry:
            # This could be a warning or an error depending on desired strictness
            print("Warning: LogicEngine initialized with an empty primitive registry.", file=sys.stderr)

    def execute_graph(self, graph: LogicGraph, initial_context: Optional[LogicContext] = None) -> LogicContext:
        import sys # Ensure sys is available
        print("!!! LOGIC_ENGINE: execute_graph ENTERED !!!", file=sys.stderr); sys.stderr.flush()
        """
        Executes the given logic graph, starting from its entry node.

        Args:
            graph: The LogicGraph instance to execute.
            initial_context: An optional initial LogicContext.

        Returns:
            The final LogicContext after graph execution.
        """
        if not graph:
            raise ValueError("Cannot execute a null graph.")

        context = initial_context if initial_context is not None else LogicContext()

        # --- BEGIN CRITICAL DIAGNOSTIC LOGGING ---
        print(f"[LogicEngine] DEBUG: Executing graph. Received graph object: {graph}", file=sys.stderr)
        if graph:
            print(f"[LogicEngine] DEBUG: Graph ID: {graph.graph_id}", file=sys.stderr)
            print(f"[LogicEngine] DEBUG: Graph Entry Node ID: {graph.entry_node_id}", file=sys.stderr)
            if graph.entry_node_id and not graph.get_node(graph.entry_node_id):
                print(f"[LogicEngine] CRITICAL DEBUG: Entry node ID '{graph.entry_node_id}' is set, but get_node() returns None for it!", file=sys.stderr)
        else:
            print("[LogicEngine] CRITICAL DEBUG: Received graph object is None!", file=sys.stderr)
        # --- END CRITICAL DIAGNOSTIC LOGGING ---
        # Automatically inject the graph's ID into the context for tracing.
        context.set_variable('graph_id', graph.graph_id)
        current_node_id: Optional[str] = graph.entry_node_id

        print(f"[LogicEngine DEBUG] Starting execution. Graph ID: {graph.graph_id}, Entry Node: {current_node_id}", file=sys.stderr)
        
        # Basic cycle detection or max steps to prevent infinite loops
        max_steps = 100 # Configurable or dynamically determined
        steps_taken = 0

        while current_node_id and steps_taken < max_steps:
            steps_taken += 1
            node = graph.get_node(current_node_id)

            if not node:
                # This could be logged as an error or raise a more specific exception
                print(f"Error: Node ID '{current_node_id}' not found in graph '{graph.graph_id}'. Halting execution.", file=sys.stderr)
                context.set_error(f"Node ID '{current_node_id}' not found.")
                break
            
            print(f"[LogicEngine DEBUG] Current Node: {current_node_id}, Primitive: {node.primitive}", file=sys.stderr)

            # print(f"[LogicEngine] Executing node: {node.node_id} (Primitive: {node.primitive})", file=sys.stderr) # Debug
            context.current_node_id = node.node_id # For primitive awareness

            primitive_func = self.primitive_registry.get(node.primitive.upper()) # Case-insensitive lookup

            if not primitive_func:
                msg = f"Primitive '{node.primitive}' not found in registry for node '{node.node_id}'."
                print(f"Error: {msg}", file=sys.stderr)
                context.set_error(msg)
                break # Halt execution if a primitive is missing
            
            print(f"[LogicEngine DEBUG]   Args: {node.args}", file=sys.stderr)

            try:
                # Ensure args are passed correctly; primitives expect PrimitiveArgs (a Dict or List)
                primitive_args = node.args # Pass node.args directly. Primitives handle if it's None/empty.
                # If a primitive specifically needs an empty dict when node.args is None, it should handle it.
                # Or, ensure node.args is at least an empty dict/list from GraphNode parsing if truly required.
                # For now, assume primitives can handle None args if that's a valid state from YAML.
                if primitive_args is None:
                    # If YAML allows omitting args, and primitive expects e.g. an empty dict, default here.
                    # Most of our primitives expect args, so None might be an issue for them.
                    # Let's assume for now that YAML will provide at least an empty dict/list if args are relevant.
                    # If a primitive like NOOP takes no args, its definition should be `def noop_primitive(context: LogicContext) -> PrimitiveResult:`
                    # and the registry type would need to accommodate that, or it takes `args: Optional[Any] = None`
                    # For simplicity with current primitives, let's assume args will be a dict or list from YAML.
                    # If node.args is truly optional and can be None from YAML for some primitives:
                    if node.primitive.upper() == "NOOP": # Example: NOOP might not have args in YAML
                        primitive_args = {} # Or whatever NOOP expects for its `args` param
                    else:
                        # Default to empty dict if None, as many primitives expect a dict.
                        # This might need refinement based on how strictly YAML enforces args presence.
                        primitive_args = {}
                
                result: PrimitiveResult = primitive_func(context, primitive_args) # Changed from **primitive_args
                print(f"[LogicEngine DEBUG]   Primitive returned success: {result.success}", file=sys.stderr)
                print(f"[LogicEngine DEBUG]   Primitive returned next_node_id: {result.next_node_id}", file=sys.stderr)
                print(f"[LogicEngine DEBUG]   Context after primitive: {context.data}", file=sys.stderr)

                if not result.success:
                    # Primitives should set context.error_message themselves if they fail.
                    # The engine can log the failure here based on result.success and result.error_message.
                    error_msg_from_primitive = result.error_message if result.error_message else "Primitive execution failed without specific error message."
                    print(f"Error executing primitive '{node.primitive}' for node '{node.node_id}': {error_msg_from_primitive}", file=sys.stderr)
                    if not context.get_error(): # If primitive didn't set it, engine ensures it's set.
                        context.set_error(f"Primitive '{node.primitive}' on node '{node.node_id}' failed: {error_msg_from_primitive}")
                    break # Halt on primitive error

                # Determine next node:
                # 1. Explicit jump from primitive result (e.g., COND then/else)
                if result.next_node_id:
                    current_node_id = result.next_node_id
                # 2. Default next node specified in the graph for sequential flow
                elif node.next_node:
                    current_node_id = node.next_node
                # 3. No next node specified, end of this execution path
                else:
                    current_node_id = None 
            
            except TypeError as te:
                # Often due to mismatch in primitive signature and args in YAML
                msg = f"TypeError executing primitive '{node.primitive}' for node '{node.node_id}': {te}. Check YAML args and primitive signature."
                print(f"Error: {msg}", file=sys.stderr)
                context.set_error(msg)
                break
            except Exception as e:
                msg = f"Unexpected error executing primitive '{node.primitive}' for node '{node.node_id}': {e}"
                print(f"Error: {msg}", file=sys.stderr)
                context.set_error(msg)
                break
        
        if steps_taken >= max_steps:
            msg = f"Graph execution exceeded maximum steps ({max_steps}). Possible infinite loop in graph '{graph.graph_id}'."
            print(f"Warning: {msg}", file=sys.stderr)
            context.set_error(msg)
            context.final_status = "MAX_STEPS_EXCEEDED"
        elif not context.get_error(): # Only set to COMPLETED if no prior error
            context.final_status = "COMPLETED"

        print(f"[LogicEngine DEBUG] Final context status: {context.final_status}", file=sys.stderr)
        print(f"[LogicEngine DEBUG] Final context data: {context.data}", file=sys.stderr)
        print(f"[LogicEngine DEBUG] Final context error: {context.get_error()}", file=sys.stderr)
        # print(f"[LogicEngine] Graph execution finished. Final context status: {context.final_status}", file=sys.stderr) # Debug
        print(f"[LogicEngine DEBUG] Execution finished. Final context: {context.data}", file=sys.stderr)
        return context

# Example Usage (for testing this file directly):
if __name__ == '__main__':
    import sys
    # Need to ensure logic_graph and logic_primitives are importable
    # This might require adjusting sys.path if run directly from core/
    # For simplicity, assume they are in PYTHONPATH or this is run as part of a package

    # Create a dummy graph YAML for testing
    dummy_graph_content = """
    graph_id: engine_test_graph_01
    description: A graph to test the LogicEngine.
    entry_node: greet
    nodes:
      greet:
        primitive: CALL # Using CALL as a simple example
        args:
          tool_name: "print_message"
          tool_args: { "message": "Hello from LogicEngine Test!" }
        next_node: check_condition
      check_condition:
        primitive: COND
        args:
          condition_type: "always_true" # Dummy condition type
          condition_params: {}
          then_node_id: "success_path"
          else_node_id: "failure_path" # Should not be taken
      success_path:
        primitive: STORE
        args:
          memory_key: "test_result"
          value_source: "literal"
          value_params: "Engine Test Passed"
        next_node: end_execution # Explicit end
      failure_path:
        primitive: STORE
        args:
          memory_key: "test_result"
          value_source: "literal"
          value_params: "Engine Test Failed - Took Else Path"
      end_execution:
        primitive: NOOP # A simple primitive that does nothing
        args: {}
    """
    dummy_yaml_path = "./dummy_engine_test_graph.yaml"
    with open(dummy_yaml_path, 'w') as f:
        f.write(dummy_graph_content)

    # Dummy primitives for the test, updated to match (context, args_dict_or_list) signature
    def print_message_primitive_dummy(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
        tool_name = args.get('tool_name')
        tool_args = args.get('tool_args')
        print(f"[Dummy CALL Primitive] Args: {args}")
        if tool_name == "print_message":
            print(tool_args.get("message", "Default message") if isinstance(tool_args, dict) else "Invalid tool_args for print_message")
        return PrimitiveResult(success=True)

    def always_true_cond_primitive_dummy(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
        condition_type = args.get('condition_type')
        condition_params = args.get('condition_params')
        then_node_id = args.get('then_node_id')
        # else_node_id = args.get('else_node_id') # Not used in this dummy
        print(f"[Dummy COND Primitive] Args: {args}")
        return PrimitiveResult(success=True, next_node_id=then_node_id)
    
    def noop_primitive_dummy(context: LogicContext, args: Optional[Dict[str, Any]] = None) -> PrimitiveResult:
        # Args might be None if not specified in YAML and GraphNode allows it.
        # Or an empty dict if YAML has `args: {}` or `args:`
        print(f"[Dummy NOOP Primitive] Doing nothing. Args: {args}")
        return PrimitiveResult(success=True)

    # Use the actual STORE primitive if available and working, or a dummy one
    from .logic_primitives import store_value # Corrected from store_primitive

    test_primitive_registry = {
        "CALL": print_message_primitive_dummy,
        "COND": always_true_cond_primitive_dummy,
        "STORE": store_value, # Corrected to use store_value
        "NOOP": noop_primitive_dummy
    }

    engine = LogicEngine(primitive_registry=test_primitive_registry)
    
    try:
        print("\n--- LogicEngine Self-Test --- ")
        graph = LogicGraph.load_from_yaml(dummy_yaml_path)
        print(f"Loaded graph: {graph.graph_id}")
        
        initial_context = LogicContext()
        initial_context.set_variable("initial_var", "hello_world")
        print(f"Initial context: {initial_context.data}")

        final_context = engine.execute_graph(graph, initial_context)

        print(f"\nFinal context status: {final_context.final_status}")
        print(f"Final context data: {final_context.data}")
        print(f"Final context error: {final_context.get_error()}")

        expected_result = "Engine Test Passed"
        actual_result = final_context.get_variable("test_result")
        if actual_result == expected_result:
            print(f"\nSelf-Test PASSED! Got expected result: '{actual_result}'")
        else:
            print(f"\nSelf-Test FAILED! Expected '{expected_result}', got '{actual_result}'")

    except Exception as e:
        print(f"Error during LogicEngine self-test: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        import os
        if os.path.exists(dummy_yaml_path):
            os.remove(dummy_yaml_path)
            print(f"Cleaned up {dummy_yaml_path}.")
        print("--- End LogicEngine Self-Test ---\n")

"""
Further considerations for LogicEngine:
- Asynchronous primitive execution (for PAR primitive).
- More sophisticated context management (e.g., nested contexts for sub-graphs from EVAL).
- Handling of SEQ/PAR primitives: The engine might need to manage a stack or queue of nodes for these, 
  or these primitives themselves become mini-engines for their child nodes.
- Integration with a global tool executor for the CALL primitive.
- Richer logging/tracing capabilities.
- Error recovery strategies (e.g., retry, alternative paths).
- Dynamic graph modification during execution (advanced).
"""
