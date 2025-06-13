"""
Core Logic Primitives for the Graph-Based Planner.

Each primitive represents a fundamental operation in a reasoning graph.
They are designed to be small, stateless (or operate on provided context),
and easily composable.
"""

from typing import Any, Dict, List, Callable, Union, Optional

class LogicContext:
    """Manages the execution context for logic primitives."""
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = initial_data.copy() if initial_data else {}
        self.error_message: Optional[str] = None
        self.final_status: Optional[str] = None # e.g., COMPLETED, ERROR, MAX_STEPS_EXCEEDED
        self.current_node_id: Optional[str] = None # For primitives to know which node they are in

    def set_variable(self, key: str, value: Any) -> None:
        """Stores a variable in the context."""
        self.data[key] = value

    def get_variable(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Retrieves a variable from the context."""
        return self.data.get(key, default)

    def set_error(self, error_message: str) -> None:
        """Sets an error message in the context. Also sets final_status to ERROR."""
        self.error_message = error_message
        self.final_status = "ERROR"
        # print(f"[LogicContext] Error set: {error_message}") # Debug

    def get_error(self) -> Optional[str]:
        """Retrieves the current error message."""
        return self.error_message

    def __repr__(self):
        return f"LogicContext(data={self.data}, error='{self.error_message}', status='{self.final_status}')"

# Placeholder for the structure of a graph node's arguments
PrimitiveArgs = Union[Dict[str, Any], List[Any]]

class PrimitiveResult:
    """Standardized result object from a primitive execution."""
    def __init__(self, success: bool, value: Any = None, error_message: str = None, next_node_id: str = None):
        self.success = success
        self.value = value
        self.error_message = error_message
        self.next_node_id = next_node_id  # For primitives that control flow explicitly

    def __repr__(self):
        return f"PrimitiveResult(success={self.success}, value={self.value}, error='{self.error_message}', next_node='{self.next_node_id}')"


def execute_sequence(context: LogicContext, args: List[str]) -> PrimitiveResult:
    """
    SEQ: Executes a list of sub-graphs or child node IDs sequentially.

    Args:
        context: The execution context (e.g., access to planner, memory).
        args: A list of node IDs to be executed in order.
              The actual execution of these nodes will be handled by the LogicEngine.
              This primitive primarily signals the *intent* to execute a sequence.

    Returns:
        PrimitiveResult indicating success (if sequence logic is valid) and potentially
        the ID of the first node in the sequence to be processed by the engine.
        Actual sub-task results are aggregated by the engine.
    """
    if not isinstance(args, list) or not all(isinstance(node_id, str) for node_id in args):
        return PrimitiveResult(False, error_message="SEQ: args must be a list of node IDs (strings).")
    if not args:
        return PrimitiveResult(False, error_message="SEQ: args list cannot be empty.")
    
    # The engine will use this to start processing the sequence.
    # This primitive itself doesn't execute them, just validates and signals.
    return PrimitiveResult(True, value=args, next_node_id=args[0] if args else None)


def execute_parallel(context: LogicContext, args: List[str]) -> PrimitiveResult:
    """
    PAR: Signals execution of a list of sub-graphs or child node IDs in parallel.

    Args:
        context: The execution context.
        args: A list of node IDs to be executed concurrently.
              Actual parallel execution is managed by the LogicEngine.

    Returns:
        PrimitiveResult indicating success and the list of node IDs to parallelize.
    """
    if not isinstance(args, list) or not all(isinstance(node_id, str) for node_id in args):
        return PrimitiveResult(False, error_message="PAR: args must be a list of node IDs (strings).")
    if not args:
        return PrimitiveResult(False, error_message="PAR: args list cannot be empty.")

    return PrimitiveResult(True, value=args)


def evaluate_condition(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    COND: Evaluates a condition and determines the next node (e.g., 'then_node' or 'else_node').

    Args:
        context: The execution context, potentially containing data to evaluate.
        args: A dictionary containing:
            - 'condition_type': (e.g., 'equals', 'greater_than', 'memory_check')
            - 'condition_params': Parameters for the condition evaluation.
            - 'then_node_id': Node ID to execute if condition is true.
            - 'else_node_id': (Optional) Node ID if condition is false.

    Returns:
        PrimitiveResult with success=True and next_node_id set to then_node or else_node.
    """
    required_keys = ['condition_type', 'condition_params', 'then_node_id']
    if not all(key in args for key in required_keys):
        return PrimitiveResult(False, error_message=f"COND: Missing one or more required keys: {required_keys}")

    # Placeholder for actual condition evaluation logic
    # This would involve looking up condition_type and applying params
    condition_met = True # Dummy value

    next_node = args['then_node_id'] if condition_met else args.get('else_node_id')
    if not next_node and not condition_met and 'else_node_id' not in args:
        # If condition is false and no else_node_id, it could mean 'do nothing' or 'end branch'
        return PrimitiveResult(True, value=None, next_node_id=None) # Or specific signal
        
    return PrimitiveResult(True, next_node_id=next_node)


def execute_call(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    CALL: Executes an external action, tool, or sub-agent method.

    Args:
        context: The execution context (e.g., access to tool registry, sub-agent dispatcher).
        args: A dictionary containing:
            - 'tool_name': The name/ID of the tool or function to call.
            - 'tool_args': Arguments to pass to the tool.

    Returns:
        PrimitiveResult with the success status and output of the call.
    """
    if 'tool_name' not in args or 'tool_args' not in args:
        return PrimitiveResult(False, error_message="CALL: 'tool_name' and 'tool_args' are required.")

    # Placeholder for actual tool execution via a registry or dispatcher in context
    # tool_result = context.get('tool_executor').execute(args['tool_name'], args['tool_args'])
    tool_output = f"Output from {args['tool_name']}({args['tool_args']})" # Dummy value
    return PrimitiveResult(True, value=tool_output)


def store_value(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    STORE: Stores a value into the LogicContext.

    Args:
        context: The execution context.
        args: A dictionary containing:
            - 'memory_key': The key under which to store the value.
            - 'value_source': How to get the value (e.g., 'literal', 'from_context').
            - 'value_params': Parameters for retrieving the value.
                             If 'value_source' is 'literal', this is the value itself.

    Returns:
        PrimitiveResult indicating success of the store operation.
    """
    memory_key = args.get('memory_key')
    value_source = args.get('value_source')
    value_params = args.get('value_params') # Can be any type for literal source

    if not memory_key or not value_source:
        return PrimitiveResult(success=False, error_message="STORE: 'memory_key' and 'value_source' are required arguments.")

    value_to_store: Any = None
    resolved_source = False

    if value_source.lower() == 'literal':
        value_to_store = value_params # value_params is the literal value itself
        resolved_source = True
    elif value_source.lower() == 'from_context':
        if not isinstance(value_params, str):
            return PrimitiveResult(success=False, error_message="STORE: 'value_params' must be a string (context key) if 'value_source' is 'from_context'.")
        value_to_store = context.get_variable(value_params)
        resolved_source = True
        # Consider if value_to_store being None here is an error or valid.
    # TODO: Implement other value_sources like 'from_last_node_output' or 'from_primitive_result_value'

    if not resolved_source:
        return PrimitiveResult(success=False, error_message=f"STORE: Unknown or unsupported 'value_source': {value_source}")

    context.set_variable(memory_key, value_to_store)
    # print(f"[STORE Primitive] Stored '{value_to_store}' under key '{memory_key}' in context: {context.data}") # Debug
    return PrimitiveResult(success=True, value=value_to_store) # Return the stored value for potential chaining/logging


def evaluate_goal(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    EVAL: Triggers the planning and execution of a sub-goal.

    This is the recursive step. The LogicEngine would typically invoke the Planner
    for the sub-goal description provided.

    Args:
        context: The execution context (e.g., access to Planner).
        args: A dictionary containing:
            - 'goal_description': The description of the sub-goal to evaluate.
            - 'priority': (Optional) Priority for the sub-goal.

    Returns:
        PrimitiveResult with the outcome of the sub-goal evaluation.
        The 'value' could be complex (e.g., a list of results from sub-tasks).
    """
    if 'goal_description' not in args:
        return PrimitiveResult(False, error_message="EVAL: 'goal_description' is required.")

    # Placeholder for planner invocation
    # sub_plan_result = context.get('planner').create_and_execute_plan(args['goal_description'], ...)
    sub_goal_outcome = f"Outcome of evaluating goal: {args['goal_description']}" # Dummy value
    return PrimitiveResult(True, value=sub_goal_outcome)


# --- RWKV7 Graph-of-Thoughts Primitives ---
def rwkv7_state_init(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    RWKV7_STATE_INIT: Initialize RWKV7 state for Graph-of-Thoughts processing.
    Args:
        context: LogicContext containing 'rwkv_client' and 'base_prompt'
        args: Dict with 'context_prompt' for state initialization
    Returns:
        PrimitiveResult with base_state stored in context
    """
    rwkv_client = context.get_variable('rwkv_client')
    if not rwkv_client or not hasattr(rwkv_client, 'model'):
        return PrimitiveResult(False, error_message="RWKV7 client not available or invalid")
    context_prompt = args.get('context_prompt', 'Initialize reasoning state for complex problem solving.')
    try:
        rwkv_client.complete_text(context_prompt, max_tokens=1)
        base_state = rwkv_client.model.get_state()
        context.set_variable('base_state', base_state)
        context.set_variable('thought_states', {})
        return PrimitiveResult(True, value={'base_state_initialized': True})
    except Exception as e:
        return PrimitiveResult(False, error_message=f"State initialization failed: {str(e)}")

def rwkv7_thought_generation(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    RWKV7_THOUGHT_GENERATION: Generate diverse thoughts using RWKV7 state variations.
    Args:
        context: LogicContext with 'rwkv_client' and 'base_state'
        args: Dict with 'thought_prompts' (list of strings) and optional 'branching_factor'
    Returns:
        PrimitiveResult with thought_states stored in context
    """
    rwkv_client = context.get_variable('rwkv_client')
    base_state = context.get_variable('base_state')
    if not rwkv_client or base_state is None:
        return PrimitiveResult(False, error_message="RWKV7 client or base state not available")
    thought_prompts = args.get('thought_prompts', [])
    branching_factor = args.get('branching_factor', len(thought_prompts))
    if not thought_prompts:
        return PrimitiveResult(False, error_message="No thought prompts provided")
    thought_states = {}
    try:
        for i, prompt in enumerate(thought_prompts[:branching_factor]):
            rwkv_client.model.set_state(base_state)
            response = rwkv_client.complete_text(prompt, max_tokens=150)
            thought_state = rwkv_client.model.get_state()
            thought_states[f"thought_{i}"] = {
                'prompt': prompt,
                'response': response,
                'state': thought_state,
                'index': i
            }
        context.set_variable('thought_states', thought_states)
        return PrimitiveResult(True, value={
            'thoughts_generated': len(thought_states),
            'thought_ids': list(thought_states.keys())
        })
    except Exception as e:
        return PrimitiveResult(False, error_message=f"Thought generation failed: {str(e)}")

def rwkv7_state_scoring(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    RWKV7_STATE_SCORING: Score thoughts using RWKV7 state analysis.
    Args:
        context: LogicContext with 'thought_states'
        args: Dict with 'scoring_prompt' (string) and optional 'scoring_fn'
    Returns:
        PrimitiveResult with scored_thoughts in context
    """
    rwkv_client = context.get_variable('rwkv_client')
    thought_states = context.get_variable('thought_states')
    if not rwkv_client or not thought_states:
        return PrimitiveResult(False, error_message="RWKV7 client or thought states not available")
    scoring_prompt = args.get('scoring_prompt', 'Score this thought for relevance and quality.')
    scoring_fn = args.get('scoring_fn')  # Optional custom scoring function
    scored_thoughts = {}
    try:
        for key, data in thought_states.items():
            rwkv_client.model.set_state(data['state'])
            if scoring_fn and callable(scoring_fn):
                score = scoring_fn(data['response'])
            else:
                # Default: ask model to score its own output
                score_resp = rwkv_client.complete_text(f"{scoring_prompt}\nThought: {data['response']}\nScore (0-10):", max_tokens=5)
                try:
                    score = float(score_resp.strip().split()[0])
                except Exception:
                    score = 0.0
            scored_thoughts[key] = {**data, 'score': score}
        context.set_variable('scored_thoughts', scored_thoughts)
        return PrimitiveResult(True, value={'scored_thoughts': scored_thoughts})
    except Exception as e:
        return PrimitiveResult(False, error_message=f"State scoring failed: {str(e)}")

def rwkv7_state_merge(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    RWKV7_STATE_MERGE: Merge multiple RWKV7 states to form a consensus or best state.
    Args:
        context: LogicContext with 'scored_thoughts'
        args: Dict with 'merge_strategy' (string)
    Returns:
        PrimitiveResult with merged_state in context
    """
    rwkv_client = context.get_variable('rwkv_client')
    scored_thoughts = context.get_variable('scored_thoughts')
    if not rwkv_client or not scored_thoughts:
        return PrimitiveResult(False, error_message="RWKV7 client or scored thoughts not available")
    merge_strategy = args.get('merge_strategy', 'best_score')
    try:
        if merge_strategy == 'best_score':
            best = max(scored_thoughts.values(), key=lambda x: x.get('score', 0))
            merged_state = best['state']
        else:
            # Placeholder: other strategies could include averaging or voting
            merged_state = list(scored_thoughts.values())[0]['state']
        context.set_variable('merged_state', merged_state)
        return PrimitiveResult(True, value={'merged_state': merged_state})
    except Exception as e:
        return PrimitiveResult(False, error_message=f"State merge failed: {str(e)}")

def rwkv7_state_response(context: LogicContext, args: Dict[str, Any]) -> PrimitiveResult:
    """
    RWKV7_STATE_RESPONSE: Generate a final response using the merged RWKV7 state.
    Args:
        context: LogicContext with 'rwkv_client' and 'merged_state'
        args: Dict with 'response_prompt' (string)
    Returns:
        PrimitiveResult with final response
    """
    rwkv_client = context.get_variable('rwkv_client')
    merged_state = context.get_variable('merged_state')
    if not rwkv_client or merged_state is None:
        return PrimitiveResult(False, error_message="RWKV7 client or merged state not available")
    response_prompt = args.get('response_prompt', 'Produce the final answer based on merged reasoning.')
    try:
        rwkv_client.model.set_state(merged_state)
        response = rwkv_client.complete_text(response_prompt, max_tokens=200)
        context.set_variable('final_response', response)
        return PrimitiveResult(True, value={'final_response': response})
    except Exception as e:
        return PrimitiveResult(False, error_message=f"Response generation failed: {str(e)}")

# --- Primitive Registry --- (Initial thought, could be a class in LogicEngine)
PRIMITIVE_REGISTRY: Dict[str, Callable[[LogicContext, PrimitiveArgs], PrimitiveResult]] = {
    "SEQ": execute_sequence,
    # RWKV7 Graph-of-Thoughts Primitives
    "RWKV7_STATE_INIT": rwkv7_state_init,
    "RWKV7_THOUGHT_GENERATION": rwkv7_thought_generation,
    "RWKV7_STATE_SCORING": rwkv7_state_scoring,
    "RWKV7_STATE_MERGE": rwkv7_state_merge,
    "RWKV7_STATE_RESPONSE": rwkv7_state_response,
    "PAR": execute_parallel,
    "COND": evaluate_condition,
    "CALL": execute_call,
    "STORE": store_value,
    "EVAL": evaluate_goal,
}

def get_primitive(name: str) -> Callable[[LogicContext, PrimitiveArgs], PrimitiveResult]:
    """Retrieves a primitive function from the registry."""
    primitive = PRIMITIVE_REGISTRY.get(name.upper()) # Ensure case-insensitive lookup
    if not primitive:
        raise ValueError(f"Unknown primitive: {name}. Available: {list(PRIMITIVE_REGISTRY.keys())}")
    return primitive

"""
Further considerations for primitives:
- Asynchronous execution for PAR and potentially CALL.
- More sophisticated value resolution for STORE (e.g., JSONPath from context).
- Standardized error handling and propagation.
- Versioning of primitives if their signatures change.
- Input/Output schema validation for each primitive's args.
"""
