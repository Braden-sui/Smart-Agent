"""
RWKV7 Graph-of-Thoughts Engine for advanced reasoning workflows.
Extends the existing LogicEngine architecture with RWKV7 state-based thought processing.
"""

from typing import Dict, Any, Optional
import time

from ..core.logic_engine import LogicEngine, LogicContext
from ..core.logic_graph import LogicGraph
from ..core.logic_primitives import PRIMITIVE_REGISTRY
from ..journal.journal import Journal


class RWKV7GraphOfThoughtsEngine(LogicEngine):
    """
    RWKV7-optimized Graph-of-Thoughts engine that leverages state persistence
    for advanced multi-step reasoning workflows.
    """
    
    def __init__(self, rwkv_client, primitive_registry: Optional[Dict] = None, journal: Optional[Journal] = None):
        """
        Initialize RWKV7 Graph-of-Thoughts engine.
        Args:
            rwkv_client: RWKV7GGUFClient instance for state-based processing
            primitive_registry: Optional custom primitive registry
            journal: Optional journal for logging
        """
        super().__init__(primitive_registry or PRIMITIVE_REGISTRY)
        self.rwkv_client = rwkv_client
        self.journal = journal or Journal.null()
        if not hasattr(rwkv_client, 'model') or not hasattr(rwkv_client.model, 'get_state'):
            raise ValueError("RWKV client must support state operations (get_state/set_state)")

    def execute_graph_of_thoughts(self, graph: LogicGraph, initial_context: Optional[LogicContext] = None, base_prompt: str = "") -> LogicContext:
        """
        Execute a Graph-of-Thoughts workflow using RWKV7 state-based processing.
        Args:
            graph: LogicGraph containing GoT workflow definition
            initial_context: Optional initial context
            base_prompt: Base reasoning prompt for state initialization
        Returns:
            LogicContext with final reasoning results
        """
        start_time = time.time()
        context = initial_context or LogicContext()
        context.set_variable('rwkv_client', self.rwkv_client)
        context.set_variable('base_prompt', base_prompt)
        context.set_variable('got_start_time', start_time)
        self.journal.log_event("got.execution_start", {
            "graph_id": graph.graph_id,
            "base_prompt_length": len(base_prompt),
            "initial_context_vars": len(context.data)
        })
        # Preserve original RWKV7 model state to avoid cross-contamination
        original_state = None
        if hasattr(self.rwkv_client, "get_state") and callable(self.rwkv_client.get_state):
            try:
                original_state = self.rwkv_client.get_state()
            except Exception:
                # Non-fatal; journal but proceed
                self.journal.log_event("got.state_preserve_error", {
                    "graph_id": graph.graph_id,
                    "error": "Could not snapshot initial state"
                })
        try:
            final_context = self.execute_graph(graph, context)
            execution_time = time.time() - start_time
            got_results = self._extract_got_results(final_context)
            self.journal.log_event("got.execution_complete", {
                "graph_id": graph.graph_id,
                "execution_time": execution_time,
                "thoughts_generated": got_results.get('thoughts_count', 0),
                "final_response_length": len(got_results.get('final_response', '')),
                "success": not final_context.get_error()
            })
            return final_context
        except Exception as e:
            self.journal.log_event("got.execution_error", {
                "graph_id": graph.graph_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            error_context = context
            error_context.set_error(f"GoT execution failed: {str(e)}")
            return error_context
        finally:
            # Restore original state to ensure caller environment stability
            if original_state is not None:
                try:
                    self.rwkv_client.set_state(original_state)
                    self.journal.log_event("got.state_restored", {"graph_id": graph.graph_id})
                except Exception as e_restore:
                    self.journal.log_event("got.state_restore_error", {
                        "graph_id": graph.graph_id,
                        "error": str(e_restore)
                    })

    def _extract_got_results(self, context: LogicContext) -> Dict[str, Any]:
        thought_states = context.get_variable('thought_states', {})
        return {
            'thoughts_count': len(thought_states),
            'final_response': context.get_variable('final_response', ''),
            'synthesis_response': context.get_variable('synthesis_response', ''),
            'thought_states': thought_states,
            'thought_scores': self._get_thought_scores(context),
            'execution_stats': {
                'base_state_initialized': context.get_variable('base_state') is not None,
                'thoughts_merged': context.get_variable('merged_state') is not None,
                'feedback_applied': context.get_variable('feedback_applied', False)
            }
        }

    def _get_thought_scores(self, context: LogicContext) -> Dict[str, float]:
        scored_thoughts = context.get_variable('scored_thoughts', {})
        return {
            thought_id: data.get('score', 0.0)
            for thought_id, data in scored_thoughts.items()
        }

    def create_reasoning_context(self, problem_statement: str, reasoning_type: str = "analytical") -> LogicContext:
        context = LogicContext()
        context.set_variable('rwkv_client', self.rwkv_client)
        context.set_variable('problem_statement', problem_statement)
        context.set_variable('reasoning_type', reasoning_type)
        if reasoning_type == "creative":
            context.set_variable('branching_factor', 5)
            context.set_variable('feedback_loops', True)
        elif reasoning_type == "analytical":
            context.set_variable('branching_factor', 3)
            context.set_variable('feedback_loops', True)
        else:
            context.set_variable('branching_factor', 4)
            context.set_variable('feedback_loops', False)
        return context

def create_got_engine(rwkv_client, journal: Optional[Journal] = None) -> RWKV7GraphOfThoughtsEngine:
    """
    Factory function to create a configured RWKV7 Graph-of-Thoughts engine.
    Args:
        rwkv_client: RWKV7GGUFClient instance
        journal: Optional journal for logging
    Returns:
        Configured RWKV7GraphOfThoughtsEngine
    """
    return RWKV7GraphOfThoughtsEngine(rwkv_client, journal=journal)
