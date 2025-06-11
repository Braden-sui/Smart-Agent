"""
Unit tests for the core logic primitives.
"""
import unittest
from typing import Dict, Any

from simulated_mind.core.logic_primitives import (
    execute_sequence,
    execute_parallel,
    evaluate_condition,
    execute_call,
    store_value,
    evaluate_goal,
    get_primitive,
    PrimitiveResult,
    LogicContext
)

# Dummy context for testing primitives that might need it
# For now, most primitives are self-contained or use it minimally.
DUMMY_CONTEXT: LogicContext = LogicContext({
    "agent_id": "test_agent",
    "journal": None, # In real use, a Journal instance
    "memory_dao": None, # In real use, a MemoryDAO instance
    "tool_executor": None, # In real use, a tool execution service
    "planner_instance": None # In real use, the Planner instance for EVAL
})

class TestLogicPrimitives(unittest.TestCase):

    def test_get_primitive_success(self):
        self.assertIs(get_primitive("SEQ"), execute_sequence)
        self.assertIs(get_primitive("par"), execute_parallel) # Test case-insensitivity if designed

    def test_get_primitive_failure(self):
        with self.assertRaisesRegex(ValueError, "Unknown primitive: UNKNOWN. Available:.*"):
            get_primitive("UNKNOWN")

    def test_execute_sequence_valid(self):
        args = ["node1", "node2", "node3"]
        result = execute_sequence(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.value, args)
        self.assertEqual(result.next_node_id, "node1")
        self.assertIsNone(result.error_message)

    def test_execute_sequence_empty_list(self):
        result = execute_sequence(DUMMY_CONTEXT, [])
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("cannot be empty", result.error_message)

    def test_execute_sequence_invalid_args_type(self):
        result = execute_sequence(DUMMY_CONTEXT, "not_a_list")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("must be a list", result.error_message)

    def test_execute_sequence_invalid_node_id_type(self):
        result = execute_sequence(DUMMY_CONTEXT, ["node1", 2, "node3"])
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("must be a list of node IDs (strings)", result.error_message)

    def test_execute_parallel_valid(self):
        args = ["node_a", "node_b"]
        result = execute_parallel(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.value, args)
        self.assertIsNone(result.error_message)

    def test_execute_parallel_empty_list(self):
        result = execute_parallel(DUMMY_CONTEXT, [])
        self.assertFalse(result.success)
        self.assertIn("cannot be empty", result.error_message)

    def test_execute_parallel_invalid_args_type(self):
        result = execute_parallel(DUMMY_CONTEXT, {"key": "value"})
        self.assertFalse(result.success)
        self.assertIn("must be a list", result.error_message)

    def test_evaluate_condition_valid_then(self):
        # This test assumes the dummy condition_met is True
        args = {
            "condition_type": "dummy_true",
            "condition_params": {},
            "then_node_id": "then_branch_node",
            "else_node_id": "else_branch_node"
        }
        result = evaluate_condition(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.next_node_id, "then_branch_node")
        self.assertIsNone(result.error_message)

    def test_evaluate_condition_valid_else(self):
        # To test this, we'd need to modify evaluate_condition to allow
        # controlled dummy evaluation or pass a mock context.
        # For now, this illustrates the structure.
        # Let's assume a way to make condition_met False for this test.
        
        global evaluate_condition # Declare global before use
        # HACK: Temporarily modify the primitive for this test case to simulate false condition
        # This is not ideal for unit tests but shows intent. A better way is mock/dependency injection.
        original_eval_cond = evaluate_condition
        def mock_eval_cond_false(context, args_dict):
            # Simulate condition being false
            return PrimitiveResult(True, next_node_id=args_dict.get('else_node_id'))
        
        evaluate_condition = mock_eval_cond_false
        
        args = {
            "condition_type": "dummy_false",
            "condition_params": {},
            "then_node_id": "then_branch_node",
            "else_node_id": "else_branch_node"
        }
        result = evaluate_condition(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.next_node_id, "else_branch_node")
        
        evaluate_condition = original_eval_cond # Restore

    def test_evaluate_condition_missing_keys(self):
        args = {"then_node_id": "some_node"}
        result = evaluate_condition(DUMMY_CONTEXT, args)
        self.assertFalse(result.success)
        self.assertIn("Missing one or more required keys", result.error_message)

    def test_evaluate_condition_no_else_and_false(self):
        global evaluate_condition # Declare global before use
        # Similar to test_evaluate_condition_valid_else, needs controlled condition
        original_eval_cond = evaluate_condition
        def mock_eval_cond_false_no_else(context, args_dict):
            # Simulate condition being false, and no else_node_id provided
            if 'else_node_id' not in args_dict:
                 return PrimitiveResult(True, value=None, next_node_id=None)
            return PrimitiveResult(True, next_node_id=args_dict.get('else_node_id'))

        evaluate_condition = mock_eval_cond_false_no_else

        args = {
            "condition_type": "dummy_false_no_else",
            "condition_params": {},
            "then_node_id": "then_branch_node",
            # No 'else_node_id'
        }
        result = evaluate_condition(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertIsNone(result.next_node_id)
        self.assertIsNone(result.value) # As per current primitive logic
        
        evaluate_condition = original_eval_cond # Restore

    def test_execute_call_valid(self):
        args = {"tool_name": "test_tool", "tool_args": {"param": 1}}
        result = execute_call(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.value, "Output from test_tool({'param': 1})")
        self.assertIsNone(result.error_message)

    def test_execute_call_missing_keys(self):
        args = {"tool_name": "test_tool"}
        result = execute_call(DUMMY_CONTEXT, args)
        self.assertFalse(result.success)
        self.assertIn("'tool_name' and 'tool_args' are required", result.error_message)

    def test_store_value_valid(self):
        local_context = LogicContext() # Create a fresh context
        args = {"memory_key": "test_key", "value_source": "literal", "value_params": "hello"}
        result = store_value(local_context, args) # Use local_context
        self.assertTrue(result.success)
        self.assertEqual(result.value, "hello") # Assert actual stored value
        self.assertEqual(local_context.get_variable("test_key"), "hello") # Verify context was updated
        self.assertIsNone(result.error_message)

    def test_store_value_missing_keys(self):
        args = {"memory_key": "test_key"}
        result = store_value(DUMMY_CONTEXT, args)
        self.assertFalse(result.success)
        self.assertIn("'memory_key' and 'value_source' are required", result.error_message)

    def test_evaluate_goal_valid(self):
        args = {"goal_description": "test sub_goal"}
        result = evaluate_goal(DUMMY_CONTEXT, args)
        self.assertTrue(result.success)
        self.assertEqual(result.value, "Outcome of evaluating goal: test sub_goal")
        self.assertIsNone(result.error_message)

    def test_evaluate_goal_missing_keys(self):
        args = {}
        result = evaluate_goal(DUMMY_CONTEXT, args)
        self.assertFalse(result.success)
        self.assertIn("'goal_description' is required", result.error_message)

if __name__ == '__main__':
    unittest.main()
