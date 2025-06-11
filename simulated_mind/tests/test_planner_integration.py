"""
Integration tests for the Planner -> SubAgent -> MetaAgent -> MemoryDAO flow.
"""

import os
import sys
import unittest

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unittest.mock import Mock

# Core component imports
from simulated_mind.core.meta_agent import MetaAgent
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.safety.guard import SafetyGuard
from simulated_mind.logging.journal import Journal


class TestPlannerIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a common environment for each test."""
        self.mock_journal = Mock(spec=Journal)
        self.mock_safety_guard = Mock(spec=SafetyGuard)
        # Use the real MemoryDAO, which will use its in-memory fallback by default
        self.memory_dao = MemoryDAO(journal=self.mock_journal)

        # Instantiate the top-level agent
        self.meta_agent = MetaAgent(
            memory_dao=self.memory_dao,
            safety_guard=self.mock_safety_guard,
            journal=self.mock_journal,
        )

    def test_sub_agent_uses_planner_to_decompose_goal(self):
        """Test that a SubAgent correctly uses its Planner to decompose a goal and stores it in memory."""
        # Arrange: Spawn a sub-agent and get a handle to it
        sub_agent_id = self.meta_agent.spawn_sub_agent()
        sub_agent = self.meta_agent.sub_agents[sub_agent_id]

        # Set a goal for the sub-agent to process. The 'decide' method uses '_last_event'.
        complex_goal = "develop feature: user authentication"
        sub_agent._last_event = complex_goal

        # Act: Trigger the sub-agent's decision-making process, which invokes the planner
        action = sub_agent.decide()

        # Assert: Check the outcome of the action and the state of memory
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "planned")
        self.assertIsInstance(action.payload, list)
        self.assertEqual(len(action.payload), 5)  # Based on the 'develop feature' template

        # Extract descriptions from the returned Goal objects
        plan_descriptions = [goal.description for goal in action.payload]
        self.assertIn('Implement the core logic for the feature.', plan_descriptions)

        # Verify that the plan was stored in the MemoryDAO
        stored_tasks_record = self.memory_dao.retrieve_memory(user_id=sub_agent_id, memory_id="agent_tasks")
        self.assertIsNotNone(stored_tasks_record)

        # The content stored should be the list of Goal objects
        stored_tasks = stored_tasks_record.get("content")
        self.assertIsNotNone(stored_tasks)
        self.assertIsInstance(stored_tasks, list)
        self.assertEqual(len(stored_tasks), 5)

        # Extract descriptions from the stored Goal objects
        stored_descriptions = [goal.description for goal in stored_tasks]
        self.assertIn('Implement the core logic for the feature.', stored_descriptions)

        # Verify that the journal was called with the planning event
        self.mock_journal.log_event.assert_any_call(
            "sub_agent.plan",
            {"agent_id": sub_agent_id, "subtasks": action.payload}
        )


if __name__ == '__main__':
    # This allows running the test directly from the command line
    # Note: You might need to run this with `python -m unittest discover` from the project root
    # for the relative imports to work correctly.
    unittest.main()
