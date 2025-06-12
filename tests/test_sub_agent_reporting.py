"""
Test upward reporting from SubAgent to CEO/global workspace via MemoryDAO.report_to_ceo.
"""
import pytest
from unittest.mock import MagicMock
from simulated_mind.core.sub_agent import SubAgent
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.journal.journal import Journal

class DummyPlanner:
    def create_plan(self, goal_desc):
        return [f"step1 for {goal_desc}", f"step2 for {goal_desc}"]

class DummyAction:
    def __init__(self, kind, payload=None):
        self.kind = kind
        self.payload = payload

def test_subagent_reports_to_ceo(monkeypatch):
    # Setup
    memory = MagicMock(spec=MemoryDAO)
    journal = Journal.null()
    agent_id = "subagent-123"
    ceo_user_id = "ceo-xyz"
    subagent = SubAgent(agent_id=agent_id, memory=memory, journal=journal)
    subagent.ceo_user_id = ceo_user_id
    subagent.planner = DummyPlanner()
    subagent._last_event = "Test Goal"
    # Patch Action to DummyAction to avoid dependency
    monkeypatch.setattr("simulated_mind.core.sub_agent.Action", DummyAction)

    # Call decide (should trigger upward report)
    memory.retrieve_memory.return_value = None  # Simulate no tasks
    subagent.decide()

    # Check upward report call
    assert memory.report_to_ceo.called, "SubAgent did not report to CEO"
    args, kwargs = memory.report_to_ceo.call_args
    assert kwargs["ceo_user_id"] == ceo_user_id
    assert kwargs["subagent_id"] == agent_id
    assert "subtasks" in kwargs["knowledge"]
    assert kwargs["report_type"] == "planning"
    assert kwargs["metadata"]["source"] == "SubAgent.decide"
    assert kwargs["task_id"] == "Test Goal"
    # Also ensure tasks are stored locally
    assert memory.store_memory.called, "SubAgent did not store tasks locally"
