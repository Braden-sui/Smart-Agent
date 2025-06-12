import os
import pytest
from simulated_mind.core.local_llm_client import create_local_llm_client
from simulated_mind.core.planner import Planner
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.journal.journal import Journal

class DummyGoal:
    def __init__(self, id, description, priority=5, parent_goal=None, created_by=None):
        self.id = id
        self.description = description
        self.priority = priority
        self.parent_goal = parent_goal
        self.created_by = created_by

def test_mock_local_llm_decomposition():
    llm_client = create_local_llm_client("mock")
    planner = Planner(memory_store=None, journal=Journal.null(), goal_class=DummyGoal, local_llm_client=llm_client)
    # Should use mock LLM when no template matches
    goal = "Organize my kitchen for efficiency"
    subgoals = planner.decompose_goal(goal)
    descriptions = [g.description for g in subgoals]
    assert any("kitchen" in d.lower() for d in descriptions)
    assert len(descriptions) >= 3

def test_mock_local_llm_generic():
    llm_client = create_local_llm_client("mock")
    planner = Planner(memory_store=None, journal=Journal.null(), goal_class=DummyGoal, local_llm_client=llm_client)
    goal = "Achieve world peace"
    subgoals = planner.decompose_goal(goal)
    descriptions = [g.description for g in subgoals]
    assert any("Break" in d or "step" in d for d in descriptions)
    assert len(descriptions) >= 3

def test_transformers_local_llm_unavailable(monkeypatch):
    # Simulate transformers not installed
    monkeypatch.setitem(os.environ, "LLM_BACKEND", "transformers")
    llm_client = create_local_llm_client("transformers", model_name="nonexistent-model")
    assert not llm_client.is_available()

def test_planner_fallback_to_template():
    llm_client = create_local_llm_client("mock")
    planner = Planner(memory_store=None, journal=Journal.null(), goal_class=DummyGoal, local_llm_client=llm_client)
    # Should use template for known goal
    from simulated_mind.templates.planner_rules import TEMPLATES
    for template_key in TEMPLATES:
        subgoals = planner.decompose_goal(template_key)
        descriptions = [g.description for g in subgoals]
        for t in TEMPLATES[template_key]:
            assert t in descriptions

def test_planner_handles_llm_error(monkeypatch):
    class FailingLLM:
        def is_available(self): return True
        def complete_text(self, prompt, max_tokens=512): raise RuntimeError("fail")
    planner = Planner(memory_store=None, journal=Journal.null(), goal_class=DummyGoal, local_llm_client=FailingLLM())
    goal = "Do something impossible"
    subgoals = planner.decompose_goal(goal)
    descriptions = [g.description for g in subgoals]
    assert any("Review and handle" in d for d in descriptions)
