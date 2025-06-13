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
# Mock-specific tests (test_mock_local_llm_decomposition, test_mock_local_llm_generic) removed.

def test_transformers_local_llm_unavailable(monkeypatch):
    # Simulate transformers not installed
    monkeypatch.setitem(os.environ, "LLM_BACKEND", "transformers")
    llm_client = create_local_llm_client("transformers", model_name="nonexistent-model")
    assert not llm_client.is_available()

def test_planner_fallback_to_template():
    # This test focuses on template matching, LLM client not essential for this part.
    llm_client = None
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
    assert not descriptions  # Expect an empty list


def test_planner_uses_real_local_llm(monkeypatch):
    """Tests that the planner uses a real, configured local LLM when no template matches."""
    print(f"\nDEBUG_TEST: RWKV_MODEL_PATH from os.environ: '{os.environ.get('RWKV_MODEL_PATH')}'")
    model_path = os.environ.get("RWKV_MODEL_PATH")
    if not model_path:
        pytest.skip("RWKV_MODEL_PATH environment variable not set. Skipping real LLM test.")

    # Ensure we are attempting to use the rwkv7-gguf backend
    monkeypatch.setitem(os.environ, "LLM_BACKEND", "rwkv7-gguf")
    
    llm_client = None
    try:
        llm_client = create_local_llm_client(
            backend="rwkv7-gguf", 
            model_path=model_path, 
            context_size=1024 # Small context for test efficiency
        )
    except Exception as e:
        pytest.fail(f"Failed to create RWKV7GGUFClient during setup: {e}")

    if not llm_client or not llm_client.is_available():
        # This might happen if the model_path is invalid or model fails to load despite path being set.
        # The RWKV7GGUFClient's _load_model method prints detailed errors.
        pytest.fail("RWKV7GGUFClient is not available. Check model path and integrity. See console output for load errors.")

    planner = Planner(
        memory_store=None, 
        journal=Journal.null(), 
        goal_class=DummyGoal, 
        local_llm_client=llm_client
    )

    # A goal unlikely to match any template
    goal_description = "Formulate a novel theory about the migratory patterns of cosmic squirrels."
    
    subgoals = planner.decompose_goal(goal_description)
    
    # If the LLM was successfully used, we should get some decomposition
    assert subgoals, "Planner should have decomposed the goal using the local LLM, but returned an empty list."
    
    # Further check: ensure it wasn't a template fallback (though unlikely for this goal)
    # and that the source indicates local_llm
    for goal_obj in subgoals:
        assert goal_obj.created_by == "local_llm", f"Subgoal '{goal_obj.description}' was not created by 'local_llm', but by '{goal_obj.created_by}'."
