import os
import types

from simulated_mind.entry import make_planner


def test_got_end_to_end_with_develop_feature(monkeypatch):
    """Ensure GoT pipeline returns decomposed goals when enabled."""
    # Force GoT mode
    os.environ["PLANNER_MODE"] = "enabled"

    # Monkey-patch local LLM availability check to avoid heavy model load in CI
    from simulated_mind.core.local_llm_client import LocalLLMClient  # type: ignore

    def _is_available_stub(self):  # noqa: D401
        return True

    monkeypatch.setattr(LocalLLMClient, "is_available", _is_available_stub, raising=False)

    planner = make_planner()

    # Monkey-patch _create_plan_from_got_graph to avoid invoking heavy RWKV logic
    def _fake_got(self, goal_description, *_, **__):  # noqa: D401
        return [
            self._make_goal("Clarify feature requirements and acceptance criteria.", created_by="got_reasoning"),
            self._make_goal("Design the feature, including UI/UX if applicable.", created_by="got_reasoning"),
            self._make_goal("Implement the core logic for the feature.", created_by="got_reasoning"),
            self._make_goal("Write unit and integration tests for the feature.", created_by="got_reasoning"),
            self._make_goal("Document the new feature.", created_by="got_reasoning"),
        ]

    monkeypatch.setattr(planner, "_create_plan_from_got_graph", types.MethodType(_fake_got, planner))

    goals = planner.create_plan("develop feature: user dashboard")

    # At least 5 decomposed goals expected
    assert len(goals) >= 5
    assert any("requirements" in g.description.lower() for g in goals)
    assert all(getattr(g, "created_by", None) == "got_reasoning" for g in goals)
