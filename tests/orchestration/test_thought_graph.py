"""Unit tests for ThoughtNode, GraphOfThoughts, AsyncGraphOfThoughts.
Run with: pytest -q tests/orchestration/test_thought_graph.py
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from simulated_mind.orchestration.thought_node import ThoughtNode, ThoughtStatus, ThoughtType
from simulated_mind.orchestration.graph_core import GraphOfThoughts
from simulated_mind.orchestration.graph_reasoning import AsyncGraphOfThoughts


class DummyLLM:
    """Very small mock client with deterministic output and token counter."""

    def __init__(self):
        self.calls: int = 0
        self.total_tokens_used: int = 0

    def complete_text(self, prompt: str, max_tokens: int = 256):  # noqa: D401
        self.calls += 1
        # Naïve token estimator
        self.total_tokens_used += len(prompt.split()) + max_tokens
        # Return JSON list for decomposition prompt, else a simple echo.
        if "Decompose" in prompt:
            items = [
                {"id": str(uuid.uuid4()), "content": f"Sub-problem {i}", "confidence": 0.6}
                for i in range(2)
            ]
            import json

            return json.dumps(items)
        if "Validate" in prompt:
            return '{"confidence": 0.8}'
        return f"Response to: {prompt[:20]}"

    # State ops -----------------------------------------------------------
    def get_state(self):
        return {"dummy": True}

    def set_state(self, state):  # noqa: D401 – mock setter
        assert "dummy" in state


# -------------------------------------------------------------------------
# ThoughtNode tests
# -------------------------------------------------------------------------

def test_thought_node_validation():
    node = ThoughtNode(
        id=str(uuid.uuid4()),
        content="Test content",
        thought_type=ThoughtType.DECOMPOSITION,
    )
    assert node.validate_integrity() is True

    with pytest.raises(ValueError):
        ThoughtNode(id="not-a-uuid", content="bad", thought_type=ThoughtType.OTHER)


# -------------------------------------------------------------------------
# GraphOfThoughts structural tests
# -------------------------------------------------------------------------

def test_graph_add_and_cycle_detection():
    client = DummyLLM()
    graph = GraphOfThoughts(client)

    n1 = ThoughtNode(id=str(uuid.uuid4()), content="root", thought_type=ThoughtType.OTHER)
    n2 = ThoughtNode(id=str(uuid.uuid4()), content="child", thought_type=ThoughtType.OTHER)

    assert graph.add_node(n1) is True
    assert graph.add_node(n2) is True

    assert graph.add_edge(n1.id, n2.id) is True
    # Cycle prevented
    with pytest.raises(Exception):
        graph.add_edge(n2.id, n1.id)

    assert graph.detect_cycles() == []

    paths = graph.find_paths(n1.id, n2.id)
    assert paths and paths[0] == [n1.id, n2.id]


# -------------------------------------------------------------------------
# AsyncGraphOfThoughts reasoning smoke test (happy path)
# -------------------------------------------------------------------------

def test_async_reasoning_cycle():
    dummy = DummyLLM()
    got = AsyncGraphOfThoughts(dummy)

    result = asyncio.run(got.reason("Simple test problem"))
    assert "answer" in result
    assert result["answer"]["confidence"] > 0.0
    assert result["metadata"]["total_nodes"] >= 1
