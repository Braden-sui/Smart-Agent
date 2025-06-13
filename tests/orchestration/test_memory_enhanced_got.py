"""Integration tests for MemoryEnhancedGoT with mocked Mem0 client."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from simulated_mind.memory.cognitive_dao import RWKV7CognitiveMemoryDAO
from simulated_mind.memory.usage_tracker import UsageTracker
from simulated_mind.orchestration.graph_memory import MemoryEnhancedGoT
from simulated_mind.orchestration.thought_node import ThoughtType


class DummyLLM:
    def __init__(self):
        self.total_tokens_used = 0

    def complete_text(self, prompt: str, max_tokens: int = 256):
        self.total_tokens_used += len(prompt.split()) + max_tokens
        # Minimal JSON list if decomposition prompt detected
        if "Decompose" in prompt:
            import json

            return json.dumps(
                [
                    {"id": str(uuid.uuid4()), "content": "Part A", "confidence": 0.7},
                    {"id": str(uuid.uuid4()), "content": "Part B", "confidence": 0.6},
                ]
            )
        if "Validate" in prompt:
            return '{"confidence": 0.9}'
        return "Mock response"

    def get_state(self):
        return {}

    def set_state(self, state):  # noqa: D401 – mock setter
        pass


@pytest.fixture
def mock_mem0():
    client = MagicMock()
    client.search = MagicMock(return_value=[{"snippet": "prior knowledge"}])
    client.store = MagicMock(return_value=None)
    return client


@pytest.fixture
def cognitive_dao(mock_mem0):
    llm = DummyLLM()
    tracker = UsageTracker()
    return RWKV7CognitiveMemoryDAO(llm_client=llm, mem0_client=mock_mem0, usage_tracker=tracker)


def test_memory_integration(cognitive_dao):
    llm = DummyLLM()
    got = MemoryEnhancedGoT(llm_client=llm, memory_dao=cognitive_dao)

    result = got.reason("What is the meaning of life?")

    # Assert memory searched and stored via the DAO
    cognitive_dao.mem0_client.search.assert_called_once()
    cognitive_dao.mem0_client.store.assert_called_once()

    assert "answer" in result
    assert result["answer"]["confidence"] > 0.0
    assert any("memories" in n for n in result["reasoning_paths"])
