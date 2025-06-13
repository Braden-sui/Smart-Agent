"""Integration tests for MemoryEnhancedGoT with mocked Mem0 client."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def test_memory_integration(mock_mem0):
    llm = DummyLLM()
    got = MemoryEnhancedGoT(llm_client=llm, memory_client=mock_mem0)

    result = asyncio.run(got.reason("What is the meaning of life?"))

    # Assert memory searched and stored
    mock_mem0.search.assert_called_once()
    mock_mem0.store.assert_called_once()

    assert "answer" in result
    assert result["answer"]["confidence"] > 0.0
    assert any("memories" in n for n in result["reasoning_paths"])
