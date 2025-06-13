"""Simple performance benchmark ensuring GoT reasoning cycle stays within SLA.

The test is intentionally lightweight to run in CI while still providing a
regression guard for pathological slow-downs. It uses the same DummyLLM mock
implementation utilised by other unit tests.
"""
from __future__ import annotations

import time
import uuid

from simulated_mind.orchestration.graph_reasoning import SynchronousGraphOfThoughts


class DummyLLM:
    def complete_text(self, prompt: str, max_tokens: int = 256):
        # Extremely fast deterministic output
        if "Decompose" in prompt:
            import json

            return json.dumps(
                [
                    {"id": str(uuid.uuid4()), "content": "part", "confidence": 0.6},
                    {"id": str(uuid.uuid4()), "content": "part", "confidence": 0.6},
                ]
            )
        if "Validate" in prompt:
            return '{"confidence": 0.9}'
        return "fast-response"

    def get_state(self):
        return {}

    def set_state(self, state):  # noqa: D401
        pass

    total_tokens_used = 0


def test_reasoning_latency():
    llm = DummyLLM()
    got = SynchronousGraphOfThoughts(llm)

    start = time.perf_counter()
    got.reason("benchmark problem")
    elapsed = time.perf_counter() - start

    # Fails if latency exceeds 0.5 seconds (generous for CI)
    assert elapsed < 0.5, f"GoT reasoning latency regression: {elapsed:.3f}s > 0.5s"
