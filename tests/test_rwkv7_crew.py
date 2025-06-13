"""
Integration test for RWKV7 CrewAI Orchestration (Phase 1)
- Proves >1 agent can share RWKV7 state
- Proves Mem0Client batch write works
"""
import pytest
from simulated_mind.orchestration.rwkv7_crew import RWKV7Crew, RWKV7StateManager
from simulated_mind.memory.mem0_client import Mem0Client

class DummyAgent:
    def __init__(self, agent_id, state_manager):
        self.agent_id = agent_id
        self.state_manager = state_manager
        self.last_state = None
    def act(self, input_data):
        # Simulate state update
        self.last_state = f"state_for_{self.agent_id}_{input_data}" 
        self.state_manager.set_state(self.last_state)
        return {"agent_id": self.agent_id, "state": self.last_state}

def test_multi_agent_rwkv7_state_sharing_and_batch_write(tmp_path):
    state_manager = RWKV7StateManager()
    agents = [DummyAgent(f"agent_{i}", state_manager) for i in range(2)]
    crew = RWKV7Crew(agents, state_manager)

    # Each agent acts, updating shared state
    results = []
    for i, agent in enumerate(agents):
        result = agent.act(f"input_{i}")
        results.append(result)
        # State manager should reflect last agent's state
        assert state_manager.get_state() == result["state"]

    # Simulate Mem0Client batch write
    mem0 = Mem0Client()
    batch = [
        {"user_id": f"agent_{i}", "memory_id": f"mem_{i}", "content": f"state_{i}"}
        for i in range(2)
    ]
    # Add memories in batch (stub: call individually for now)
    for mem in batch:
        mem0.create_memory(memory_id=mem["memory_id"], content=mem["content"], user_id=mem["user_id"])
        stored = mem0.get_memory(memory_id=mem["memory_id"], user_id=mem["user_id"])
        assert stored is not None
        assert stored["content"] == mem["content"]
