"""Simple smoke test for MetaAgent + SubAgent plumbing."""
from simulated_mind.core.meta_agent import MetaAgent
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.safety.guard import SafetyGuard


def test_spawn_and_retire():
    memory = MemoryDAO()
    guard = SafetyGuard(memory)
    meta = MetaAgent(memory_dao=memory, safety_guard=guard)

    aid = meta.spawn_sub_agent()
    assert aid in meta.sub_agents

    # run one cycle with a goal
    meta.sub_agents[aid].process("learn self-modification")

    meta.retire_agent(aid)
    assert aid not in meta.sub_agents
