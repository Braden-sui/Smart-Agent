"""
Entrypoint for initializing and running the integrated agent system:
- Loads MEM0_API_KEY from environment (or .env if present)
- Wires MemoryDAO, TaskManager, MetaAgent, SubAgent, Planner
- Supports both live mem0 and in-memory fallback
- Provides a CLI for simple user command input and system response
"""
import os
import sys
from dotenv import load_dotenv
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.core.task_manager import TaskManager
from simulated_mind.core.planner import Planner, Goal
from simulated_mind.core.sub_agent import SubAgent
from simulated_mind.core.meta_agent import MetaAgent
from simulated_mind.journal.journal import Journal
from simulated_mind.core.local_llm_client import create_local_llm_client

# Load .env if present
load_dotenv()

# Read MEM0_API_KEY from env
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")
USER_ID = os.environ.get("USER_ID", "default_user")

# Initialize Journal
journal = Journal()

# Initialize MemoryDAO (live or fallback)
memory_dao = MemoryDAO(api_key=MEM0_API_KEY, journal=journal)

# Initialize TaskManager
# (user_id can be extended for multi-user scenarios)
task_manager = TaskManager(memory_dao=memory_dao, journal=journal, user_id=USER_ID)

# Initialize Planner with TaskManager and MemoryDAO
def make_planner():
    # Local LLM config
    llm_backend = os.getenv("LLM_BACKEND", "mock")
    model_name = os.getenv("LLM_MODEL_NAME", "microsoft/DialoGPT-small")
    model_path = os.getenv("LLM_MODEL_PATH")
    local_llm_client = None
    try:
        if llm_backend == "transformers":
            local_llm_client = create_local_llm_client("transformers", model_name=model_name)
        elif llm_backend == "rwkv":
            local_llm_client = create_local_llm_client("rwkv", model_path=model_path)
        else:
            local_llm_client = create_local_llm_client("mock")
        if local_llm_client.is_available():
            print(f"Local LLM ({llm_backend}) loaded successfully")
        else:
            print(f"Local LLM ({llm_backend}) not available, using templates only")
    except Exception as e:
        print(f"Failed to load local LLM: {e}")
        print("Falling back to template-only planning")
        local_llm_client = None
    return Planner(memory_store=memory_dao, journal=journal, goal_class=Goal, task_manager=task_manager, local_llm_client=local_llm_client)


# Initialize SubAgent (for demo, single agent)
sub_agent = SubAgent(agent_id=USER_ID, memory=memory_dao, journal=journal)

# Initialize MetaAgent (stub safety_guard for now)
class DummySafetyGuard:
    pass
meta_agent = MetaAgent(memory_dao=memory_dao, safety_guard=DummySafetyGuard(), journal=journal)

# CLI loop for demo/testing
def main():
    print("Simulated Mind Agent System (mem0-backed)")
    print("Type 'exit' to quit.\n")
    while True:
        user_cmd = input("You: ").strip()
        if user_cmd.lower() in ("exit", "quit"): break
        # For demo, treat input as a planning goal
        sub_agent.perceive(user_cmd)
        action = sub_agent.decide()
        print(f"Agent: {action.payload}")

if __name__ == "__main__":
    main()
