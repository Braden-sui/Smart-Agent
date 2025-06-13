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
    llm_backend = os.getenv("LLM_BACKEND", "rwkv7-gguf")
    model_path = os.getenv("RWKV_MODEL_PATH", "./models/rwkv-v7-2.9b-g1-f16.gguf") 
    context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096")) # Default to 4096 if not set
    
    print(f"--- Initializing Planner --- ")
    print(f"Attempting to use LLM Backend: {llm_backend}")
    if llm_backend == "rwkv7-gguf":
        print(f"RWKV Model Path: {model_path}")
        print(f"Context Size: {context_size}")

    local_llm_client = None
    try:
        # The create_local_llm_client factory will handle specific backend logic including RWKV7GGUF
        local_llm_client = create_local_llm_client(
            backend=llm_backend,
            model_path=model_path,      # Relevant for rwkv7-gguf
            context_size=context_size,  # Relevant for rwkv7-gguf
            # model_name=os.getenv("LLM_MODEL_NAME") # Relevant for transformers, pass if needed
        )

        if local_llm_client and local_llm_client.is_available():
            print(f" {llm_backend.upper()} client created and model is available.")
        else:
            # If client was created but model isn't available, or if client is None from the start
            print(f" {llm_backend.upper()} client could not be made available (model missing, config error, or client is None).")
            print(" Planner will operate without LLM assistance (relying on templates and fallback logic).")
            # local_llm_client might be an instance that's not available, or None if create_local_llm_client failed early.
            # If it's an instance, Planner will see is_available() as False. If None, Planner handles it.
            
    except ValueError as ve:
        print(f" Configuration error for LLM backend '{llm_backend}': {ve}.")
        print(" Planner will operate without LLM assistance.")
        local_llm_client = None
    except Exception as e:
        print(f" Critical error during {llm_backend.upper()} client setup: {type(e).__name__}: {e}.")
        print(" Planner will operate without LLM assistance.")
        local_llm_client = None
    
    print(f"Planner will use LLM client: {type(local_llm_client).__name__}")
    print("--- Planner Initialized --- ")
    
    return Planner(
        memory_store=memory_dao, 
        journal=journal, 
        goal_class=Goal, 
        task_manager=task_manager, 
        local_llm_client=local_llm_client
    )


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
