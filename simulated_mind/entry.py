"""
Entrypoint for initializing and running the integrated agent system:
- Loads MEM0_API_KEY from environment (or .env if present)
- Wires MemoryDAO, TaskManager, MetaAgent, SubAgent, Planner
- Supports both live mem0 and in-memory fallback
- Provides a CLI for simple user command input and system response
"""
import os
import sys

import threading
import time
from dotenv import load_dotenv
from simulated_mind.memory.cognitive_dao import RWKV7CognitiveMemoryDAO
from simulated_mind.memory.mem0_client import Mem0Client
from simulated_mind.memory.usage_tracker import UsageTracker
from simulated_mind.core.task_manager import TaskManager
from simulated_mind.core.planner import Planner, Goal
from simulated_mind.core.sub_agent import SubAgent
from simulated_mind.core.meta_agent import MetaAgent
from simulated_mind.journal.journal import Journal
from simulated_mind.core.local_llm_client import create_local_llm_client

# --- Path Setup ---
# Get the absolute path to the directory containing this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the default model path relative to the script's location
_DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'models', 'rwkv7-2.9B-g1-F16.gguf')


# Load .env if present
load_dotenv()

# Read MEM0_API_KEY from env
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")
USER_ID = os.environ.get("USER_ID", "default_user")

# Initialize Journal
journal = Journal()

# Initialize Cognitive Memory DAO
llm_client_for_memory = create_local_llm_client(
    backend=os.getenv("LLM_BACKEND", "rwkv7-gguf"),
    model_path=os.getenv("RWKV_MODEL_PATH", _DEFAULT_MODEL_PATH),
    context_size=int(os.getenv("LLM_CONTEXT_SIZE", "4096")),
    journal=journal
)

if not llm_client_for_memory or not llm_client_for_memory.is_available():
    journal.log_event("memory.init.llm_fail", {"reason": "LLM for memory DAO is not available."})
    # Depending on strictness, we might exit or continue with limited functionality
    # For now, we allow it to proceed, but memory enhancement will be disabled.


memory_dao = RWKV7CognitiveMemoryDAO(
    llm_client=llm_client_for_memory,
    mem0_client=Mem0Client(api_key=MEM0_API_KEY, journal=journal),
    usage_tracker=UsageTracker()
)

# Initialize TaskManager
# (user_id can be extended for multi-user scenarios)
task_manager = TaskManager(memory_dao=memory_dao, journal=journal, user_id=USER_ID)

# Initialize Planner with TaskManager and MemoryDAO
def make_planner():
    journal.log_event("planner.init.start", {})
    llm_backend = os.getenv("LLM_BACKEND", "rwkv7-gguf")
    model_path = os.getenv("RWKV_MODEL_PATH", _DEFAULT_MODEL_PATH) 
    context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))

    init_config = {"llm_backend": llm_backend}
    if llm_backend == "rwkv7-gguf":
        init_config.update({"model_path": model_path, "context_size": context_size})
    journal.log_event("planner.init.config", init_config)

    local_llm_client = None
    try:
        local_llm_client = create_local_llm_client(
            backend=llm_backend,
            model_path=model_path,
            context_size=context_size,
            journal=journal
        )

        if local_llm_client and local_llm_client.is_available():
            journal.log_event("planner.init.llm_success", {"backend": llm_backend})
        else:
            journal.log_event("planner.init.llm_fail", {
                "backend": llm_backend,
                "reason": "Client created but model unavailable, or client is None."
            })
            
    except ValueError as ve:
        journal.log_event("planner.init.llm_error", {
            "backend": llm_backend, 
            "error_type": "ValueError", 
            "error": str(ve)
        })
        local_llm_client = None
    except Exception as e:
        journal.log_event("planner.init.llm_error", {
            "backend": llm_backend, 
            "error_type": type(e).__name__, 
            "error": str(e)
        })
        local_llm_client = None
    
    journal.log_event("planner.init.client_final", {"client_type": type(local_llm_client).__name__})
    journal.log_event("planner.init.end", {})
    
    return Planner(
        memory_store=memory_dao, 
        journal=journal, 
        goal_class=Goal, 
        task_manager=task_manager, 
        local_llm_client=local_llm_client
    )


# Initialize Planner
planner = make_planner()

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
        if user_cmd.lower() in ("exit", "quit"):
            break
        if not user_cmd:
            continue

        # Use Planner to create a plan from the user's command
        plan = planner.create_plan(user_cmd)
        
        if plan:
            print(f"\nPlanner: Created plan with {len(plan)} steps.")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step.description} (ID: {step.id}, Priority: {step.priority})")
            print("Plan stored. Execution logic to be triggered here.\n")
        else:
            print("\nPlanner: Could not create a plan for the given command.\n")

def background_memory_tasks(dao, user_id, interval_seconds=300):
    """Periodically run memory consolidation and distillation."""
    while True:
        journal.log_event("background_memory.run.start", {"user_id": user_id})
        try:
            dao.consolidate_memories(user_id)
            journal.log_event("background_memory.consolidate.success", {"user_id": user_id})
            dao.distill_semantic_knowledge(user_id)
            journal.log_event("background_memory.distill.success", {"user_id": user_id})
        except Exception as e:
            journal.log_event("background_memory.run.error", {"user_id": user_id, "error": str(e)})
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Start background memory tasks in a separate thread
    memory_thread = threading.Thread(
        target=background_memory_tasks,
        args=(memory_dao, USER_ID),
        daemon=True
    )
    memory_thread.start()

    main()
