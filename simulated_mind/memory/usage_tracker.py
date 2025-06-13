from .cognitive_dao import MemoryType

class BudgetExceededError(Exception):
    """Exception raised when a memory budget is exceeded."""
    pass

class Mem0UsageTracker:
    """Tracks Mem0 Pro API usage against defined budgets."""

    def __init__(self, max_memories: int = 50000, max_retrievals: int = 5000):
        self.max_memories = max_memories
        self.max_retrievals = max_retrievals
        self.memories_added = 0
        self.retrievals_made = 0

    def track_add(self, memory_type: MemoryType, size: int):
        """Track a memory addition and check against the budget."""
        if self.memories_added >= self.max_memories:
            raise BudgetExceededError(f"Memory budget of {self.max_memories} exceeded.")
        self.memories_added += 1

    def track_search(self, memory_type: MemoryType, query_length: int, results_count: int):
        """Track a memory search and check against the budget."""
        if self.retrievals_made >= self.max_retrievals:
            raise BudgetExceededError(f"Retrieval budget of {self.max_retrievals} exceeded.")
        self.retrievals_made += 1
