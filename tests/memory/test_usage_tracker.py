import pytest
from simulated_mind.memory.usage_tracker import Mem0UsageTracker, BudgetExceededError
from simulated_mind.memory.cognitive_dao import MemoryType

@pytest.fixture
def usage_tracker():
    """Fixture for a Mem0UsageTracker with a small budget for testing."""
    return Mem0UsageTracker(max_memories=2, max_retrievals=2)

class TestMem0UsageTracker:
    def test_initialization(self):
        """Test that the tracker initializes with correct values."""
        tracker = Mem0UsageTracker(max_memories=100, max_retrievals=50)
        assert tracker.max_memories == 100
        assert tracker.max_retrievals == 50
        assert tracker.memories_added == 0
        assert tracker.retrievals_made == 0

    def test_track_add(self, usage_tracker):
        """Test tracking memory additions."""
        usage_tracker.track_add(MemoryType.EPISODIC, size=10)
        assert usage_tracker.memories_added == 1
        usage_tracker.track_add(MemoryType.SEMANTIC, size=20)
        assert usage_tracker.memories_added == 2

    def test_track_add_exceeds_budget(self, usage_tracker):
        """Test that tracking additions raises an error when the budget is exceeded."""
        usage_tracker.track_add(MemoryType.EPISODIC, size=10)
        usage_tracker.track_add(MemoryType.EPISODIC, size=10)
        with pytest.raises(BudgetExceededError):
            usage_tracker.track_add(MemoryType.EPISODIC, size=10)

    def test_track_search(self, usage_tracker):
        """Test tracking memory searches."""
        usage_tracker.track_search(MemoryType.EPISODIC, query_length=5, results_count=1)
        assert usage_tracker.retrievals_made == 1
        usage_tracker.track_search(MemoryType.SEMANTIC, query_length=10, results_count=3)
        assert usage_tracker.retrievals_made == 2

    def test_track_search_exceeds_budget(self, usage_tracker):
        """Test that tracking searches raises an error when the budget is exceeded."""
        usage_tracker.track_search(MemoryType.EPISODIC, query_length=5, results_count=1)
        usage_tracker.track_search(MemoryType.EPISODIC, query_length=5, results_count=1)
        with pytest.raises(BudgetExceededError):
            usage_tracker.track_search(MemoryType.EPISODIC, query_length=5, results_count=1)
