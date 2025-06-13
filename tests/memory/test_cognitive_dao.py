import pytest
from unittest.mock import Mock
from simulated_mind.memory.cognitive_dao import RWKV7CognitiveMemoryDAO, MemoryType

@pytest.fixture
def mock_mem0_client():
    return Mock()

@pytest.fixture
def mock_usage_tracker():
    return Mock()

@pytest.fixture
def cognitive_dao(mock_mem0_client, mock_usage_tracker):
    """Fixture to create a RWKV7CognitiveMemoryDAO instance."""
    return RWKV7CognitiveMemoryDAO(mem0_client=mock_mem0_client, usage_tracker=mock_usage_tracker)

class TestCognitiveDAO:
    def test_initialization(self, cognitive_dao):
        """Test that the DAO initializes correctly."""
        assert cognitive_dao is not None
        assert cognitive_dao.mem0_client is not None
        assert cognitive_dao.usage_tracker is not None
        assert len(cognitive_dao._memories) == len(MemoryType)

    def test_add_working_memory(self, cognitive_dao):
        """Test adding a memory to working memory."""
        test_content = "This is a test memory."
        cognitive_dao.add_memory(MemoryType.WORKING, test_content)
        
        working_mem = cognitive_dao.get_working_memory()
        
        assert len(working_mem) == 1
        assert working_mem[0]['content'] == test_content
        assert 'metadata' in working_mem[0]

    def test_get_working_memory_retrieval(self, cognitive_dao):
        """Test retrieving memories from working memory with a query."""
        cognitive_dao.add_memory(MemoryType.WORKING, "First test memory about cats.")
        cognitive_dao.add_memory(MemoryType.WORKING, "Second test memory about dogs.")
        cognitive_dao.add_memory(MemoryType.WORKING, "Third test memory, also about cats.")

        results = cognitive_dao.get_memories(MemoryType.WORKING, "cats")

        assert len(results) == 2
        assert "cats" in results[0]['content']
        assert "cats" in results[1]['content']

    def test_clear_working_memory(self, cognitive_dao):
        """Test clearing the working memory."""
        cognitive_dao.add_memory(MemoryType.WORKING, "A memory to be cleared.")
        assert len(cognitive_dao.get_working_memory()) == 1

        cognitive_dao.clear_working_memory()
        assert len(cognitive_dao.get_working_memory()) == 0

    def test_add_episodic_memory_uses_mem0(self, cognitive_dao, mock_mem0_client, mock_usage_tracker):
        """Test that adding to episodic memory calls the mem0 client and usage tracker."""
        test_content = "An episodic memory."
        test_metadata = {"timestamp": "2025-06-13"}
        
        cognitive_dao.add_memory(MemoryType.EPISODIC, test_content, metadata=test_metadata.copy())
        
        # Verify mem0_client was called
        expected_metadata = test_metadata.copy()
        expected_metadata['type'] = MemoryType.EPISODIC.value
        mock_mem0_client.add.assert_called_once_with(test_content, metadata=expected_metadata)
        
        # Verify usage_tracker was called
        mock_usage_tracker.track_add.assert_called_once_with(memory_type=MemoryType.EPISODIC, size=len(test_content))
        
        # Verify that working memory was not affected
        assert len(cognitive_dao.get_working_memory()) == 0

    def test_get_episodic_memory_uses_mem0(self, cognitive_dao, mock_mem0_client, mock_usage_tracker):
        """Test that getting episodic memories calls the mem0 client and usage tracker."""
        query = "search for something"
        limit = 10
        mock_results = [{"content": "result1"}, {"content": "result2"}]
        mock_mem0_client.search.return_value = mock_results

        results = cognitive_dao.get_memories(MemoryType.EPISODIC, query, limit=limit)

        # Verify mem0_client was called
        mock_mem0_client.search.assert_called_once_with(query, limit=limit, metadata={'type': MemoryType.EPISODIC.value})

        # Verify usage_tracker was called
        mock_usage_tracker.track_search.assert_called_once_with(memory_type=MemoryType.EPISODIC, query_length=len(query), results_count=len(mock_results))

        # Verify results
        assert results == mock_results

    def test_consolidate_memories(self, cognitive_dao, mock_mem0_client, mock_usage_tracker):
        """Test that consolidation moves memories from working to episodic and clears working memory."""
        # Add memories to working memory
        cognitive_dao.add_memory(MemoryType.WORKING, "memory 1", {"id": 1})
        cognitive_dao.add_memory(MemoryType.WORKING, "memory 2", {"id": 2})

        assert len(cognitive_dao.get_working_memory()) == 2

        # Consolidate
        cognitive_dao.consolidate_memories()

        # Verify working memory is cleared
        assert len(cognitive_dao.get_working_memory()) == 0

        # Verify mem0_client was called for each memory
        assert mock_mem0_client.add.call_count == 2
        mock_mem0_client.add.assert_any_call("memory 1", metadata={'id': 1, 'type': 'episodic'})
        mock_mem0_client.add.assert_any_call("memory 2", metadata={'id': 2, 'type': 'episodic'})

        # Verify usage_tracker was called
        assert mock_usage_tracker.track_add.call_count == 2
        mock_usage_tracker.track_add.assert_any_call(memory_type=MemoryType.EPISODIC, size=len("memory 1"))
        mock_usage_tracker.track_add.assert_any_call(memory_type=MemoryType.EPISODIC, size=len("memory 2"))
