import pytest
from unittest.mock import Mock
from simulated_mind.memory.cognitive_dao import RWKV7CognitiveMemoryDAO, MemoryType

@pytest.fixture
def mock_llm_client():
    client = Mock()
    client.complete_text.return_value = "This is a consolidated summary."
    return client

@pytest.fixture
def mock_mem0_client():
    return Mock()

@pytest.fixture
def mock_usage_tracker():
    return Mock()

@pytest.fixture
def cognitive_dao(mock_llm_client, mock_mem0_client, mock_usage_tracker):
    """Fixture to create a RWKV7CognitiveMemoryDAO instance."""
    return RWKV7CognitiveMemoryDAO(
        llm_client=mock_llm_client,
        mem0_client=mock_mem0_client,
        usage_tracker=mock_usage_tracker
    )

class TestCognitiveDAO:
    def test_initialization(self, cognitive_dao):
        """Test that the DAO initializes correctly."""
        assert cognitive_dao is not None

    def test_add_working_memory(self, cognitive_dao_fixture):
        """Test adding a memory to working memory."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        dao.add_memory(user_id, MemoryType.WORKING, "Test memory 1")
        dao.add_memory(user_id, MemoryType.WORKING, "Another test memory")

        memories = dao.get_working_memory(user_id)
        assert len(memories) == 2
        assert memories[0]['content'] == "Test memory 1"
        assert 'metadata' in memories[0]

    def test_get_working_memory_retrieval(self, cognitive_dao_fixture):
        """Test retrieving memories from working memory with a query."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        dao.add_memory(user_id, MemoryType.WORKING, "Queryable content")

        # Test working memory retrieval
        results = dao.get_memories(user_id, MemoryType.WORKING, query="Queryable")
        assert len(results) == 1
        assert results[0]['content'] == "Queryable content"

        # Test other memory retrieval (delegated to mock)
        results = dao.get_memories(user_id, MemoryType.EPISODIC, query="any")
        dao.mem0_client.search.assert_called_with("any", user_id=user_id, limit=5, metadata={'type': 'episodic'})
        assert len(results) > 0

    def test_clear_working_memory(self, cognitive_dao_fixture):
        """Test clearing the working memory."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        dao.add_memory(user_id, MemoryType.WORKING, "A memory to be cleared.")
        assert len(dao.get_working_memory(user_id)) == 1

        dao.clear_working_memory(user_id)
        assert len(dao.get_working_memory(user_id)) == 0

    def test_add_episodic_memory_uses_mem0(self, cognitive_dao_fixture, mock_mem0_client, mock_usage_tracker):
        """Test that adding to episodic memory calls the mem0 client and usage tracker."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        test_content = "An episodic memory."
        test_metadata = {"timestamp": "2025-06-13"}
        
        dao.add_memory(user_id, MemoryType.EPISODIC, test_content, metadata=test_metadata.copy())
        
        # Verify mem0_client was called
        expected_metadata = test_metadata.copy()
        expected_metadata['type'] = MemoryType.EPISODIC.value
        mock_mem0_client.add.assert_called_once_with(test_content, user_id=user_id, metadata=expected_metadata)
        
        # Verify usage_tracker was called
        mock_usage_tracker.track_add.assert_called_once_with(memory_type=MemoryType.EPISODIC, size=len(test_content))
        
        # Verify that working memory was not affected
        assert len(dao.get_working_memory(user_id)) == 0

    def test_get_episodic_memory_uses_mem0(self, cognitive_dao_fixture, mock_mem0_client, mock_usage_tracker):
        """Test that getting episodic memories calls the mem0 client and usage tracker."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        query = "search for something"
        limit = 10
        mock_results = [{"content": "result1"}, {"content": "result2"}]
        mock_mem0_client.search.return_value = mock_results

        results = dao.get_memories(user_id, MemoryType.EPISODIC, query, limit=limit)

        # Verify mem0_client was called
        mock_mem0_client.search.assert_called_once_with(query, user_id=user_id, limit=limit, metadata={'type': MemoryType.EPISODIC.value})

        # Verify usage_tracker was called
        mock_usage_tracker.track_search.assert_called_once_with(memory_type=MemoryType.EPISODIC, query_length=len(query), results_count=len(mock_results))

        # Verify results
        assert results == mock_results

    def test_consolidation(self, cognitive_dao_fixture):
        """Test that consolidation summarizes working memory using the LLM."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        dao.add_memory(user_id, MemoryType.WORKING, "Event A happened")
        dao.add_memory(user_id, MemoryType.WORKING, "Event B followed")

        dao.consolidate_memories(user_id)

        # Verify LLM was called for summary
        dao.llm_client.complete_text.assert_called_once()
        call_args = dao.llm_client.complete_text.call_args[0][0]
        assert "Event A happened" in call_args
        assert "Event B followed" in call_args

        # Verify episodic memory was added via mem0
        dao.mem0_client.add.assert_called_once()
        args, kwargs = dao.mem0_client.add.call_args
        assert "Mocked summary" in args[0]
        assert kwargs['user_id'] == user_id
        assert kwargs['metadata']['type'] == 'episodic'

        # Verify working memory was cleared
        assert len(dao.get_working_memory(user_id)) == 0

    def test_distill_semantic_knowledge(self, cognitive_dao_fixture, mock_llm_client, mock_mem0_client):
        """Test that distillation extracts facts from episodic memories and stores them semantically."""
        dao = cognitive_dao_fixture
        user_id = "test_user"
        # Mock episodic memories and LLM response
        episodic_memories = [{'content': 'The sky was blue.'}, {'content': 'The cat sat on the mat.'}]
        dao.get_memories = Mock(return_value=episodic_memories)

        distilled_facts_json = '''
        [
            {"subject": "sky", "predicate": "was", "object": "blue"},
            {"subject": "cat", "predicate": "sat on", "object": "mat"}
        ]
        '''
        mock_llm_client.complete_text.return_value = distilled_facts_json

        # Run distillation
        dao.distill_semantic_knowledge(user_id)

        # Verify LLM was called correctly
        mock_llm_client.complete_text.assert_called_once()
        assert "Extract key facts" in mock_llm_client.complete_text.call_args[0][0]

        # Verify that semantic memories were added via mem0_client
        assert mock_mem0_client.add.call_count == 2
        mock_mem0_client.add.assert_any_call(
            "sky was blue",
            user_id=user_id,
            metadata={'source': 'distillation', 'fact': {'subject': 'sky', 'predicate': 'was', 'object': 'blue'}, 'type': 'semantic'}
        )
        mock_mem0_client.add.assert_any_call(
            "cat sat on mat",
            user_id=user_id,
            metadata={'source': 'distillation', 'fact': {'subject': 'cat', 'predicate': 'sat on', 'object': 'mat'}, 'type': 'semantic'}
        )
