import pytest
from unittest.mock import MagicMock
from simulated_mind.core.rwkv7_optimized_client import RWKV7OptimizedClient

@pytest.fixture
def mock_optimized_client(mocker):
    """
    Provides a mocked RWKV7OptimizedClient instance.

    This fixture now relies on the lazy-loading mechanism of the parent class.
    It instantiates the client normally and then assigns a mock to the `model`
    attribute, avoiding the need for complex patching of the parent's __init__.
    """
    # Instantiate the client. The __init__ is now safe due to lazy loading.
    client = RWKV7OptimizedClient(model_path="/fake/path/model.gguf", cache_size=10)

    # Manually assign a mock model to prevent any actual model operations.
    client.model = MagicMock()
    
    # The parent __init__ is called, but _load_model is not. We need to manually
    # create the initial empty state that _load_model would have created.
    client._current_state = client._create_empty_state()
    
    yield client

class TestRWKV7OptimizedClient:
    def test_initialization(self, mock_optimized_client):
        """Test that the optimized client initializes correctly.""""""Test that the optimized client initializes correctly."""
        assert mock_optimized_client is not None
        assert mock_optimized_client.cache_size == 10
        assert isinstance(mock_optimized_client.state_cache, dict)

    def test_state_caching(self, mock_optimized_client):
        """Test that get_state and set_state use the cache correctly."""
        # Set an initial state before caching tests
        mock_optimized_client.set_state(mock_optimized_client._create_empty_state())

        state1 = {"data": "state1", "conversation_history": []}
        
        # Set and get a cached state
        mock_optimized_client.set_state(state1, state_key="key1")
        retrieved_state = mock_optimized_client.get_state(state_key="key1")
        
        assert retrieved_state == state1
        assert "key1" in mock_optimized_client.state_cache

        # Verify that getting a non-existent key returns the current running state
        current_state = mock_optimized_client.get_state() # This is now state1
        retrieved_state_no_key = mock_optimized_client.get_state(state_key="non_existent_key")
        assert retrieved_state_no_key == current_state

    def test_cache_eviction(self, mock_optimized_client):
        """Test that the cache correctly evicts the oldest item when full."""
        # Configure a smaller cache for this test
        mock_optimized_client.cache_size = 2
        mock_optimized_client.state_cache.clear()
        mock_optimized_client.cache_keys.clear()

        state1 = {"data": "state1", "conversation_history": []}
        state2 = {"data": "state2", "conversation_history": []}
        state3 = {"data": "state3", "conversation_history": []}

        mock_optimized_client.set_state(state1, state_key="key1")
        mock_optimized_client.set_state(state2, state_key="key2")
        
        assert "key1" in mock_optimized_client.state_cache
        assert "key2" in mock_optimized_client.state_cache
        assert len(mock_optimized_client.state_cache) == 2

        # This set_state call should trigger eviction of "key1"
        mock_optimized_client.set_state(state3, state_key="key3")
        
        assert "key1" not in mock_optimized_client.state_cache
        assert "key2" in mock_optimized_client.state_cache
        assert "key3" in mock_optimized_client.state_cache
        assert len(mock_optimized_client.state_cache) == 2
