"""
Test for RWKV7GGUFClient state operations
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from simulated_mind.core.local_llm_client import RWKV7GGUFClient


class TestRWKV7StateOperations:
    """Test state operations for RWKV7GGUFClient"""
    
    @pytest.fixture
    def mock_rwkv_client(self):
        """Create a mock RWKV client for testing without needing actual model file"""
        with patch('simulated_mind.core.local_llm_client.Llama') as mock_llama:
            # Mock the Llama model
            mock_model = Mock()
            mock_model.return_value = {
                'choices': [{'text': 'Test response'}]
            }
            mock_llama.return_value = mock_model
            
            client = RWKV7GGUFClient(
                model_path="dummy_path.gguf",
                context_size=1024
            )
            client.model = mock_model  # Ensure model is set
            return client
    
    def test_get_state_returns_valid_dict(self, mock_rwkv_client):
        """Test that get_state returns a valid dictionary"""
        state = mock_rwkv_client.get_state()
        
        assert isinstance(state, dict)
        assert "version" in state
        assert "context_tokens" in state
        assert "conversation_history" in state
        assert "step_count" in state
        assert "last_response" in state
        assert "metadata" in state
    
    def test_set_state_accepts_valid_dict(self, mock_rwkv_client):
        """Test that set_state accepts a valid dictionary"""
        test_state = {
            "version": "1.0",
            "context_tokens": ["hello", "hi there"],
            "conversation_history": [{"user": "hello", "assistant": "hi there"}],
            "step_count": 1,
            "last_response": "hi there",
            "metadata": {"test": "value"}
        }
        
        # Should not raise an exception
        mock_rwkv_client.set_state(test_state)
        
        # State should be updated
        current_state = mock_rwkv_client.get_state()
        assert current_state["step_count"] == 1
        assert current_state["last_response"] == "hi there"
        assert len(current_state["conversation_history"]) == 1
    
    def test_set_state_rejects_invalid_input(self, mock_rwkv_client):
        """Test that set_state rejects invalid input"""
        with pytest.raises(ValueError, match="State must be a dictionary"):
            mock_rwkv_client.set_state("not a dict")
        
        with pytest.raises(ValueError, match="State must be a dictionary"):
            mock_rwkv_client.set_state(123)
    
    def test_state_persistence_across_operations(self, mock_rwkv_client):
        """Test that state persists across get/set operations"""
        # Set initial state
        initial_state = {
            "version": "1.0",
            "context_tokens": ["test"],
            "conversation_history": [],
            "step_count": 0,
            "last_response": "",
            "metadata": {"initial": True}
        }
        mock_rwkv_client.set_state(initial_state)
        
        # Get state and verify it matches
        retrieved_state = mock_rwkv_client.get_state()
        assert retrieved_state["metadata"]["initial"] == True
        assert retrieved_state["context_tokens"] == ["test"]
        
        # Modify and set again
        retrieved_state["step_count"] = 5
        retrieved_state["metadata"]["modified"] = True
        mock_rwkv_client.set_state(retrieved_state)
        
        # Verify changes persist
        final_state = mock_rwkv_client.get_state()
        assert final_state["step_count"] == 5
        assert final_state["metadata"]["modified"] == True
        assert final_state["metadata"]["initial"] == True
    
    def test_state_updates_after_text_generation(self, mock_rwkv_client):
        """Test that state is updated after text generation"""
        initial_state = mock_rwkv_client.get_state()
        initial_step_count = initial_state["step_count"]
        
        # Mock text generation
        with patch.object(mock_rwkv_client, '_build_minimal_prompt', return_value="Test prompt"):
            mock_rwkv_client.complete_text("Hello", max_tokens=10)
        
        # Check that state was updated
        updated_state = mock_rwkv_client.get_state()
        assert updated_state["step_count"] == initial_step_count + 1
        assert updated_state["last_response"] == "Test response"
        assert len(updated_state["context_tokens"]) > len(initial_state["context_tokens"])
    
    def test_save_and_load_state(self, mock_rwkv_client):
        """Test saving and loading state to/from file"""
        # Set up test state
        test_state = {
            "version": "1.0",
            "context_tokens": ["saved", "state"],
            "conversation_history": [{"user": "save", "assistant": "loaded"}],
            "step_count": 42,
            "last_response": "loaded",
            "metadata": {"saved": True}
        }
        mock_rwkv_client.set_state(test_state)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save state
            success = mock_rwkv_client.save_state(temp_path)
            assert success == True
            assert os.path.exists(temp_path)
            
            # Reset client state
            mock_rwkv_client.reset_conversation()
            empty_state = mock_rwkv_client.get_state()
            assert empty_state["step_count"] == 0
            
            # Load state
            success = mock_rwkv_client.load_state(temp_path)
            assert success == True
            
            # Verify loaded state
            loaded_state = mock_rwkv_client.get_state()
            assert loaded_state["step_count"] == 42
            assert loaded_state["last_response"] == "loaded"
            assert loaded_state["metadata"]["saved"] == True
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_reset_conversation_clears_state(self, mock_rwkv_client):
        """Test that reset_conversation clears both conversation and state"""
        # Add some conversation history
        mock_rwkv_client.conversation_history = [
            {"user": "hello", "assistant": "hi"},
            {"user": "how are you", "assistant": "good"}
        ]
        
        # Set some state
        test_state = mock_rwkv_client.get_state()
        test_state["step_count"] = 10
        test_state["last_response"] = "previous response"
        mock_rwkv_client.set_state(test_state)
        
        # Reset
        mock_rwkv_client.reset_conversation()
        
        # Verify everything is cleared
        assert len(mock_rwkv_client.conversation_history) == 0
        
        reset_state = mock_rwkv_client.get_state()
        assert reset_state["step_count"] == 0
        assert reset_state["last_response"] == ""
        assert len(reset_state["context_tokens"]) == 0
        assert len(reset_state["conversation_history"]) == 0


def test_got_engine_compatibility():
    """Test that the state methods are compatible with GoT engine expectations"""
    with patch('simulated_mind.core.local_llm_client.Llama') as mock_llama:
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        client = RWKV7GGUFClient("dummy.gguf", 1024)
        client.model = mock_model
        
        # Test that the client has the required methods
        assert hasattr(client, 'get_state')
        assert hasattr(client, 'set_state')
        assert callable(client.get_state)
        assert callable(client.set_state)
        
        # Test that the methods work as expected by GoT engine
        state = client.get_state()
        assert isinstance(state, dict)
        
        # Should be able to set the same state back
        client.set_state(state)
        
        # Should be able to get it again
        state2 = client.get_state()
        assert isinstance(state2, dict)


if __name__ == "__main__":
    pytest.main([__file__])
