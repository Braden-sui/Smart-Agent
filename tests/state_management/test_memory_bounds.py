import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulated_mind.core.local_llm_client import RWKV7GGUFClient
from simulated_mind.journal.journal import Journal

class TestMemoryBounds(unittest.TestCase):
    """
    Tests the memory bounding capabilities of the ThreadSafeStateManager,
    specifically the trimming of conversation history.
    """

    @patch('simulated_mind.core.local_llm_client.Llama')
    def test_conversation_history_is_bounded(self, mock_llama_class):
        """
        Verify that the conversation history is trimmed to max_history.
        """
        # Arrange
        max_history_limit = 5
        journal = Journal()
        
        # Mock the Llama instance to prevent actual model loading
        mock_llama_instance = MagicMock()
        mock_llama_class.return_value = mock_llama_instance

        client = RWKV7GGUFClient(
            model_path="dummy/path/model.gguf",
            max_history=max_history_limit,
            journal=journal
        )

        # Act
        # Simulate more interactions than the history limit
        num_interactions = 10
        for i in range(num_interactions):
            client._update_conversation_history(f"user_turn_{i}", f"assistant_turn_{i}")

        # Assert
        final_state = client.get_state()
        final_history = final_state.get("conversation_history", [])
        
        self.assertEqual(len(final_history), max_history_limit, "History should be trimmed to the max limit.")
        
        # Verify that the last entry in the trimmed history is correct
        last_entry = final_history[-1]
        self.assertEqual(last_entry['role'], 'assistant')
        self.assertEqual(last_entry['content'], f"assistant_turn_{num_interactions - 1}")
        
        # Verify the first entry in the trimmed history is also correct
        total_entries_generated = num_interactions * 2
        first_kept_entry_index = total_entries_generated - max_history_limit
        
        expected_role = 'user' if first_kept_entry_index % 2 == 0 else 'assistant'
        expected_interaction_index = first_kept_entry_index // 2
        
        first_entry = final_history[0]
        self.assertEqual(first_entry['role'], expected_role)
        self.assertEqual(first_entry['content'], f"{expected_role}_turn_{expected_interaction_index}")

if __name__ == '__main__':
    unittest.main()
