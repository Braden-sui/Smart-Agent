import pytest
import threading
from simulated_mind.core.local_llm_client import RWKV7GGUFClient, StateSnapshot
from simulated_mind.journal.journal import Journal

@pytest.fixture
def client() -> RWKV7GGUFClient:
    """
    Provides a RWKV7GGUFClient instance for testing.
    The model is not loaded, as state management is independent.
    """
    # Using a dummy path since the model won't actually be loaded.
    return RWKV7GGUFClient(model_path="dummy/path/model.gguf")

def worker_atomic(client: RWKV7GGUFClient, num_iterations: int):
    """
    Worker function that performs thread-safe atomic updates to the client's state.
    It increments a counter in the state's metadata.
    """
    def updater(current_snapshot: StateSnapshot) -> StateSnapshot:
        # Correctly access metadata and get the counter
        current_counter = current_snapshot.metadata.get('counter', 0)
        new_metadata = current_snapshot.metadata.copy()
        new_metadata['counter'] = current_counter + 1
        # Return a new snapshot with the updated metadata
        return current_snapshot.with_update(metadata=new_metadata)

    for _ in range(num_iterations):
        client.atomic_state_update(updater)

def test_state_concurrency_is_fixed(client: RWKV7GGUFClient):
    """
    Verifies that the ThreadSafeStateManager and atomic_state_update method
    prevent race conditions during concurrent state modifications.
    """
    num_threads = 10
    iterations_per_thread = 100
    
    # Initialize the counter in the state to 0
    def init_updater(snapshot: StateSnapshot) -> StateSnapshot:
        return snapshot.with_update(metadata={'counter': 0})
    client.atomic_state_update(init_updater)

    # Create and start worker threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker_atomic, args=(client, iterations_per_thread))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Get the final state and check the counter
    final_state = client.get_state()
    final_counter = final_state['metadata'].get('counter', 0)
    expected_counter = num_threads * iterations_per_thread

    # Assert that the final counter matches the expected value
    assert final_counter == expected_counter, (
        f"Race condition still present! "
        f"Final counter was {final_counter}, expected {expected_counter}."
    )

def test_state_recovery_and_journaling():
    """
    Verifies that state changes are logged to the journal and that state can be
    successfully restored from a snapshot.
    """
    # Use a custom sink to capture journal events for verification
    captured_events = []
    def test_sink(label, payload):
        captured_events.append((label, payload))

    journal = Journal(sink=test_sink)
    client1 = RWKV7GGUFClient(model_path="dummy/path", journal=journal)

    # 1. Perform a state update and verify it's logged
    def updater(s: StateSnapshot) -> StateSnapshot:
        return s.with_update(metadata={'test_key': 'test_value'})
    client1.atomic_state_update(updater)

    update_events = [e for e in captured_events if e[0] == 'state_update_success']
    assert len(update_events) == 1
    assert update_events[0][1]['new_version'] == 1

    # 2. Get a snapshot of the state
    snapshot = client1._state_manager.get_current_snapshot()
    assert snapshot.metadata['test_key'] == 'test_value'

    # 3. Create a new client and restore its state from the snapshot
    client2 = RWKV7GGUFClient(model_path="dummy/path", journal=journal)
    client2._state_manager.set_snapshot(snapshot)

    # 4. Verify the state was restored correctly
    restored_state = client2.get_state()
    assert restored_state['metadata']['test_key'] == 'test_value'
    assert restored_state['version'] == 1

    set_events = [e for e in captured_events if e[0] == 'state_set_success']
    assert len(set_events) == 1
    assert set_events[0][1]['set_to_version'] == 1


def test_state_memory_bounding():
    """
    Verifies that the state manager correctly bounds the conversation history
    to prevent unbounded memory growth.
    """
    max_history = 5
    client = RWKV7GGUFClient(model_path="dummy/path", max_history=max_history)

    # Add more items than the max_history limit allows
    for i in range(max_history + 5):
        user_input = f"User message {i}"
        assistant_response = f"Assistant response {i}"
        client._update_conversation_history(user_input, assistant_response)

    final_state = client.get_state()
    history_length = len(final_state['conversation_history'])

    # The history should be trimmed to the max_history limit, which is the total
    # number of individual entries (user or assistant messages).
    expected_length = max_history

    assert history_length == expected_length, (
        f"Memory bounding failed! History length is {history_length}, "
        f"expected {expected_length}."
    )

