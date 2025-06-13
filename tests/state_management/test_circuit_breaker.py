import pytest
import time
from simulated_mind.core.local_llm_client import ThreadSafeStateManager, StateSnapshot, CircuitBreakerOpenError, CircuitBreakerState
from simulated_mind.journal.journal import Journal

@pytest.fixture
def circuit_breaker_manager():
    """Provides a ThreadSafeStateManager with a sensitive circuit breaker for testing."""
    return ThreadSafeStateManager(
        failure_threshold=2,
        failure_window_seconds=1, # short window
        cooldown_seconds=0.2  # Short cooldown for faster testing
    )

def failing_updater(snapshot: StateSnapshot) -> StateSnapshot:
    """An updater function that always fails."""
    raise ValueError("Simulated update failure")

def successful_updater(snapshot: StateSnapshot) -> StateSnapshot:
    """An updater function that always succeeds."""
    return snapshot.with_update(metadata={'success': True})

def test_circuit_breaker_opens_and_resets(circuit_breaker_manager: ThreadSafeStateManager):
    """
    Verifies that the circuit breaker:
    1. Opens after reaching the failure threshold.
    2. Blocks calls while open.
    3. Transitions to half-open after the cooldown.
    4. Resets to closed after a successful call in half-open state.
    """
    manager = circuit_breaker_manager

    # 1. Trip the circuit breaker by causing failures
    for _ in range(manager._failure_threshold):
        with pytest.raises(ValueError, match="Simulated update failure"):
            manager.atomic_update(failing_updater)

    # 2. Verify the circuit is now open and blocks calls
    with pytest.raises(CircuitBreakerOpenError):
        manager.atomic_update(successful_updater)

    # 3. Wait for the cooldown period to elapse
    time.sleep(manager._cooldown.total_seconds() + 0.05)

    # 4. Verify the circuit is now half-open and allows one attempt
    # A successful update should close the circuit again.
    manager.atomic_update(successful_updater)
    assert manager._circuit_state == CircuitBreakerState.CLOSED

    # 5. Verify the circuit is closed and allows further calls
    manager.atomic_update(successful_updater)

def test_circuit_breaker_resets_after_window(circuit_breaker_manager: ThreadSafeStateManager):
    """
    Verifies that failures older than the failure window are pruned,
    preventing the circuit from opening.
    """
    manager = circuit_breaker_manager

    # Record one failure
    with pytest.raises(ValueError):
        manager.atomic_update(failing_updater)

    # Wait for the failure window to pass
    time.sleep(manager._failure_window.total_seconds() + 0.05)

    # Record another failure. The first one should have expired.
    with pytest.raises(ValueError):
        manager.atomic_update(failing_updater)

    # The circuit should not be open, so a successful update should pass
    try:
        manager.atomic_update(successful_updater)
    except CircuitBreakerOpenError:
        pytest.fail("Circuit breaker opened when it should have remained closed.")
