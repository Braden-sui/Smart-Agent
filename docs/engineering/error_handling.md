# Error Handling and Observability

This document outlines the standards for error handling, logging, and observability within the `simulated-minds` project. Adherence to these patterns is critical for maintaining system stability and ensuring that failures are detectable, diagnosable, and recoverable.

## 1. Core Philosophy

- **No Silent Failures**: Errors must never be silently ignored. All exceptions should be caught, logged with context, and handled gracefully.
- **Structured Logging Over Prints**: Unstructured `print()` statements are forbidden in application code. All diagnostic output, events, and errors must be logged through the `Journal` system.
- **Fail Fast, Recover Gracefully**: Components should be designed to fail fast when an unrecoverable error occurs, while system-level orchestrators should implement recovery patterns like retries and circuit breakers.
- **Clear Error Boundaries**: Errors should be handled at the appropriate architectural layer. Low-level components should throw specific, typed exceptions, while higher-level services handle them.

## 2. The Journal System

The `Journal` is the central system for all logging and event tracking. It provides a structured, queryable record of system behavior.

### Usage

- **Events**: Log significant business logic events (e.g., `planner.create_plan.start`).
- **Errors**: Log all caught exceptions with a unique error code (e.g., `rwkv7_client.generation.fail`) and include the exception details in the payload.
- **State Changes**: Log all critical state transitions, especially in the `ThreadSafeStateManager`.

**Example:**

```python
# In an LLM client
try:
    # ... call to model ...
except Exception as e:
    self.journal.log_event("llm_client.generation.fail", {"error": str(e)})
    raise
```

## 3. Exception Handling

- **Typed Exceptions**: Define custom, specific exception classes for different failure domains (e.g., `CircuitBreakerOpenError`). Avoid catching broad `Exception` where possible.
- **Error Boundaries**: The `LogicEngine` and `Planner` act as key error boundaries. They are responsible for catching exceptions from underlying components (like LLM clients) and preventing them from crashing the main application loop.

## 4. State Management and Stability Patterns

### ThreadSafeStateManager & StateSnapshot

- **Immutability**: `StateSnapshot` objects are immutable, preventing race conditions and ensuring that state can be safely passed between threads.
- **Atomic Updates**: All state modifications must go through `ThreadSafeStateManager.atomic_update()`, which guarantees thread safety via locks.

### Circuit Breaker

- **Purpose**: The `ThreadSafeStateManager` implements a circuit breaker to prevent a component from being repeatedly called when it is consistently failing.
- **States**: The breaker operates in three states: `CLOSED` (normal operation), `OPEN` (calls are blocked), and `HALF_OPEN` (a single trial call is allowed after a cooldown).
- **Logging**: All state transitions of the circuit breaker are logged to the `Journal` for observability.

## 5. Memory Management

- **Bounded History**: Long-lived objects like the `RWKV7GGUFClient` must enforce memory limits. The client's conversation history is strictly bounded by the `max_history` parameter to prevent memory leaks.
- **No Unbounded Growth**: Be vigilant about collections or caches that can grow indefinitely. The `context_tokens` field was removed from the state for this reason.
