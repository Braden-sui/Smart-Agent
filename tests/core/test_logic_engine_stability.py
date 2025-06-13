import pytest
from simulated_mind.core.logic_engine import (
    LogicEngine, 
    LogicContext, 
    LogicEngineError,
    PrimitiveNotFoundError,
    GraphExecutionError,
    MaxStepsExceededError
)
from simulated_mind.core.logic_graph import LogicGraph
from simulated_mind.journal.journal import Journal
from simulated_mind.core.logic_primitives import PrimitiveResult

# --- Test Fixtures ---

@pytest.fixture
def captured_journal():
    """Provides a journal that captures all events for inspection."""
    captured_events = []
    def test_sink(label, payload):
        captured_events.append((label, payload))
    journal = Journal(sink=test_sink)
    journal.captured_events = captured_events
    return journal

@pytest.fixture
def engine(captured_journal):
    """Provides a LogicEngine instance with a limited primitive registry and a journal."""
    def success_primitive(context, args):
        return PrimitiveResult(success=True)

    def failure_primitive(context, args):
        return PrimitiveResult(success=False, error_message="Simulated failure")

    registry = {
        "SUCCESS": success_primitive,
        "FAILURE": failure_primitive,
    }
    return LogicEngine(primitive_registry=registry, journal=captured_journal)

# --- Test Cases ---

def test_engine_raises_primitive_not_found(engine):
    """Verifies that a PrimitiveNotFoundError is raised for an unknown primitive."""
    graph_yaml = """
    graph_id: test_missing_primitive
    entry_node: start
    nodes:
      start:
        primitive: UNKNOWN_PRIMITIVE
    """
    graph = LogicGraph.load_from_yaml_string(graph_yaml)
    with pytest.raises(PrimitiveNotFoundError, match="'UNKNOWN_PRIMITIVE' not found"):
        engine.execute_graph(graph)

def test_engine_raises_graph_execution_error_on_invalid_node(engine):
    """Verifies a GraphExecutionError is raised for a non-existent node ID."""
    graph_yaml = """
    graph_id: test_invalid_node
    entry_node: start
    nodes:
      start:
        primitive: SUCCESS
        next_node: non_existent_node
    """
    graph = LogicGraph.load_from_yaml_string(graph_yaml)
    with pytest.raises(GraphExecutionError, match="'non_existent_node' not found"):
        engine.execute_graph(graph)

def test_engine_raises_graph_execution_error_on_primitive_failure(engine):
    """Verifies a GraphExecutionError is raised when a primitive returns success=False."""
    graph_yaml = """
    graph_id: test_primitive_failure
    entry_node: start
    nodes:
      start:
        primitive: FAILURE
    """
    graph = LogicGraph.load_from_yaml_string(graph_yaml)
    with pytest.raises(GraphExecutionError, match="Simulated failure"):
        engine.execute_graph(graph)

def test_engine_raises_max_steps_exceeded(engine):
    """Verifies a MaxStepsExceededError is raised for an infinite loop."""
    graph_yaml = """
    graph_id: test_infinite_loop
    entry_node: loop
    nodes:
      loop:
        primitive: SUCCESS
        next_node: loop
    """
    graph = LogicGraph.load_from_yaml_string(graph_yaml)
    with pytest.raises(MaxStepsExceededError, match="exceeded 5 steps"):
        engine.execute_graph(graph, max_steps=5)

def test_journal_logs_successful_execution(engine, captured_journal):
    """Verifies that the journal correctly logs a successful graph execution."""
    graph_yaml = """
    graph_id: test_success_log
    entry_node: start
    nodes:
      start:
        primitive: SUCCESS
        next_node: end
      end:
        primitive: SUCCESS
    """
    graph = LogicGraph.load_from_yaml_string(graph_yaml)
    engine.execute_graph(graph)

    labels = [e[0] for e in captured_journal.captured_events]
    assert labels.count("logic_engine:start") == 1
    assert labels.count("logic_engine:step") == 2
    assert labels.count("logic_engine:complete") == 1

    start_event = captured_journal.captured_events[0]
    assert start_event[1]["graph_id"] == "test_success_log"
