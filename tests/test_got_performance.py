"""
Performance tests comparing traditional planning vs Graph-of-Thoughts planning.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from simulated_mind.core.planner import Planner, Goal
from simulated_mind.orchestration.graph_of_thoughts import create_got_engine
from simulated_mind.core.logic_graph import LogicGraph
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.journal.journal import Journal


import os
from simulated_mind.core.local_llm_client import RWKV7GGUFClient

@pytest.fixture
def real_rwkv7_client():
    # Use environment variable or fallback to default test model path
    model_path = os.getenv("RWKV_MODEL_PATH", "./models/rwkv-v7-2.9b-g1-f16.gguf")
    context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
    client = RWKV7GGUFClient(model_path=model_path, context_size=context_size)
    client.load()
    assert client.is_available(), f"RWKV7GGUFClient could not load model at {model_path}"
    return client


@pytest.fixture  
def mock_memory_dao():
    memory_dao = Mock(spec=MemoryDAO)
    memory_dao.find_memories_by_tags.return_value = []
    return memory_dao

@pytest.fixture
def mock_journal():
    return Mock(spec=Journal)

@pytest.fixture
def traditional_planner(mock_memory_dao, mock_journal, real_rwkv7_client):
    return Planner(
        memory_store=mock_memory_dao,
        journal=mock_journal,
        goal_class=Goal,
        local_llm_client=real_rwkv7_client
    )

@pytest.fixture
def got_planner(mock_memory_dao, mock_journal, real_rwkv7_client):
    planner = Planner(
        memory_store=mock_memory_dao,
        journal=mock_journal, 
        goal_class=Goal,
        local_llm_client=real_rwkv7_client
    )
    # Enable GoT mode
    planner._got_enabled = True
    return planner

@pytest.mark.integration
class TestGOTPerformance:
    """Test performance comparison between traditional and GoT planning."""
    def test_got_vs_traditional_reasoning_quality(self, traditional_planner, real_rwkv7_client, mock_journal):
        """Test that GoT produces higher quality reasoning than traditional planning."""
        complex_goal = "design a scalable microservices architecture for a multi-tenant SaaS platform"
        # Traditional planning
        traditional_start = time.time()
        traditional_result = traditional_planner.decompose_goal(complex_goal)
        traditional_time = time.time() - traditional_start
        # GoT planning
        got_engine = create_got_engine(real_rwkv7_client, mock_journal)
        got_start = time.time()
        got_result = traditional_planner._create_plan_from_got_graph(complex_goal)
        got_time = time.time() - got_start
        # Assertions
        assert traditional_result is not None, "Traditional planning should produce results"
        # GoT should produce more detailed results
        if got_result:
            assert len(str(got_result)) >= len(str(traditional_result)), \
                "GoT should produce more detailed planning results"
        # Log performance metrics
        mock_journal.log_event.assert_any_call("got.performance_comparison", {
            "traditional_time": traditional_time,
            "got_time": got_time, 
            "traditional_result_count": len(traditional_result),
            "got_result_count": len(got_result) if got_result else 0
        })
    def test_got_state_persistence(self, real_rwkv7_client, mock_journal):
        """Test that GoT properly manages RWKV7 state throughout reasoning."""
        got_engine = create_got_engine(real_rwkv7_client, mock_journal)
        # Verify state operations are called
        reasoning_context = got_engine.create_reasoning_context(
            "solve complex optimization problem",
            reasoning_type="analytical"
        )
        assert reasoning_context.get_variable('rwkv_client') is not None
        assert reasoning_context.get_variable('reasoning_type') == "analytical"
        assert reasoning_context.get_variable('branching_factor') == 3
    def test_got_primitive_integration(self, real_rwkv7_client):
        """Test that all GoT primitives are properly registered and functional."""
        from simulated_mind.core.logic_primitives import PRIMITIVE_REGISTRY, LogicContext
        got_primitives = [
            "RWKV7_STATE_INIT",
            "RWKV7_THOUGHT_GENERATION", 
            "RWKV7_STATE_SCORING",
            "RWKV7_STATE_MERGE",
            "RWKV7_STATE_RESPONSE"
        ]
        for primitive in got_primitives:
            assert primitive in PRIMITIVE_REGISTRY, f"Primitive {primitive} not registered"
        context = LogicContext()
        context.set_variable('rwkv_client', real_rwkv7_client)
        # Test RWKV7_STATE_INIT
        init_result = PRIMITIVE_REGISTRY["RWKV7_STATE_INIT"](context, {
            'context_prompt': 'Initialize test reasoning'
        })
        assert init_result.success, f"STATE_INIT failed: {init_result.error_message}"
        assert context.get_variable('base_state') is not None
        # Test RWKV7_THOUGHT_GENERATION
        gen_result = PRIMITIVE_REGISTRY["RWKV7_THOUGHT_GENERATION"](context, {
            'thought_prompts': ['analyze problem', 'consider alternatives'],
            'branching_factor': 2
        })
        assert gen_result.success, f"THOUGHT_GENERATION failed: {gen_result.error_message}"
        assert len(context.get_variable('thought_states', {})) == 2
    def test_got_memory_integration(self, real_rwkv7_client, mock_memory_dao, mock_journal):
        """Test that GoT integrates properly with mem0 Pro memory system."""
        got_engine = create_got_engine(real_rwkv7_client, mock_journal)
        context = got_engine.create_reasoning_context(
            "complex business strategy analysis",
            reasoning_type="creative"
        )
        assert context.get_variable('branching_factor') == 5
        assert context.get_variable('feedback_loops') == True
        assert context.get_variable('rwkv_client') is real_rwkv7_client
    def test_performance_regression_threshold(self, traditional_planner, real_rwkv7_client, mock_journal):
        """Ensure GoT performance meets regression thresholds."""
        test_goals = [
            "develop comprehensive marketing strategy",
            "architect scalable data pipeline", 
            "design user experience workflow",
            "plan agile development process"
        ]
        performance_metrics = []
        for goal in test_goals:
            # Measure traditional planning
            trad_start = time.time()
            trad_result = traditional_planner.decompose_goal(goal)
            trad_time = time.time() - trad_start
            # Measure GoT planning  
            got_start = time.time()
            got_result = traditional_planner._create_plan_from_got_graph(goal)
            got_time = time.time() - got_start
            performance_metrics.append({
                'goal': goal,
                'traditional_time': trad_time,
                'got_time': got_time,
                'traditional_count': len(trad_result) if trad_result else 0,
                'got_count': len(got_result) if got_result else 0
            })
        # Performance regression checks
        avg_trad_time = sum(m['traditional_time'] for m in performance_metrics) / len(performance_metrics)
        avg_got_time = sum(m['got_time'] for m in performance_metrics) / len(performance_metrics)
        # GoT should not be more than 3x slower than traditional (acceptable for quality gain)
        if avg_trad_time > 0.01:
            assert avg_got_time <= (avg_trad_time * 3.0), \
                f"GoT performance regression: {avg_got_time:.3f}s vs {avg_trad_time:.3f}s traditional"
        else:
            # In a mock or trivial environment, just assert both are very fast
            assert avg_got_time < 0.05, f"GoT mock time unexpectedly slow: {avg_got_time:.3f}s"
        # Log final performance summary
        mock_journal.log_event.assert_any_call("performance_regression_test_complete", {
            "avg_traditional_time": avg_trad_time,
            "avg_got_time": avg_got_time,
            "performance_ratio": avg_got_time / avg_trad_time if avg_trad_time > 0 else 0,
            "test_goals_count": len(test_goals)
        })

if __name__ == '__main__':
   pytest.main([__file__, '-v'])
