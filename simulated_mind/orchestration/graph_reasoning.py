"""Async reasoning pipeline implementation for Graph-of-Thoughts.

This module extends :class:`~simulated_mind.orchestration.graph_core.GraphOfThoughts`
with fully-functional asynchronous decomposition → exploration → synthesis →
validation phases, plus state-preservation utilities and error-recovery logic.

All long-running calls to the LLM client are wrapped with exponential back-off
and proper cancellation handling.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .graph_core import GraphConsistencyError, GraphOfThoughts
from .thought_node import ThoughtNode, ThoughtType, ThoughtStatus

LOGGER = logging.getLogger(__name__)

__all__ = ["StateSnapshot", "ReasoningMetrics", "AsyncGraphOfThoughts"]


@dataclass
class StateSnapshot:
    """Lightweight container for an LLM client state snapshot."""

    raw_state: Any
    created_at: float
    client_type: str

    def is_compatible(self, llm_client: Any) -> bool:
        return hasattr(llm_client, "set_state") and hasattr(llm_client, "get_state")


@dataclass
class ReasoningMetrics:
    total_nodes_created: int
    reasoning_depth: int
    confidence_distribution: Dict[str, float]
    contradiction_count: int
    evidence_quality_score: float
    coherence_score: float
    time_to_completion: float
    token_efficiency: float


class AsyncGraphOfThoughts(GraphOfThoughts):
    """Concrete Graph-of-Thoughts engine with async reasoning pipeline."""

    # --------------------- Public entry point ---------------------

    async def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a complete reasoning cycle and return structured result."""
        start_ts = time.time()
        root_node = self._make_root_node(query)
        self.add_node(root_node)

        snapshot = await self._preserve_llm_state()
        try:
            decomp_nodes = await self._decomposition_phase(root_node)
            exploration_nodes = await self._exploration_phase(decomp_nodes)
            synthesis_nodes = await self._synthesis_phase(exploration_nodes)
            final_nodes = await self._validation_phase(synthesis_nodes)
            answer_node = max(final_nodes, key=lambda n: n.confidence, default=root_node)
        finally:
            await self._restore_llm_state(snapshot)

        elapsed = time.time() - start_ts
        metrics = self._compute_metrics(elapsed)
        return {
            "answer": {
                "content": answer_node.content,
                "confidence": answer_node.confidence,
            },
            "reasoning_paths": [n.to_dict() for n in final_nodes],
            "metadata": {
                "total_nodes": len(self.nodes),
                "reasoning_depth": metrics.reasoning_depth,
                **metrics.__dict__,
            },
        }

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _make_root_node(self, query: str) -> ThoughtNode:  # noqa: D401 – helper
        """Construct the root :class:`ThoughtNode` for a reasoning session.

        Parameters
        ----------
        query:
            Natural-language problem statement to seed the reasoning graph.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        return ThoughtNode(
            id=random_uuid(),
            content=query.strip(),
            thought_type=ThoughtType.OTHER,
            confidence=1.0,
        )

    # ----------------------- Reasoning phases -----------------------

    async def _decomposition_phase(self, root_node: ThoughtNode) -> List[ThoughtNode]:
        children: List[ThoughtNode] = []
        prompt = (
            "Decompose the following problem into 2-5 sub-problems with clear dependencies. "
            "Return JSON list of objects with keys 'id', 'content', 'confidence'.\n\n"
            f"Problem: {root_node.content}\n"
        )
        response = await self._call_llm(prompt)
        try:
            import json  # local import to keep top clean
            items = json.loads(response)
            if not isinstance(items, list):
                raise ValueError("LLM did not return list")
        except Exception as exc:
            LOGGER.error("Decomposition parse error: %s", exc)
            raise
        for item in items[:5]:
            child = ThoughtNode(
                id=item.get("id") or str(random_uuid()),
                content=item["content"],
                thought_type=ThoughtType.DECOMPOSITION,
                confidence=float(item.get("confidence", 0.5)),
            )
            self.add_node(child)
            self.add_edge(root_node.id, child.id)
            children.append(child)
        return children

    async def _exploration_phase(self, decomp_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        tasks = [self._explore_node(n) for n in decomp_nodes]
        results: List[ThoughtNode] = []
        for coro in asyncio.as_completed(tasks, timeout=self.config["execution_timeout"]):
            try:
                node = await coro
                results.append(node)
            except asyncio.TimeoutError:
                LOGGER.warning("Exploration branch timed out")
            except asyncio.CancelledError:
                LOGGER.warning("Exploration cancelled")
                raise
        return results

    async def _synthesis_phase(self, exploration_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        # Simple synthesis: concatenate insights
        synthesis_nodes: List[ThoughtNode] = []
        for parent in exploration_nodes:
            prompt = (
                "Synthesize the following insights into a coherent intermediate answer "
                "with confidence: \n" + parent.content
            )
            response = await self._call_llm(prompt)
            child = ThoughtNode(
                id=str(random_uuid()),
                content=response.strip(),
                thought_type=ThoughtType.SYNTHESIS,
                confidence=min(1.0, parent.confidence + 0.1),
            )
            self.add_node(child)
            self.add_edge(parent.id, child.id)
            synthesis_nodes.append(child)
        return synthesis_nodes

    async def _validation_phase(self, synthesis_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        validated: List[ThoughtNode] = []
        for node in synthesis_nodes:
            prompt = (
                "Validate the following answer. Rate confidence 0-1 JSON {{'confidence': x}}.\n" + node.content
            )
            response = await self._call_llm(prompt)
            try:
                import json;
                data = json.loads(response)
                conf = float(data.get("confidence", 0.0))
            except Exception:
                conf = node.confidence * 0.9
            node.update_confidence(conf)
            node.status = ThoughtStatus.COMPLETED
            validated.append(node)
        return validated

    # ------------------- State management helpers -------------------

    async def _preserve_llm_state(self) -> StateSnapshot:
        if not hasattr(self.llm_client, "get_state"):
            return StateSnapshot(raw_state=None, created_at=time.time(), client_type=type(self.llm_client).__name__)
        try:
            state = self.llm_client.get_state()
            return StateSnapshot(raw_state=state, created_at=time.time(), client_type=type(self.llm_client).__name__)
        except Exception as exc:
            LOGGER.warning("Failed to snapshot LLM state: %s", exc)
            return StateSnapshot(raw_state=None, created_at=time.time(), client_type=type(self.llm_client).__name__)

    async def _restore_llm_state(self, snapshot: StateSnapshot) -> bool:
        if snapshot.raw_state is None:
            return True
        if not snapshot.is_compatible(self.llm_client):
            LOGGER.error("Cannot restore state: incompatible client type")
            return False
        try:
            self.llm_client.set_state(snapshot.raw_state)
            return True
        except Exception as exc:
            LOGGER.error("Failed to restore LLM state: %s", exc)
            return False

    # ---------------------- Utility helpers ------------------------

    async def _call_llm(self, prompt: str, *, max_tokens: int = 256, retries: int = 3) -> str:
        backoff = self.config["retry_backoff_base"]
        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.llm_client.complete_text, prompt, max_tokens
                )
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("LLM returned empty result")
                return result
            except asyncio.CancelledError:
                LOGGER.warning("LLM call cancelled")
                raise
            except Exception as exc:
                LOGGER.warning("LLM failure (attempt %d/%d): %s", attempt + 1, retries, exc)
                await self._async_sleep(backoff)
                backoff = min(backoff * 2, self.config["retry_backoff_max"])
        raise RuntimeError("LLM API failed after retries")

    async def _explore_node(self, node: ThoughtNode) -> ThoughtNode:
        prompt = (
            "Explore the sub-problem thoroughly and provide detailed reasoning steps "
            "with evidence.\n" + node.content
        )
        response = await self._call_llm(prompt, max_tokens=512)
        node.content = response.strip()
        node.reasoning_steps.append("explored")
        node.status = ThoughtStatus.IN_PROGRESS
        node.update_confidence(min(node.confidence + 0.1, 1.0))
        return node

    def _compute_metrics(self, elapsed: float) -> ReasoningMetrics:
        conf_vals = [n.confidence for n in self.nodes.values()]
        depth = max((len(self.find_paths(start_id=root_id, end_id=nid)[0]) - 1)
                     if rid_paths else 0
                     for nid in self.nodes
                     for root_id in [min(self.nodes.keys())]  # simplistic root select
                     for rid_paths in [self.find_paths(root_id, nid)]
                     )
        return ReasoningMetrics(
            total_nodes_created=len(self.nodes),
            reasoning_depth=depth,
            confidence_distribution={"avg": sum(conf_vals) / len(conf_vals) if conf_vals else 0},
            contradiction_count=len(self.detect_cycles()),
            evidence_quality_score=0.0,  # placeholder scoring
            coherence_score=0.0,
            time_to_completion=elapsed,
            token_efficiency=len(self.nodes) / max(1, self.llm_client.total_tokens_used) if hasattr(self.llm_client, "total_tokens_used") else 0.0,
        )

# ----------------- Helpers -----------------

def random_uuid() -> str:  # Lightweight util avoids repeating imports
    import uuid

    return str(uuid.uuid4())
