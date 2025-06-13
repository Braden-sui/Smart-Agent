"""Memory-enhanced Graph-of-Thoughts subclass.

This layer integrates the Mem0 Pro Cognitive Memory Hierarchy with the core
async reasoning engine implemented in :pymod:`graph_reasoning`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from .graph_reasoning import AsyncGraphOfThoughts, random_uuid
from .thought_node import ThoughtNode, ThoughtType

LOGGER = logging.getLogger(__name__)

__all__ = ["MemoryEnhancedGoT"]


class CognitiveMemoryHierarchy:
    """Simplified in-process memory hierarchy wrapper for Mem0 client."""

    def __init__(self, mem0_client: Any):
        self.mem0 = mem0_client

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        # Placeholder implementation using mem0 search API semantics
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.mem0.search, query, limit)
        except Exception as exc:  # pragma: no cover – external failure
            LOGGER.warning("Mem0 search failure: %s", exc)
            return []

    async def store(self, key: str, payload: Dict[str, Any]) -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.mem0.store, key, payload)
        except Exception as exc:
            LOGGER.warning("Mem0 store failure: %s", exc)


class MemoryEnhancedGoT(AsyncGraphOfThoughts):
    """Graph-of-Thoughts engine with Mem0 Pro integration."""

    def __init__(self, llm_client: Any, memory_client: Any, **kwargs):
        super().__init__(llm_client, memory_client=memory_client, **kwargs)
        if memory_client is None:
            raise ValueError("memory_client is required for MemoryEnhancedGoT")
        self.memory_hierarchy = CognitiveMemoryHierarchy(memory_client)

    # ------------------------------------------------------------------
    # Memory augmentation & storage
    # ------------------------------------------------------------------

    async def _augment_with_memory(self, thought: ThoughtNode) -> ThoughtNode:
        """Enrich *thought* content using episodic, semantic and procedural memory."""
        memories = await self.memory_hierarchy.search(thought.content, limit=3)
        if memories:
            combined = '\n'.join(m.get("snippet", "") for m in memories)
            thought.content += f"\n\n[Related memories]\n{combined}"
            thought.metadata.setdefault("memories", memories)
            thought.update_confidence(min(thought.confidence + 0.05 * len(memories), 1.0))
        return thought

    async def _store_reasoning_session(self, session_data: Dict[str, Any]) -> str:
        """Persist full reasoning graph & metadata under a unique key."""
        import uuid

        key = f"got_session:{uuid.uuid4()}"
        await self.memory_hierarchy.store(key, session_data)
        return key

    # ------------------------------------------------------------------
    # Public API override
    # ------------------------------------------------------------------

    async def reason(self, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Run reasoning with memory augmentation and persistence."""
        # Pre-query augmentation using semantic/episodic recall
        # Create a temporary node with valid UUID
        temp_node = ThoughtNode(
            id=random_uuid(),
            content=query,
            thought_type=ThoughtType.OTHER,
            confidence=1.0,
        )
        temp_node = await self._augment_with_memory(temp_node)

        memories = temp_node.metadata.get("memories", [])

        # Delegate to base async pipeline with augmented query
        result = await super().reason(temp_node.content, context)

        # Expose memory snippets in first reasoning path for downstream use
        if memories and result.get("reasoning_paths"):
            # Expose memories at top-level for test visibility
            result["reasoning_paths"][0]["memories"] = memories

        # Persist session synchronously to guarantee store() is called once
        await self._store_reasoning_session(result)
        return result
