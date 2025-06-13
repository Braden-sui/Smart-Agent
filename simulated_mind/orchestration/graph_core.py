"""Graph management for Graph-of-Thoughts reasoning engine.

This module provides the foundational `GraphOfThoughts` class implementing all
mandatory graph operations required by the project specification. It is strictly
focused on structural concerns – reasoning phases live in the higher-level
engine class.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .thought_node import ThoughtNode, ThoughtStatus

__all__ = ["GraphOfThoughts", "GraphConsistencyError"]

LOGGER = logging.getLogger(__name__)


class GraphConsistencyError(RuntimeError):
    """Raised when adjacency lists diverge from expected invariants."""


class GraphOfThoughts:
    """In-memory DAG supporting Graph-of-Thoughts reasoning.

    Only structural methods are included here; async reasoning phases are
    attached in higher-level specialised classes.
    """

    # ---------------------------------------------------------------------
    # Construction & configuration
    # ---------------------------------------------------------------------

    def __init__(
        self,
        llm_client: Any,
        memory_client: Any | None = None,
        *,
        max_depth: int = 4,
        branching_factor: int = 3,
        feedback_loops: bool = True,
        prune_threshold: float = 0.2,
        reuse_cache: bool = True,
        execution_timeout: int = 60,
        evidence_timeout: int = 10,
        enable_metrics: bool = True,
        token_budget: int | None = None,
        parallel_branches: int = 10,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 4.0,
        log_level: int = logging.INFO,
        **config: Any,
    ) -> None:
        # Parameter validation ------------------------------------------------
        if llm_client is None:
            raise ValueError("llm_client is required")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if not isinstance(branching_factor, int) or branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if not (0.0 < prune_threshold <= 1.0):
            raise ValueError("prune_threshold must be within (0, 1]")
        if execution_timeout <= 0:
            raise ValueError("execution_timeout must be > 0 seconds")
        if evidence_timeout <= 0:
            raise ValueError("evidence_timeout must be > 0 seconds")

        # Core storage -------------------------------------------------------
        self.nodes: Dict[str, ThoughtNode] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

        # Extended bookkeeping ----------------------------------------------
        self.llm_client = llm_client
        self.memory_client = memory_client
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.config: Dict[str, Any] = {
            "max_depth": max_depth,
            "branching_factor": branching_factor,
            "feedback_loops": feedback_loops,
            "prune_threshold": prune_threshold,
            "reuse_cache": reuse_cache,
            "execution_timeout": execution_timeout,
            "evidence_timeout": evidence_timeout,
            "enable_metrics": enable_metrics,
            "token_budget": token_budget,
            "parallel_branches": parallel_branches,
            "retry_backoff_base": retry_backoff_base,
            "retry_backoff_max": retry_backoff_max,
            **config,
        }
        self.metrics: Dict[str, Any] = {}
        self._log_level = log_level
        LOGGER.setLevel(log_level)

    # ------------------------------------------------------------------
    # Public – Core Reasoning Pipeline (signatures only – implementation
    # lives in specialised class to follow spec separation)
    # ------------------------------------------------------------------

    async def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # noqa: D401
        raise NotImplementedError("Reasoning pipeline implemented in subclass")

    async def _decomposition_phase(self, root_node: ThoughtNode) -> List[ThoughtNode]:
        raise NotImplementedError

    async def _exploration_phase(self, decomp_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        raise NotImplementedError

    async def _synthesis_phase(self, exploration_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        raise NotImplementedError

    async def _validation_phase(self, synthesis_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Graph manipulation ------------------------------------------------
    # ------------------------------------------------------------------

    def add_node(self, node: ThoughtNode) -> bool:
        """Insert *node* into the graph after validation."""
        if node.id in self.nodes:
            LOGGER.debug("add_node: Node %s already present", node.id)
            return False
        if not node.validate_integrity():
            raise ValueError("Cannot add invalid ThoughtNode to graph")

        self.nodes[node.id] = node
        self._touch()
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove node and all connecting edges. Returns *True* if present."""
        if node_id not in self.nodes:
            return False
        # Remove edges
        for child in list(self.adjacency[node_id]):
            self.reverse_adjacency[child].discard(node_id)
        for parent in list(self.reverse_adjacency[node_id]):
            self.adjacency[parent].discard(node_id)
        self.adjacency.pop(node_id, None)
        self.reverse_adjacency.pop(node_id, None)
        self.nodes.pop(node_id)
        self._touch()
        return True

    def add_edge(self, parent_id: str, child_id: str) -> bool:
        """Create directed edge parent → child.

        Ensures both nodes exist and avoids self-loops / cycles.
        """
        if parent_id == child_id:
            raise ValueError("Self-loops are not allowed in Thought graph")
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise KeyError("Both nodes must exist before connecting")
        if child_id in self.adjacency[parent_id]:
            return False  # Edge already in place
        # Detect cycle introduction
        if self._would_create_cycle(parent_id, child_id):
            raise GraphConsistencyError("Edge would introduce cycle")
        self.adjacency[parent_id].add(child_id)
        self.reverse_adjacency[child_id].add(parent_id)
        self.nodes[parent_id].child_ids.append(child_id)
        self.nodes[child_id].parent_ids.append(parent_id)
        self._touch()
        return True

    def remove_edge(self, parent_id: str, child_id: str) -> bool:
        if child_id in self.adjacency.get(parent_id, set()):
            self.adjacency[parent_id].remove(child_id)
            self.reverse_adjacency[child_id].remove(parent_id)
            self.nodes[parent_id].child_ids.remove(child_id)
            self.nodes[child_id].parent_ids.remove(parent_id)
            self._touch()
            return True
        return False

    def find_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """Return *all* simple paths from *start_id* to *end_id* (defensive copy)."""
        if start_id not in self.nodes or end_id not in self.nodes:
            raise KeyError("Both nodes must exist in graph")
        paths: List[List[str]] = []
        path: List[str] = []

        def _dfs(current: str, visited: Set[str]) -> None:
            visited.add(current)
            path.append(current)
            if current == end_id:
                paths.append(path.copy())
            else:
                for nxt in self.adjacency.get(current, set()):
                    if nxt not in visited:
                        _dfs(nxt, visited)
            path.pop()
            visited.remove(current)

        _dfs(start_id, set())
        return [p.copy() for p in paths]

    def detect_cycles(self) -> List[List[str]]:
        """Return list of cycles detected (each as node ID list)."""
        visited: Set[str] = set()
        stack: Set[str] = set()
        cycles: List[List[str]] = []

        def _visit(node_id: str):
            visited.add(node_id)
            stack.add(node_id)
            for child in self.adjacency.get(node_id, set()):
                if child not in visited:
                    _visit(child)
                elif child in stack:
                    # Found back-edge -> cycle; collect path
                    cycles.append([*stack, child])
            stack.remove(node_id)

        for node in self.nodes:
            if node not in visited:
                _visit(node)
        return [c.copy() for c in cycles]

    def prune_low_confidence_branches(self, threshold: float) -> int:
        """Prune sub-graphs whose root node confidence < *threshold*.

        Returns the number of nodes removed.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be within [0,1]")
        to_remove: Set[str] = set()
        for node_id, node in self.nodes.items():
            if node.confidence < threshold and not node.parent_ids:  # root low-conf
                to_remove.update(self._collect_subtree(node_id))
        removed = sum(self.remove_node(nid) for nid in to_remove)
        return removed

    def get_subgraph(self, node_ids: Set[str]) -> "GraphOfThoughts":
        """Return defensive copy of induced sub-graph."""
        missing = node_ids - self.nodes.keys()
        if missing:
            raise KeyError(f"Nodes absent from graph: {missing}")
        sub = GraphOfThoughts(self.llm_client, self.memory_client, **self.config)
        # Duplicate nodes first
        for nid in node_ids:
            sub.add_node(self.nodes[nid])
        # Duplicate edges if both endpoints in subset
        for pid in node_ids:
            for cid in self.adjacency.get(pid, set()):
                if cid in node_ids:
                    sub.add_edge(pid, cid)
        return sub

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _collect_subtree(self, root_id: str) -> Set[str]:
        q = deque([root_id])
        collected: Set[str] = set()
        while q:
            nid = q.popleft()
            if nid in collected:
                continue
            collected.add(nid)
            q.extend(self.adjacency.get(nid, set()))
        return collected

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        # Simple reachability: is parent reachable from child?
        return any(parent_id in path for path in self.find_paths(child_id, parent_id))

    def _touch(self) -> None:
        self.updated_at = datetime.utcnow()
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        # Ensure adjacency <-> reverse adjacency congruence
        for pid, children in self.adjacency.items():
            for cid in children:
                if pid not in self.reverse_adjacency[cid]:
                    raise GraphConsistencyError(f"Reverse adjacency missing {pid}->{cid}")
        for cid, parents in self.reverse_adjacency.items():
            for pid in parents:
                if cid not in self.adjacency[pid]:
                    raise GraphConsistencyError(f"Adjacency missing {pid}->{cid}")

    # ------------------------------------------------------------------
    # Async cancellation safety ----------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    async def _async_sleep(seconds: float) -> None:  # Helper that handles cancellation
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:  # pragma: no cover
            LOGGER.warning("Async operation cancelled while sleeping")
            raise
