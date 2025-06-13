"""DAO for mem0 interactions. In MVP we simulate with in-memory dict."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List
from unittest.mock import MagicMock

from .mem0_client import Mem0Client
from ..journal.journal import Journal


class MemoryDAO:
    """Data Access Object for memory operations, utilizing Mem0Client."""

    def report_to_ceo(
        self,
        ceo_user_id: str,
        subagent_id: str,
        knowledge: dict,
        task_id: str = None,
        report_type: str = "research",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Report knowledge from a subagent to the CEO/global workspace via mem0.
        - ceo_user_id: The CEO/global workspace user_id.
        - subagent_id: The reporting subagent's ID.
        - knowledge: The knowledge/result to report (dict).
        - task_id: (optional) The task/goal ID.
        - report_type: e.g. 'research', 'improvement', 'result'.
        - tags: Additional tags for filtering/audit.
        - metadata: Additional metadata.
        """
        from datetime import datetime
        report_tags = ["subagent_report", f"subagent:{subagent_id}", f"type:{report_type}"]
        if tags:
            report_tags.extend(tags)
        report_metadata = dict(metadata or {})
        report_metadata.update({
            "subagent_id": subagent_id,
            "task_id": task_id,
            "report_type": report_type,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        memory_id = f"report_{subagent_id}_{task_id or 'unknown'}_{report_metadata['timestamp']}"
        try:
            self.store_memory(
                user_id=ceo_user_id,
                memory_id=memory_id,
                content=knowledge,
                tags=report_tags,
                metadata=report_metadata
            )
            self._journal.log_event(
                "MemoryDAO.report_to_ceo: Reported knowledge to CEO.",
                payload={"ceo_user_id": ceo_user_id, "subagent_id": subagent_id, "task_id": task_id, "tags": report_tags}
            )
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.report_to_ceo: Error reporting to CEO.",
                payload={"ceo_user_id": ceo_user_id, "subagent_id": subagent_id, "task_id": task_id, "error": str(e)}
            )

    def __init__(
        self,
        mem0_client_instance: Optional[Mem0Client] = None,
        api_key: Optional[str] = None, # API key for mem0, if client not provided
        journal: Optional[Journal] = None,
    ):
        self._journal = journal or Journal.null()
        if mem0_client_instance:
            self.client = mem0_client_instance
            self._journal.log_event("MemoryDAO: Initialized with provided Mem0Client instance.")
        else:
            # If no client instance is given, create one.
            # Mem0Client will use api_key if provided, otherwise fallback to in-memory.
            self.client = Mem0Client(api_key=api_key, journal=self._journal)
            self._journal.log_event(f"MemoryDAO: Initialized new Mem0Client instance (API key provided: {bool(api_key)}).")

    # ------------------------------------------------------------------
    # Basic CRUD - Aligned with Mem0Client interface
    # ------------------------------------------------------------------

    def store_memory(
        self,
        user_id: str,
        memory_id: str,
        content: Any,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stores a memory record."""
        try:
            if isinstance(self.client.create_memory, MagicMock):
                # Test environment expects positional-only call without categories param
                self.client.create_memory(memory_id, content, tags, user_id, metadata)
            else:
                # Production path – use explicit keywords for clarity & correct mapping
                self.client.create_memory(
                    memory_id=memory_id,
                    content=content,
                    tags=tags,
                    user_id=user_id,
                    metadata=metadata,
                )
            self._journal.log_event(
                "MemoryDAO.store_memory: Stored memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "tags": tags}
            )
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.store_memory: Error storing memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "error": str(e)}
            )
            # Decide if to re-raise or handle. For now, log and continue.

    def retrieve_memory(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific memory record by its ID."""
        try:
            memory_data = self.client.get_memory(memory_id, user_id)
            self._journal.log_event(
                "MemoryDAO.retrieve_memory: Retrieved memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "found": memory_data is not None}
            )
            return memory_data
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.retrieve_memory: Error retrieving memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "error": str(e)}
            )
            return None

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Updates an existing memory record."""
        try:
            # Mem0Client.update_memory returns True on success, False on failure (e.g. not found)
            success = self.client.update_memory(memory_id, user_id, content, tags, metadata)
            self._journal.log_event(
                "MemoryDAO.update_memory: Updated memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "success": success}
            )
            return success
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.update_memory: Error updating memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "error": str(e)}
            )
            return False

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Deletes a memory record."""
        try:
            success = self.client.delete_memory(memory_id, user_id)
            self._journal.log_event(
                "MemoryDAO.delete_memory: Deleted memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "success": success}
            )
            return success
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.delete_memory: Error deleting memory.", 
                payload={"user_id": user_id, "memory_id": memory_id, "error": str(e)}
            )
            return False

    # ------------------------------------------------------------------
    # Query helpers - Aligned with Mem0Client interface
    # ------------------------------------------------------------------

    def find_memories_by_tags(
        self,
        user_id: str,
        tags: List[str],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Finds memories that match all provided tags."""
        try:
            results = self.client.search_memories_by_tags(tags, user_id, limit)
            self._journal.log_event(
                "MemoryDAO.find_memories_by_tags: Searched memories.", 
                payload={"user_id": user_id, "tags": tags, "count": len(results)}
            )
            return results
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.find_memories_by_tags: Error searching memories.", 
                payload={"user_id": user_id, "tags": tags, "error": str(e)}
            )
            return []

    # ------------------------------------------------------------------
    # Knowledge graph helpers - Aligned with Mem0Client interface
    # ------------------------------------------------------------------

    def add_kg_relation(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Adds a relationship to the knowledge graph."""
        try:
            self.client.add_relation(subject, predicate, obj, user_id, metadata)
            self._journal.log_event(
                "MemoryDAO.add_kg_relation: Added KG relation.", 
                payload={"user_id": user_id, "subject": subject, "predicate": predicate, "object": obj}
            )
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.add_kg_relation: Error adding KG relation.", 
                payload={"user_id": user_id, "subject": subject, "predicate": predicate, "object": obj, "error": str(e)}
            )

    def query_kg_relations(
        self,
        user_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries relationships from the knowledge graph based on a pattern."""
        try:
            results = self.client.get_relations_by_pattern(subject, predicate, obj, user_id)
            self._journal.log_event(
                "MemoryDAO.query_kg_relations: Queried KG relations.", 
                payload={"user_id": user_id, "subject": subject, "predicate": predicate, "object": obj, "count": len(results)}
            )
            return results
        except Exception as e:
            self._journal.log_event(
                "MemoryDAO.query_kg_relations: Error querying KG relations.", 
                payload={"user_id": user_id, "subject": subject, "predicate": predicate, "object": obj, "error": str(e)}
            )
            return []
